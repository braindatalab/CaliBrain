"""
Per-run aggregation preset with selectable experiment settings.

Each run of ``calibrain/workflows/aggregation.py`` will:
  * read the run manifest CSV (no filesystem scanning)
  * filter runs by metadata stored in the manifest
  * write one calibration dataset (NPZ + JSON metadata) per summary/run
into the configured ``output_dir``.

Experiment definitions (what this file produces)

- ``precal`` (Pre-calibration / raw uncertainty):
    No isotonic model is fitted later. We aggregate only the 25 test runs per
    head at the test condition (SNR=0.5, NNZ=5), i.e. run_id 11..35. This
    yields 4×25 = 100 per-run NPZ files per orientation.

- ``post_oracle`` (Post-calibration, matched condition):
    For each head, aggregate a training pool of 10 calibration runs from the
    *matching* head at the same test condition (SNR=0.5, NNZ=5), i.e. run_id
    1..10, and an evaluation split of 25 test runs (run_id 11..35). Intended
    to fit 4 isotonic models (one per head) later.

- ``post_pooled`` (Post-calibration, pooled within head):
    For each head, aggregate a training pool using the 10 calibration runs
    (run_id 1..10) pooled over all SNR×NNZ settings (5×5×10 = 250 runs per
    head). Evaluate on the 25 test runs at the test condition (SNR=0.5, NNZ=5,
    run_id 11..35). Intended to fit 4 isotonic models (one per head) later.

- ``post_pooled_mismatch`` (Post-calibration, pooled across other heads):
    For a chosen held-out head H, aggregate a training pool using the 10
    calibration runs (run_id 1..10) pooled over all SNR×NNZ settings but using
    only the *other 3 heads* (3×5×5×10 = 750 runs). Evaluate on head H's 25
    test runs at the test condition (SNR=0.5, NNZ=5, run_id 11..35).

How to run (no env vars, no CLI changes):

1) Choose what to aggregate by editing:
   - SELECT_ORIENTATION: "fixed" or "free"
   - SELECT_EXPERIMENT: "precal" | "post_oracle" | "post_pooled" | "post_pooled_mismatch"
   - SELECT_SOLVER_NOISE_PAIRS: list of (solver, noise_type) tuples to include
   - (and update MANIFEST_PATH / BASE_OUTPUT_ROOT / COIL_TYPE if needed)

2) Run aggregation:
   ``python calibrain/workflows/aggregation.py``

What to change between runs (typical sequence):
  - Pre-calibration (raw uncertainty, eval-only):
      SELECT_EXPERIMENT = "precal"
      Run once per orientation (fixed, free). This produces all solver×noise panels in one run.

  - Post-oracle (per-head train/eval datasets at SNR=0.5, NNZ=5):
      SELECT_EXPERIMENT = "post_oracle"
      Run once per orientation (fixed, free). This produces all solver×noise panels in one run.

  - Post-pooled (per-head train pooled over all SNR/NNZ; eval at SNR=0.5, NNZ=5):
      SELECT_EXPERIMENT = "post_pooled"
      Run once per orientation (fixed, free). This produces all solver×noise panels in one run.

  - Post-pooled-mismatch (train pooled over other 3 heads; eval on held-out head):
      SELECT_EXPERIMENT = "post_pooled_mismatch"
      Run once per orientation (fixed, free). This produces all 4 held-out heads for every solver×noise panel.

This file supports the paper-style aggregation experiments:
  - precal: evaluate raw uncertainty (no isotonic fit later)
  - post_oracle: 10 calibration runs at (SNR=0.5, NNZ=5), test on 25 runs same condition
  - post_pooled: 10 calibration runs pooled over all SNR/NNZ, test at (SNR=0.5, NNZ=5)
  - post_pooled_mismatch: hold out one head for test; calibrate on the other three pooled
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from mne.io.constants import FIFF

# ---------------------------------------------------------------------------
# Manual selector (edit these)
# ---------------------------------------------------------------------------

SELECT_ORIENTATION = "free"  # "fixed" or "free"
SELECT_EXPERIMENT = "post_pooled_mismatch"  # "precal" | "post_oracle" | "post_pooled" | "post_pooled_mismatch"

# Run all solver×noise pairs defined in configs/data_generation_default.py.
# Define the solver×noise pairs you want to aggregate in *this* run.
# Keep this in sync with the solvers/noise methods you generated in the
# data-generation stage (see `configs/data_generation_default.py`).
SELECT_SOLVER_NOISE_PAIRS: List[Tuple[str, str]] = [
    ("BMN", "oracle"),
    ("BMN", "baseline"),
    ("BMN_joint", "adaptive_joint_learning"),
    ("gamma_map_sflex", "oracle"),
    ("gamma_map_sflex", "baseline"),
    ("gamma_lambda_map_sflex", "adaptive_joint_learning"),
]

# ---------------------------------------------------------------------------
# Experiment constants (paper defaults)
# ---------------------------------------------------------------------------

HEADS = ["CC120166", "CC120264", "CC120309", "CC120313"]

CAL_RUNS = 10
TEST_RUNS = 25
TEST_SNR = 0.5
TEST_NNZ = 5

SNR_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]
NNZ_GRID = [1, 3, 5, 10, 100]

# ---------------------------------------------------------------------------
# Paths + common filters (edit these to match your environment)
# ---------------------------------------------------------------------------

MANIFEST_PATH = Path("/data/orabem/calibrain/results/run_manifest") / f"{SELECT_ORIENTATION}.csv"
BASE_OUTPUT_ROOT = Path("/data/orabem/calibrain/results/aggregated_data") / SELECT_ORIENTATION

# Keep these restrictive to avoid mixing multiple solver/noise regimes into the
# same aggregated dataset directory.
COIL_TYPE = [FIFF.FIFFV_COIL_VV_MAG_T1]


def _run_id_inclusive(min_id: int, max_id: int) -> Callable[[Any], bool]:
    """True for run_id values within the closed interval [min_id, max_id]."""

    def _predicate(rid: Any) -> bool:
        if rid is None:
            return False
        return min_id <= int(rid) <= max_id

    _predicate.__doc__ = f"run_id in [{min_id}, {max_id}]"
    return _predicate


def _split(*, output_dir: Path, extra_filter: Dict[str, Any]) -> Dict[str, Any]:
    criteria: Dict[str, Any] = {
        "orientation_type": [SELECT_ORIENTATION],
        "coil_type": COIL_TYPE,
    }
    criteria.update(extra_filter)
    return {
        "manifest_path": str(MANIFEST_PATH),
        "output_dir": str(output_dir),
        "filter": criteria,
    }


CAL_FILTER = _run_id_inclusive(1, CAL_RUNS)
TEST_FILTER = _run_id_inclusive(CAL_RUNS + 1, CAL_RUNS + TEST_RUNS)

def _add_experiment_splits(
    *,
    splits: Dict[str, Dict[str, Any]],
    solver: str,
    noise_type: str,
) -> None:
    base_output_dir = BASE_OUTPUT_ROOT / solver / noise_type / SELECT_EXPERIMENT
    base_filter = {"solver": [solver], "noise_type": [noise_type]}

    if SELECT_EXPERIMENT == "precal":
        splits[f"{solver}__{noise_type}__eval"] = _split(
            output_dir=base_output_dir / "eval",
            extra_filter={
                **base_filter,
                "subject": HEADS,
                "alpha_SNR": [TEST_SNR],
                "nnz": [TEST_NNZ],
                "run_id": TEST_FILTER,
            },
        )
        return

    if SELECT_EXPERIMENT == "post_oracle":
        for head in HEADS:
            splits[f"{solver}__{noise_type}__{head}__train"] = _split(
                output_dir=base_output_dir / head / "train",
                extra_filter={
                    **base_filter,
                    "subject": [head],
                    "alpha_SNR": [TEST_SNR],
                    "nnz": [TEST_NNZ],
                    "run_id": CAL_FILTER,
                },
            )
            splits[f"{solver}__{noise_type}__{head}__eval"] = _split(
                output_dir=base_output_dir / head / "eval",
                extra_filter={
                    **base_filter,
                    "subject": [head],
                    "alpha_SNR": [TEST_SNR],
                    "nnz": [TEST_NNZ],
                    "run_id": TEST_FILTER,
                },
            )
        return

    if SELECT_EXPERIMENT == "post_pooled":
        for head in HEADS:
            splits[f"{solver}__{noise_type}__{head}__train"] = _split(
                output_dir=base_output_dir / head / "train",
                extra_filter={
                    **base_filter,
                    "subject": [head],
                    "alpha_SNR": SNR_GRID,
                    "nnz": NNZ_GRID,
                    "run_id": CAL_FILTER,
                },
            )
            splits[f"{solver}__{noise_type}__{head}__eval"] = _split(
                output_dir=base_output_dir / head / "eval",
                extra_filter={
                    **base_filter,
                    "subject": [head],
                    "alpha_SNR": [TEST_SNR],
                    "nnz": [TEST_NNZ],
                    "run_id": TEST_FILTER,
                },
            )
        return

    if SELECT_EXPERIMENT == "post_pooled_mismatch":
        # Build all held-out heads in one run (so one aggregation call produces
        # all 4 mismatch models' train/eval pools for this solver/noise combo).
        for heldout in HEADS:
            other_heads = [h for h in HEADS if h != heldout]
            splits[f"{solver}__{noise_type}__{heldout}__train"] = _split(
                output_dir=base_output_dir / heldout / "train",
                extra_filter={
                    **base_filter,
                    "subject": other_heads,
                    "alpha_SNR": SNR_GRID,
                    "nnz": NNZ_GRID,
                    "run_id": CAL_FILTER,
                },
            )
            splits[f"{solver}__{noise_type}__{heldout}__eval"] = _split(
                output_dir=base_output_dir / heldout / "eval",
                extra_filter={
                    **base_filter,
                    "subject": [heldout],
                    "alpha_SNR": [TEST_SNR],
                    "nnz": [TEST_NNZ],
                    "run_id": TEST_FILTER,
                },
            )
        return

    raise ValueError(
        "Unknown SELECT_EXPERIMENT. Expected one of: "
        "'precal', 'post_oracle', 'post_pooled', 'post_pooled_mismatch'."
    )


SPLITS: Dict[str, Dict[str, Any]] = {}
for _solver, _noise in SELECT_SOLVER_NOISE_PAIRS:
    _add_experiment_splits(splits=SPLITS, solver=_solver, noise_type=_noise)


CONFIG = {"splits": SPLITS}
