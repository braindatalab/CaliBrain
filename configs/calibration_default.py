"""
Per-run calibration configuration with selectable experiment settings.

Experiment definitions (what this config runs)

- ``precal`` (Pre-calibration / raw uncertainty):
    Skip isotonic regression fitting entirely (``fit_calibration=False``). Each
    eval NPZ is evaluated directly and contributes one pre-calibration curve;
    aggregated curves can be computed later from the JSON outputs.

- ``post_oracle`` (Post-calibration, matched condition):
    Fit one isotonic model per head using that head's 10 calibration runs at
    the test condition (SNR=0.5, NNZ=5), then evaluate on that head's 25 test
    runs at the same condition.

- ``post_pooled`` (Post-calibration, pooled within head):
    Fit one isotonic model per head using that head's calibration runs pooled
    across all SNR×NNZ settings, then evaluate on that head's 25 test runs at
    the test condition (SNR=0.5, NNZ=5).

- ``post_pooled_mismatch`` (Post-calibration, pooled across other heads):
    For held-out head H, fit an isotonic model using pooled calibration runs
    from the other 3 heads (all SNR×NNZ settings), then evaluate on H's 25 test
    runs at the test condition.

How to run (no env vars, no CLI changes):

0) First run aggregation using ``configs/aggregate_default.py`` for the same
   SELECT_ORIENTATION / SELECT_EXPERIMENT (and SELECT_HELDOUT if applicable).

1) Choose what to calibrate/evaluate by editing:
   - SELECT_ORIENTATION: "fixed" or "free"
   - SELECT_EXPERIMENT: "precal" | "post_oracle" | "post_pooled" | "post_pooled_mismatch" | "post_fixed"
   - SOLVER_NOISE_PAIRS: list of (solver, noise_type) tuples to include
   - AGG_ROOT: must match where aggregation wrote datasets


Optional (free orientation only):
- Set `free_interval_type` to "full_cov" (default; ellipsoid/ellipse) or "marginal" (per component).

2) Run calibration:
   ``python calibrain/workflows/calibration.py``

What to change between runs:
  - Pre-calibration:
      SELECT_EXPERIMENT = "precal"
      Run once per orientation. (This sets fit_calibration=False automatically.)

  - Post-oracle:
      SELECT_EXPERIMENT = "post_oracle"
      Run once per orientation (script loops all solver×noise×heads).

  - Post-pooled:
      SELECT_EXPERIMENT = "post_pooled"
      Run once per orientation.

  - Post-pooled-mismatch:
      SELECT_EXPERIMENT = "post_pooled_mismatch"
      Run once per orientation.

  - Post-fixed:
      SELECT_EXPERIMENT = "post_fixed"
      SELECT_SWEEP = "snr" (vary alpha_SNR with nnz fixed) or "nnz" (vary nnz with alpha_SNR fixed)
      Run once per orientation.

This is designed to pair with the directory layout produced by
``configs/aggregate_default.py`` in this repo (per-head train/eval folders).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Manual selector (edit these)
# ---------------------------------------------------------------------------

SELECT_ORIENTATION = "free"  # "fixed" or "free"
SELECT_EXPERIMENT = "post_fixed"  # "precal" | "post_oracle" | "post_pooled" | "post_pooled_mismatch" | "post_fixed"

# Only used when SELECT_EXPERIMENT == "post_fixed".
# - "snr": evaluate across SNR sweep with nnz fixed
# - "nnz": evaluate across NNZ sweep with alpha_SNR fixed
SELECT_SWEEP = "snr"  # "snr" | "nnz"

# Keep this list in sync with `configs/aggregate_default.py:SELECT_SOLVER_NOISE_PAIRS`.
SOLVER_NOISE_PAIRS = [
    ("BMN", "oracle"),
    ("BMN", "baseline"),
    ("BMN_joint", "adaptive_joint_learning"),
    ("gamma_map_sflex", "oracle"),
    ("gamma_map_sflex", "baseline"),
    ("gamma_lambda_map_sflex", "adaptive_joint_learning"),
]

# ---------------------------------------------------------------------------
# Aggregation output root (edit to match your aggregation config)
# ---------------------------------------------------------------------------

AGG_ROOT = Path("/data/orabem/calibrain/results/aggregated_data")

HEADS = ("CC120166", "CC120264", "CC120309", "CC120313")


def _single_run_config(*, solver: str, noise: str, head: str | None) -> dict:
    base_dir = AGG_ROOT / SELECT_ORIENTATION / solver / noise / SELECT_EXPERIMENT
    fit_calibration = SELECT_EXPERIMENT != "precal"
    if SELECT_EXPERIMENT == "precal":
        train_dir = None
        eval_dir = base_dir / "eval"
        tag = "all"
    elif SELECT_EXPERIMENT in {"post_oracle", "post_pooled"}:
        if head is None:
            raise ValueError("head is required for post_oracle/post_pooled")
        train_dir = base_dir / head / "train"
        eval_dir = base_dir / head / "eval"
        tag = head
    elif SELECT_EXPERIMENT == "post_pooled_mismatch":
        if head is None:
            raise ValueError("head (held-out) is required for post_pooled_mismatch")
        train_dir = base_dir / head / "train"
        eval_dir = base_dir / head / "eval"
        tag = head
    elif SELECT_EXPERIMENT == "post_fixed":
        if head is None:
            raise ValueError("head is required for post_fixed")
        sweep = str(SELECT_SWEEP).lower()
        if sweep not in {"snr", "nnz"}:
            raise ValueError("SELECT_SWEEP must be 'snr' or 'nnz' when SELECT_EXPERIMENT == 'post_fixed'.")
        train_dir = base_dir / head / "train"
        eval_dir = base_dir / head / ("eval_snr" if sweep == "snr" else "eval_nnz")
        tag = f"{head}__{sweep}"
    else:
        raise ValueError(f"Unknown SELECT_EXPERIMENT: {SELECT_EXPERIMENT}")

    output_dir = (
        Path("/data/orabem/calibrain/results/calibration_eval")
        / SELECT_ORIENTATION
        / solver
        / noise
        / SELECT_EXPERIMENT
        / tag
    )
    cfg = {
        "fit_calibration": fit_calibration,
        "fit_once": bool(SELECT_EXPERIMENT == "post_fixed"),
        "eval_dir": str(eval_dir),
        "eval_pattern": "*.npz",
        "eval_limit": None,
        "emd_mode": "lifted",
        "output_dir": str(output_dir),
        "run_name": f"{SELECT_EXPERIMENT}__{SELECT_ORIENTATION}__{solver}__{noise}__{tag}",
        "plot_curve": False,
        "plot_nominal": 0.95,
        "plot_source_idx": 0,
        "free_interval_type": "full_cov",  # or "marginal"
        "nominal_coverages": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    }
    if fit_calibration:
        cfg["train_dir"] = str(train_dir)
        cfg["train_pattern"] = "*.npz"
    return cfg


runs = {}
for solver, noise in SOLVER_NOISE_PAIRS:
    if SELECT_EXPERIMENT == "precal":
        key = f"{solver}__{noise}__precal"
        runs[key] = _single_run_config(solver=solver, noise=noise, head=None)
    elif SELECT_EXPERIMENT in {"post_oracle", "post_pooled", "post_fixed"}:
        for head in HEADS:
            key = f"{solver}__{noise}__{head}__{SELECT_EXPERIMENT}"
            runs[key] = _single_run_config(solver=solver, noise=noise, head=head)
    elif SELECT_EXPERIMENT == "post_pooled_mismatch":
        for heldout in HEADS:
            key = f"{solver}__{noise}__{heldout}__mismatch"
            runs[key] = _single_run_config(solver=solver, noise=noise, head=heldout)
    else:
        raise ValueError(f"Unknown SELECT_EXPERIMENT: {SELECT_EXPERIMENT}")

CONFIG = {"runs": runs}
