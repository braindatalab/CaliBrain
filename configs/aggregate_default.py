"""
Per-run aggregation preset.

Each run of ``calibrain/workflows/aggregation.py`` will:
  * scan ``summaries_root`` for posterior_summary*.h5/.npz files
  * optionally filter them by metadata
  * write one calibration dataset (NPZ + JSON metadata) per summary/run
into the configured ``output_dir``.
"""

from pathlib import Path

from mne.io.constants import FIFF

RUN_SPLIT_THRESHOLD = 10  # runs <= threshold for train; >threshold for eval
SUMMARIES_ROOT = "results/posterior_summaries"
BASE_OUTPUT_DIR = Path("results/calibration_datasets_per_run")


def train_run_filter(rid, threshold=RUN_SPLIT_THRESHOLD):
    return rid is not None and int(rid) <= threshold

def eval_run_filter(rid, threshold=RUN_SPLIT_THRESHOLD):
    return rid is not None and int(rid) > threshold

EXPERIMENT_NAME = "POST_BMN_ORACLE_FIXED"
# "subject": ["CC120166", "CC120264", "CC120309", "CC120313"],

CONFIGS = {
    "train": {
        "summaries_root": SUMMARIES_ROOT,
        "output_dir": str(BASE_OUTPUT_DIR / "train"),
        "filter": {
            "alpha_SNR": [0.5],
            "nnz": [5],
            "solver": ["BMN"],
            "noise_type": ["oracle"],
            # "subject": ["CC120166"],
            "orientation_type": ["fixed"],
            "coil_type": [FIFF.FIFFV_COIL_VV_MAG_T1],
            "run_id": train_run_filter,
        },
    },
    "eval": {
        "summaries_root": SUMMARIES_ROOT,
        "output_dir": str(BASE_OUTPUT_DIR / "eval"),
        "filter": {
            "alpha_SNR": [0.5],
            "nnz": [5],
            "solver": ["BMN"],
            "noise_type": ["oracle"],
            # "subject": ["CC120166", "CC120264", "CC120309", "CC120313"],
            "orientation_type": ["fixed"],
            "coil_type": [FIFF.FIFFV_COIL_VV_MAG_T1],
            "run_id": eval_run_filter,
        },
    },
}

CONFIG = {
    "splits": CONFIGS,
}
