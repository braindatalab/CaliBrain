# Per-run calibration configuration.
# This script pools every NPZ file in ``TRAIN_DIR`` for training and then
# evaluates the calibrated model on each NPZ in ``EVAL_DIR`` individually.

from pathlib import Path

TRAIN_DIR = Path("results/calibration_datasets_per_run/train")
EVAL_DIR = Path("results/calibration_datasets_per_run/eval")

CONFIG = {
    # Pooled training data: every *.npz file inside TRAIN_DIR will be merged.
    "train_dir": str(TRAIN_DIR),
    "train_pattern": "*.npz",
    
    # Evaluation data: iterate over each *.npz inside EVAL_DIR.
    "eval_dir": str(EVAL_DIR),
    "eval_pattern": "*.npz",
    
    # Optionally limit the number of eval files processed (set to None for all).
    "eval_limit": None,
    
    # Output + visualization controls.
    "output_dir": "results/calibration_eval",
    "run_name": "POST_BMN_ORACLE_FIXED",
    "plot_curve": True,
    "plot_nominal": 0.95,
    "plot_source_idx": 0,
    "nominal_coverages": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}
