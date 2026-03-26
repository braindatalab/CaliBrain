"""
Aggregation presets.

Metadata keys available for filtering:
    solver, noise_type, subject, orientation_type, nnz, alpha_SNR,
    seed, run_in_config, global_run_id, nruns,
    experiment_dir, posterior_dir, posterior_filename
"""

from pathlib import Path
from typing import Dict

SUMMARIES_ROOT = "results/posterior_summaries"
BASE_OUTPUT_DIR = Path("results/calibration_datasets")

RUN_SPLIT_THRESHOLD = 10  # Inclusive: run_id <= threshold is train, > threshold is test.

RUN_FILTERS = {
    "train": lambda rid: rid is not None and 1 <= int(rid) <= RUN_SPLIT_THRESHOLD,
    "test": lambda rid: rid is not None and int(rid) > RUN_SPLIT_THRESHOLD,
}


def _paths(tag: str) -> Dict[str, str]:
    base = BASE_OUTPUT_DIR / tag
    return {
        "output_path": str(base / "posterior_dataset.npz"),
        "train_output": str(base / "posterior_dataset_train.npz"),
        "test_output": str(base / "posterior_dataset_test.npz"),
    }


# Fixed SNR and NNZ, vary solver/noise combinations.
CONFIGS = {
    "POST_ORACLE_BMN_oracle": {
        "summaries_root": SUMMARIES_ROOT,
        **_paths("POST_ORACLE_BMN_oracle"),
        "train_filter": {
            "alpha_SNR": [0.5],
            "nnz": [5],
            "solver": ["BMN"],
            "noise_type": ["oracle"],
            "run_id": RUN_FILTERS["train"],
        },
        "test_filter": {
            "alpha_SNR": [0.5],
            "nnz": [5],
            "solver": ["BMN"],
            "noise_type": ["oracle"],
            "run_id": RUN_FILTERS["test"],
        },
    },
    # "BMN_baseline": {
    #     "summaries_root": SUMMARIES_ROOT,
    #     **_paths("BMN_baseline"),
    #     "train_filter": {
    #         "alpha_SNR": [0.5],
    #         "nnz": [3],
    #         "solver": ["BMN"],
    #         "noise_type": ["baseline"],
    #         "run_id": RUN_FILTERS["train"],
    #     },
    #     "test_filter": {
    #         "alpha_SNR": [0.5],
    #         "nnz": [3],
    #         "solver": ["BMN"],
    #         "noise_type": ["baseline"],
    #         "run_id": RUN_FILTERS["test"],
    #     },
    # },
    # "BMN_joint_adaptive_joint_learning": {
    #     "summaries_root": SUMMARIES_ROOT,
    #     **_paths("BMN_joint_adaptive_joint_learning"),
    #     "train_filter": {
    #         "alpha_SNR": [0.5],
    #         "nnz": [3],
    #         "solver": ["BMN_joint"],
    #         "noise_type": ["adaptive_joint_learning"],
    #         "run_id": RUN_FILTERS["train"],
    #     },
    #     "test_filter": {
    #         "alpha_SNR": [0.5],
    #         "nnz": [3],
    #         "solver": ["BMN_joint"],
    #         "noise_type": ["adaptive_joint_learning"],
    #         "run_id": RUN_FILTERS["test"],
    #     },
    # },
    # "gamma_map_sflex_oracle": {
    #     "summaries_root": SUMMARIES_ROOT,
    #     **_paths("gamma_map_sflex_oracle"),
    #     "train_filter": {
    #         "alpha_SNR": [0.5],
    #         "nnz": [3],
    #         "solver": ["gamma_map_sflex"],
    #         "noise_type": ["oracle"],
    #         "run_id": RUN_FILTERS["train"],
    #     },
    #     "test_filter": {
    #         "alpha_SNR": [0.5],
    #         "nnz": [3],
    #         "solver": ["gamma_map_sflex"],
    #         "noise_type": ["oracle"],
    #         "run_id": RUN_FILTERS["test"],
    #     },
    # },
    # "gamma_map_sflex_baseline": {
    #     "summaries_root": SUMMARIES_ROOT,
    #     **_paths("gamma_map_sflex_baseline"),
    #     "train_filter": {
    #         "alpha_SNR": [0.5],
    #         "nnz": [3],
    #         "solver": ["gamma_map_sflex"],
    #         "noise_type": ["baseline"],
    #         "run_id": RUN_FILTERS["train"],
    #     },
    #     "test_filter": {
    #         "alpha_SNR": [0.5],
    #         "nnz": [3],
    #         "solver": ["gamma_map_sflex"],
    #         "noise_type": ["baseline"],
    #         "run_id": RUN_FILTERS["test"],
    #     },
    # },
    # "sflex_gamma_lambda_map_adaptive_joint_learning": {
    #     "summaries_root": SUMMARIES_ROOT,
    #     **_paths("sflex_gamma_lambda_map_adaptive_joint_learning"),
    #     "train_filter": {
    #         "alpha_SNR": [0.5],
    #         "nnz": [1],
    #         "solver": ["sflex_gamma_lambda_map"],
    #         "noise_type": ["adaptive_joint_learning"],
    #         "run_id": RUN_FILTERS["train"],
    #     },
    #     "test_filter": {
    #         "alpha_SNR": [0.5],
    #         "nnz": [3],
    #         "solver": ["sflex_gamma_lambda_map"],
    #         "noise_type": ["adaptive_joint_learning"],
    #         "run_id": RUN_FILTERS["test"],
    #     },
    # },
}

# Pick the desired configuration here
CONFIG = CONFIGS["POST_ORACLE_BMN_oracle"]
