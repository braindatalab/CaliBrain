from copy import deepcopy


RUN_PARAMS = {
    "nruns": 10 + 25,
    "benchmark_n_jobs": 1,
    "random_state": 42,
}

PATHS = {
    "log_dir": "results/logs",
    "results_dir": "results/benchmark_results",
    "results_filename_prefix": "benchmark_results",
    "posterior_dir": "results/posterior_summaries",
}

UNCERTAINTY = {
    "nominal_coverages": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "save_posterior_stats": True,
}

ERP_CONFIG = {
    "tmin": -0.5,
    "tmax": 0.5,
    "stim_onset": 0.0,
    "sfreq": 250,
    "fmin": 1,
    "fmax": 5,
    "amplitude_distribution": {
        "median": 20.0,
        "sigma": 0.2,
        "clip": [2.5, 50.0],
    },
    "random_erp_timing": True,
    "erp_min_length": None,
}

COMMON_DATA_GRID = {
    "subject": ["CC120166", "CC120264", "CC120309", "CC120313"],
    "nnz": [5], # [5, 10, 100],
    "orientation_type": ["fixed"],
    "alpha_SNR": [0.5],
    "sensor_white_noise_std": [0.001],
}

BASIC_NOISE = {"noise_type": ["oracle"]}
ADAPTIVE_NOISE = {"noise_type": ["adaptive_joint_learning"]}

def _estimator(solver: str, solver_params: dict, noise_grid: dict) -> dict:
    return {
        "solver": solver,
        "solver_params": solver_params,
        "data_param_grid": deepcopy(COMMON_DATA_GRID),
        "noise_param_grid": deepcopy(noise_grid),
    }


ESTIMATORS = [
    _estimator("BMN", {"max_iter": [1000], "normalization": [True]}, BASIC_NOISE),
    # _estimator("BMN_joint", {
    #     "max_iter": [1000],
    #     "normalization": [True],
    #     "learn_noise":[True]
    #     }, ADAPTIVE_NOISE),
    # _estimator("BMN_joint", {"learn_noise": [True]}, ADAPTIVE_NOISE),
    # _estimator(
    #     "gamma_map_sflex",
    #     {"init_gamma": [0.1], "sigma": [0.001], "max_iter": [1000]},
    #     BASIC_NOISE,
    # ),
    # _estimator(
    #     "sflex_gamma_lambda_map",
    #     {"sigma": [0.001], "max_iter": [100], "learn_lambda": [True]},
    #     ADAPTIVE_NOISE,
    # ),
]


CONFIG = {
    **RUN_PARAMS,
    **PATHS,
    **UNCERTAINTY,
    "ERP_config": ERP_CONFIG,
    "estimators": ESTIMATORS,
}
