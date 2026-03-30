import datetime
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import mne
import pandas as pd
from sklearn.model_selection import ParameterGrid

from calibrain import (
    Benchmark,
    LeadfieldBuilder,
    SourceSimulator,
    SensorSimulator,
    eloreta,
    gamma_map,
    BMN,
    BMN_joint,
    gamma_map_sflex,
    sflex_gamma_lambda_map,
)
from calibrain.utils import get_data_path
from calibrain.workflows.common import load_python_config

DEFAULT_CONFIG_PATH = Path("configs/benchmark_default.py")

_SOLVER_REGISTRY = {
    "gamma_map": gamma_map,
    "eloreta": eloreta,
    "BMN": BMN,
    "BMN_joint": BMN_joint,
    "gamma_map_sflex": gamma_map_sflex,
    "sflex_gamma_lambda_map": sflex_gamma_lambda_map,
}


def _resolve_solver(name: str):
    try:
        return _SOLVER_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown solver '{name}'. Available: {sorted(_SOLVER_REGISTRY)}"
        ) from exc


def run_benchmark(config: Union[str, Path, Dict[str, Any]]) -> Path:
    if isinstance(config, (str, Path)):
        config = load_python_config(config)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(config.get("log_dir", "results/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"benchmark_log_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )
    os.environ["CALIBRAIN_LOG_FILE"] = str(log_file)
    mne.set_log_level("ERROR")
    logging.getLogger("mne").setLevel(logging.ERROR)
    logger = logging.getLogger(__name__)

    nruns = int(config.get("nruns", 1))
    benchmark_n_jobs = int(config.get("benchmark_n_jobs", 1))
    random_state = int(config.get("random_state", 42))

    ERP_config = config.get("ERP_config", {})
    leadfield_dir = Path(
        config.get("leadfield_dir", get_data_path() / "1284src_leadfield")
    )
    leadfield_builder = LeadfieldBuilder(leadfield_dir=leadfield_dir, logger=logger)
    sensor_simulator = SensorSimulator(logger=logger)
    source_simulator = SourceSimulator(ERP_config=ERP_config, logger=logger)

    estimators_cfg: List[Dict[str, Any]] = config.get("estimators", [])
    if not estimators_cfg:
        raise ValueError("Benchmark config must define at least one estimator entry.")

    save_posterior_stats = bool(config.get("save_posterior_stats", True))

    results_dir = Path(config.get("results_dir", "results/benchmark_results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    results_prefix = config.get("results_filename_prefix", "benchmark_results")
    posterior_dir = Path(config.get("posterior_dir", "results/posterior_summaries"))
    posterior_dir.mkdir(parents=True, exist_ok=True)

    df_list = []
    config_counts = []
    for estimator in estimators_cfg:
        solver = _resolve_solver(estimator["solver"])
        entry = (
            solver,
            estimator.get("solver_params", {}),
            estimator.get("data_param_grid", {}),
            estimator.get("noise_param_grid", {}),
        )
        config_counts.append(entry)

    total_experiments = 0
    for _, solver_params, data_grid, noise_grid in config_counts:
        num_configs = (
            len(ParameterGrid(solver_params))
            * len(ParameterGrid(data_grid))
            * len(ParameterGrid(noise_grid))
        )
        total_experiments += nruns * max(1, num_configs)

    run_offset = 0
    for solver, solver_params, data_grid, noise_grid in config_counts:
        num_configs = (
            len(ParameterGrid(solver_params))
            * len(ParameterGrid(data_grid))
            * len(ParameterGrid(noise_grid))
        )
        total_local_runs = nruns * max(1, num_configs)
        benchmark = Benchmark(
            solver=solver,
            solver_param_grid=solver_params,
            data_param_grid=data_grid,
            noise_param_grid=noise_grid,
            ERP_config=ERP_config,
            source_simulator=source_simulator,
            leadfield_builder=leadfield_builder,
            sensor_simulator=sensor_simulator,
            save_posterior_stats=save_posterior_stats,
            posterior_dir=posterior_dir,
            random_state=random_state,
            logger=logger,
        )
        results_df = benchmark.run(
            nruns=nruns,
            n_jobs=benchmark_n_jobs,
            run_offset=run_offset,
            global_total_runs=total_experiments,
        )
        df_list.append(results_df)
        run_offset += total_local_runs

    if not df_list:
        raise RuntimeError("Benchmark produced no results data.")

    final_df = pd.concat(df_list)
    final_df.sort_values(
        by=[
            "run_id",
            "subject",
            "orientation_type",
            "nnz",
            "solver",
            "noise_type",
            "alpha_SNR",
        ],
        inplace=True,
        ascending=True,
    )
    final_df["nruns"] = nruns
    desired_cols = [
        "global_run_id",
        "run_id",
        "subject",
        "orientation_type",
        "nnz",
        "solver",
        "noise_type",
        "alpha_SNR",
        "gamma",
        "nruns",
    ]
    other_cols = [c for c in final_df.columns if c not in desired_cols]
    final_df = final_df[[c for c in desired_cols if c in final_df.columns] + other_cols]

    output_csv = results_dir / f"{results_prefix}_{timestamp}.csv"
    final_df.to_csv(output_csv, index=False)
    logger.info("Benchmark results written to %s", output_csv)
    return output_csv


if __name__ == "__main__":
    run_benchmark(DEFAULT_CONFIG_PATH)
