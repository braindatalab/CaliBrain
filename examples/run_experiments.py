"""
.. _example-run_experiment:

=============================================
Run benchmark experiments for source localization and uncertainty estimation.
=============================================

This example demonstrates how to run comprehensive benchmarking experiments
for evaluating source localization algorithms using CaliBrain.

The script shows how to:
- Set up ERP simulation parameters and logging
- Configure parameter grids for MEG and EEG experiments  
- Compare multiple algorithms (Gamma-MAP, eLORETA)
- Evaluate uncertainty estimation and calibration performance
- Generate comprehensive metrics and save results
"""

import os
import datetime
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from calibrain import Benchmark, LeadfieldBuilder, MetricEvaluator, UncertaintyEstimator, SourceSimulator, SensorSimulator, gamma_map, eloreta

# https://github.com/mne-tools/mne-python/blob/main/mne/_fiff/constants.py
# print(fwd['info']['chs'][0]['unit'])  # Will show 107 (FIFF_UNIT_V)

def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results/benchmark_results", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    log_file = f"results/logs/benchmark_log_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode="w"),  # Overwrite log file each run
            logging.StreamHandler()                   # Also print to console
        ]
    )
    logger = logging.getLogger(__name__)

    # n_trials = 4
    ERP_config = {
        "tmin": -0.5,
        "tmax": 0.5,
        "stim_onset": 0,
        "sfreq": 250,
        "fmin": 1,
        "fmax": 5,
        "amplitude": 50.0,
        "random_erp_timing": True,
        "erp_min_length": None,
    }
    
    source_simulator = SourceSimulator(
        ERP_config=ERP_config,
        logger=logger
    )

    leadfield_builder = LeadfieldBuilder(
        leadfield_dir=Path("BSI-ZOO_forward_data"),
        logger=logger,
    )
    
    sensor_simulator = SensorSimulator(
        logger=logger,
    )

    confidence_levels = np.arange(0.0, 1.1, 0.1) # 11 levels: [0.0, 0.1, ..., 1.0]
    
    uncertainty_estimator = UncertaintyEstimator(
        confidence_levels=confidence_levels,
        logger=logger,
    )  
      
    # Define parameter grids for different data types
    data_param_grid_meg = {
        "subject": ["CC120166", "CC120264", "CC120309", "CC120313"],
        "nnz": [1, 10, 50, 100],
        "orientation_type": ["fixed"], # "fixed", "free"
        "alpha_SNR": [0.0, 0.2, 0.4, 0.6, 0.8, 0.99],
    }
    
    data_param_grid_eeg = {
        "subject": ["fsaverage"], # "caliBrain_fsaverage", "fsaverage",
        "nnz": [1, 10, 50, 100],
        "orientation_type": ["fixed"], # "fixed", "free"
        "alpha_SNR": [0.0, 0.2, 0.4, 0.6, 0.8, 0.99],
    }
        
    gamma_map_params = {
        "init_gamma": [0.001], #  0.001, 1.0, or tuple for random values (0.001, 0.1)   
        "noise_type": ["oracle"], # "baseline", "oracle", "joint_learning", "CV"
    }
    
    eloreta_params = {
        "noise_type": ["oracle"],
    }
    
    estimators = [
        (gamma_map, gamma_map_params, data_param_grid_meg),
        (eloreta, eloreta_params, data_param_grid_meg),
        (gamma_map, gamma_map_params, data_param_grid_eeg),
        (eloreta, eloreta_params, data_param_grid_eeg),
    ]

    metrics = [
        "mean_posterior_std",               # Uncertainty
        "mean_calibration_error",           # Calibration (auc)
        "max_underconfidence_deviation",    # Calibration
        "max_overconfidence_deviation",     # Calibration
        "mean_absolute_deviation",          # Calibration
        "mean_signed_deviation",            # Calibration
        "emd",                              # spatial accuracy
        "jaccard_error",                    # spatial accuracy
        "mse",                              # spatial accuracy
        "euclidean_distance",               # detection performance
        "f1",                               # detection performance
        "accuracy",                         # detection performance
    ]

    metric_evaluator = MetricEvaluator(
        confidence_levels=confidence_levels,
        metrics=metrics,
        logger=logger
    )

    nruns = 3
    df = []
    for solver, solver_param_grid, data_param_grid in estimators:
        benchmark = Benchmark(
            solver=solver,
            solver_param_grid=solver_param_grid,
            data_param_grid=data_param_grid,
            ERP_config=ERP_config,
            source_simulator=source_simulator,
            leadfield_builder=leadfield_builder,
            sensor_simulator=sensor_simulator,
            uncertainty_estimator=uncertainty_estimator,
            metric_evaluator=metric_evaluator,
            random_state=42,
            logger=logger
        )
        results_df = benchmark.run(nruns=nruns)
        df.append(results_df)

    results_df = pd.concat(df)
    results_df.to_csv(f"results/benchmark_results/benchmark_results_{timestamp}.csv", index=False)
    
    print(results_df.head())

if __name__ == "__main__":
    main()