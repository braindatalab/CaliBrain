"""
.. _example-run_experiment:

=========================
Run benchmark experiments
=========================

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

from calibrain import Benchmark, LeadfieldBuilder, MetricEvaluator, UncertaintyEstimator, SourceSimulator, SensorSimulator, sflex_gamma_map, gamma_map, eloreta, BMN, sflex_gamma_lambda_map
from calibrain.utils import get_data_path

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
        "amplitude": 12.0,
        "random_erp_timing": True,
        "erp_min_length": None,
    }
    
    source_simulator = SourceSimulator(
        ERP_config=ERP_config,
        logger=logger
    )

    leadfield_dir = get_data_path()
    leadfield_builder = LeadfieldBuilder(
        leadfield_dir=leadfield_dir,
        logger=logger,
    )
    
    sensor_simulator = SensorSimulator(
        logger=logger,
    )

    nominal_coverages = np.arange(0.1, 1.1, 0.1)  # [0.1, 0.2, ..., 1.0] # 10 values
    uncertainty_estimator = UncertaintyEstimator(
        nominal_coverages=nominal_coverages,
        logger=logger,
    )  
      
    # ==================================================================
    # Define parameter grids for different data types
    # ==================================================================
    # MEG data parameters
    data_param_grid_meg = {
        "subject": ["CC120166"],# "CC120264", "CC120309", "CC120313"],
        "nnz": [5],
        "orientation_type": ["fixed"], # "fixed", "free"
        "alpha_SNR": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.99],
        "sensor_white_noise_var": [1.0],
    }
    
    # EEG data parameters
    # data_param_grid_eeg = {
    #     "subject": ["fsaverage"], # "caliBrain_fsaverage", "fsaverage",
    #     "nnz": [1, 10, 50],
    #     "orientation_type": ["fixed"], # "fixed", "free"
    #     "alpha_SNR": [0.0, 0.5, 0.99],
    #     "sensor_white_noise_var": [1.0],
    # }
    
    # =================================================================
    # Define noise parameter grids
    # =================================================================
    basic_noise_params = {
        "noise_type": ["oracle"], # "baseline", 
        # add noise parameters here if needed
    }
    
    default_alphas_grid = np.logspace(0, -2, 20)[1:]
    CV_noise_params = {
        "noise_type": ["spatial_cv", "temporal_cv"],
        'default_alphas_grid': [default_alphas_grid], # will be set within the benchmark loop based on baseline noise variance
        'cv': [2],
        'n_jobs': [10],
        # add noise parameters here if needed
    }

    adaptive_noise_params = {
        "noise_type": ["joint_learning"], # for sflex_gamma_lambda_map
        # add noise parameters here if needed
    }

    # ==================================================================
    # Define estimators and their parameter grids
    # ==================================================================
    eloreta_params = {
        # No specific hyperparameters to tune for eLORETA
    }

    BMN_params = {
        "max_iter": [1000],
        'normalization': [True]
    }
        
    sflex_gamma_map_params = {
        'init_gamma': [0.1],
        'sigma': [0.001],
        'max_iter': [1000],
        # fwd_path to each subject will be set within the benchmark loop (when instantiating SourceEstimator) after selecting the subject
    }
    
    # NOTE: sflex_gamma_lambda_map does not use noise variance from data simulation, but adaptively (noise learning) estimates noise variance along with source amplitude and location. So it ignores the noise_type in data_param_grid.
    sflex_gamma_lambda_map_params = {
        'init_gamma': [0.1], 
        'init_lambda': [0.1],
        'sigma': [0.001],
        'max_iter': [1000],
        # fwd_path to each subject will be set within the benchmark loop (when instantiating SourceEstimator) after selecting the subject
    }
    
    gamma_map_params = {
        'init_gamma': [0.1],
        'max_iter': [1000],
        'tol': [1e-15]
    }
    
    estimators = [
        # ================ MEG experiments ================
        # ---------------- eLORETA ----------------
        # (eloreta, eloreta_params, data_param_grid_meg, basic_noise_params),
        (eloreta, eloreta_params, data_param_grid_meg, CV_noise_params),
        # ---------------- BMN ----------------
        # (BMN, BMN_params, data_param_grid_meg, basic_noise_params),
        (BMN, BMN_params, data_param_grid_meg, CV_noise_params),
        # ---------------- sFLEX-Gamma-MAP ----------------
        # (sflex_gamma_map, sflex_gamma_map_params, data_param_grid_meg, basic_noise_params),
        (sflex_gamma_map, sflex_gamma_map_params, data_param_grid_meg, CV_noise_params),
        # ---------------- sFLEX-Gamma-Lambda-MAP ----------------
        # (sflex_gamma_lambda_map, sflex_gamma_lambda_map_params, data_param_grid_meg, adaptive_noise_params),
        # ---------------- Gamma-MAP ----------------
        # (gamma_map, gamma_map_params, data_param_grid_meg, basic_noise_params),
        (gamma_map, gamma_map_params, data_param_grid_meg, CV_noise_params),
        

        # ================ EEG experiments ================
        # ---------------- eLORETA ----------------
        # (eloreta, eloreta_params, data_param_grid_eeg, basic_noise_params),
        # (eloreta, eloreta_params, data_param_grid_eeg, CV_noise_params),
        # ---------------- BMN ----------------
        # (BMN, BMN_params, data_param_grid_eeg, basic_noise_params),
        # (BMN, BMN_params, data_param_grid_eeg, CV_noise_params),
        # ---------------- sFLEX-Gamma-MAP ----------------
        # (sflex_gamma_map, sflex_gamma_map_params, data_param_grid_eeg, basic_noise_params),
        # (sflex_gamma_map, sflex_gamma_map_params, data_param_grid_eeg, CV_noise_params),
        # ---------------- sFLEX-Gamma-Lambda-MAP ----------------
        # (sflex_gamma_lambda_map, sflex_gamma_lambda_map_params, data_param_grid_eeg, adaptive_noise_params),
    ]

    metrics = [
        "mean_posterior_std",               # Uncertainty
        
        "mean_calibration_error",           # Calibration (auc)
        "mean_signed_deviation",            # Calibration
        "mean_absolute_deviation",          # Calibration
        "max_underconfidence_deviation",    # Calibration
        "max_overconfidence_deviation",     # Calibration
        
        "emd",                              # spatial accuracy
        "jaccard_error",                    # spatial accuracy
        "mse",                              # spatial accuracy
        "euclidean_distance",               # detection performance
        "f1",                               # detection performance
        "accuracy",                         # detection performance
    ]

    metric_evaluator = MetricEvaluator(
        confidence_levels=nominal_coverages,
        nominal_coverages= 1 - nominal_coverages,
        metrics=metrics,
        logger=logger
    )

    nruns = 1
    df = []
    for solver, solver_param_grid, data_param_grid, noise_param_grid in estimators:
        benchmark = Benchmark(
            solver=solver,
            solver_param_grid=solver_param_grid,
            data_param_grid=data_param_grid,
            noise_param_grid=noise_param_grid,
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
    results_df.sort_values(by=['subject', 'nnz','solver', 'noise_type',  'alpha_SNR', 'gamma'],inplace=True, ascending=True)
    desired_order = ['run_id', 'subject', 'orientation_type', 'nnz','solver', 'noise_type',  'alpha_SNR', 'gamma', 'noise_var', 'mean_posterior_variance']
    cols = [c for c in desired_order if c in results_df.columns] + \
           [c for c in results_df.columns if c not in desired_order]
    results_df = results_df[cols]
    results_df.to_csv(f"results/benchmark_results/benchmark_results_{timestamp}.csv", index=False)
    
    print(results_df.head())

if __name__ == "__main__":
    main()