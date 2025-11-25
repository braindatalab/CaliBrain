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
os.environ.setdefault("MPLBACKEND", "Agg")
import datetime
import logging
import numpy as np
import pandas as pd
import mne

from calibrain import Benchmark, LeadfieldBuilder, MetricEvaluator, UncertaintyEstimator, SourceSimulator, SensorSimulator, sflex_gamma_map, gamma_map, eloreta, BMN, sflex_gamma_lambda_map
from calibrain.utils import get_data_path

def main():
    start_time = datetime.datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
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
    mne.set_log_level('ERROR')
    logging.getLogger('mne').setLevel(logging.ERROR)
    logger = logging.getLogger(__name__)

    # n_trials = 4
    ERP_config = {
        "tmin": -0.5,
        "tmax": 0.5,
        "stim_onset": 0,
        "sfreq": 250,
        "fmin": 1,
        "fmax": 5,
        "amplitude_distribution": {
            "median": 20.0,   # nAm
            "sigma": 0.2,
            "clip": (2.5, 50.0),
        },
        "random_erp_timing": True,
        "erp_min_length": None,
    }
    
    source_simulator = SourceSimulator(
        ERP_config=ERP_config,
        logger=logger
    )

    leadfield_dir = get_data_path() / "leadfield"
    leadfield_builder = LeadfieldBuilder(
        leadfield_dir=leadfield_dir,
        logger=logger,
    )
    
    sensor_simulator = SensorSimulator(
        logger=logger,
    )

    # nominal coverage levels   
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
        "subject": ["CC120166", "CC120264", "CC120309", "CC120313"],
        #"subject": ["CC120166"],
        # "nnz": [1, 5, 10],
        "nnz": [5],
        "orientation_type": ["fixed"], # "fixed", "free"
        "alpha_SNR": [0.1, 0.4, 0.7, 0.99],
        #"alpha_SNR": [ 0.7],
        "sensor_white_noise_var": [1.0 * 0.001],
    }
    
    # EEG data parameters
    data_param_grid_eeg = {
        "subject": ["fsaverage"], # "caliBrain_fsaverage", "fsaverage",
        "nnz": [1, 5, 10],
        "orientation_type": ["fixed"], # "fixed", "free"
        "alpha_SNR": [0.1, 0.4, 0.7, 0.99],
        "sensor_white_noise_var": [1.0 * 0.001],
    }
    
    # =================================================================
    # Define noise parameter grids
    # =================================================================
    basic_noise_params = {
        "noise_type": ["oracle", 'baseline'], 
        # add noise parameters here if needed
    }
    
    default_alphas_grid = np.logspace(0, -2, 10)[1:]
    CV_noise_params = {
        "noise_type": ["temporal_cv"],
        'default_alphas_grid': [default_alphas_grid], # will be set within the benchmark loop based on baseline noise variance
        'cv': [2],
        'n_jobs': [1],
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
        "max_iter": [500],
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
        # (eloreta, eloreta_params, data_param_grid_meg, CV_noise_params),
        # ---------------- BMN ----------------
         (BMN, BMN_params, data_param_grid_meg, basic_noise_params),
        # (BMN, BMN_params, data_param_grid_meg, CV_noise_params),
        # ---------------- sFLEX-Gamma-MAP ----------------
         (sflex_gamma_map, sflex_gamma_map_params, data_param_grid_meg, basic_noise_params),
        # (sflex_gamma_map, sflex_gamma_map_params, data_param_grid_meg, CV_noise_params),
        # ---------------- sFLEX-Gamma-Lambda-MAP ----------------
        # (sflex_gamma_lambda_map, sflex_gamma_lambda_map_params, data_param_grid_meg, adaptive_noise_params),
        # ---------------- Gamma-MAP ----------------
        # (gamma_map, gamma_map_params, data_param_grid_meg, basic_noise_params),
        # (gamma_map, gamma_map_params, data_param_grid_meg, CV_noise_params),
        

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

    metrics_config = {
        "evaluation": (
            "mean_posterior_std",           # Uncertainty summary
            "emd",                          # Spatial accuracy
            "jaccard_error",
            "mse",
            "euclidean_distance",           # Detection
            "f1",
            "accuracy",
        ),
        "calibration": (
            "mean_calibration_error",
            "mean_signed_deviation",
            "mean_absolute_deviation",
            "max_underconfidence_deviation",
            "max_overconfidence_deviation",
        ),
    }

    metric_evaluator = MetricEvaluator(
        nominal_coverages=nominal_coverages,
        evaluation_metrics=metrics_config["evaluation"],
        calibration_metrics=metrics_config["calibration"],
        logger=logger,
    )

    nruns = 20
    benchmark_n_jobs = 1
    logger.info(
        "Benchmark parallel workers: n_jobs=%s, experiments per configuration: %s",
        benchmark_n_jobs,
        nruns,
    )
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
            logger=logger,
        )
        results_df = benchmark.run(nruns=nruns, n_jobs=benchmark_n_jobs)
        df.append(results_df)

    # this_result['avg_posterior_variance'] = np.mean(posterior_var)
    # this_result['std_posterior_variance'] = np.std(np.sqrt(posterior_var))
    # this_result['avg_posterior_mean'] = np.mean(x_hat_one_trial)
    # this_result['std_posterior_mean'] = np.std(x_hat_one_trial)
    
    results_df = pd.concat(df)
    sort_cols = [c for c in ['subject', 'nnz','solver', 'noise_type',  'alpha_SNR', 'gamma'] if c in results_df.columns]
    if sort_cols:
        results_df.sort_values(by=sort_cols, inplace=True, ascending=True)
    desired_order = ['run_id', 'subject', 'orientation_type', 'nnz','solver', 'noise_type',  'alpha_SNR', 'gamma', 'noise_var']
    
    cols = [c for c in desired_order if c in results_df.columns] + \
           [c for c in results_df.columns if c not in desired_order]
    results_df = results_df[cols]
    results_df.to_csv(f"results/benchmark_results/benchmark_results_{timestamp}.csv", index=False)
    
    print(results_df.head())
    elapsed = datetime.datetime.now() - start_time
    logger.info("Benchmark completed in %s", elapsed)

if __name__ == "__main__":
    main()
