import datetime
import os
import logging
from typing import Optional
from zipfile import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.utils import check_random_state
from itertools import product
import mne
from itertools import combinations
from scipy.stats import wishart
from matplotlib.patches import Ellipse
from scipy.stats import chi2

from calibrain import MetricEvaluator
from calibrain import SourceEstimator, SensorSimulator, SourceSimulator, UncertaintyEstimator, Visualizer, LeadfieldBuilder, gamma_map, eloreta
# from calibrain.leadfield_builder import LeadfieldBuilder
from calibrain.utils import get_data_path, inspect_object
from mne.io.constants import FIFF
from calibrain import SpatialCVSolver, TemporalCVSolver

class Benchmark:
    def __init__(self, solver : callable, solver_param_grid : dict, data_param_grid : dict, noise_param_grid : dict, ERP_config : dict, source_simulator : SourceSimulator, leadfield_builder : LeadfieldBuilder, sensor_simulator : SensorSimulator, uncertainty_estimator : UncertaintyEstimator, metric_evaluator : MetricEvaluator, random_state=42, logger=None):
        """
        Initialize the Benchmark class.

        Parameters:
        ----------
        solver : callable
            The solver function (e.g., gamma_map, eloreta).
        solver_param_grid : dict
            Grid of solver hyperparameters (e.g. noise_type, init_gamma).
        data_param_grid : dict
            Grid of data generation hyperparameters.
        noise_param_grid : dict
            Grid of noise-related hyperparameters.
        ERP_config : dict
            Configuration for ERP simulation.
        source_simulator : SourceSimulator
            Instance of SourceSimulator for generating source data.
        leadfield_builder : LeadfieldBuilder
            Instance of LeadfieldBuilder for generating leadfields.
        sensor_simulator : SensorSimulator
            Instance of SensorSimulator for generating data.
        uncertainty_estimator : UncertaintyEstimator
            Instance of UncertaintyEstimator for uncertainty estimation.
        metric_evaluator : MetricEvaluator
            Instance of MetricEvaluator for evaluating metrics.
        random_state : int, optional
            Random seed for reproducibility.
        logger : logging.Logger, optional
            Logger instance for logging messages.
        """
        self.solver = solver
        self.solver_param_grid = solver_param_grid
        self.data_param_grid = data_param_grid
        self.noise_param_grid = noise_param_grid
        self.ERP_config = ERP_config
        self.source_simulator = source_simulator
        self.leadfield_builder = leadfield_builder
        self.sensor_simulator = sensor_simulator
        self.uncertainty_estimator = uncertainty_estimator
        self.metric_evaluator = metric_evaluator
        self.random_state = random_state
        self.logger = logger if logger else logging.getLogger(__name__)
    
    def create_experiment_directory(self, base_dir, params, desired_order):
        """
        Create a directory structure for the experiment, with subdirectories for each parameter in a specified order, followed by any remaining parameters.
    
        Parameters:
        - base_dir (str): Base directory for the experiment.
        - params (dict): Dictionary of parameters.
        - desired_order (list): List of parameter keys in the desired order for the directory structure.
    
        Returns:
        - experiment_dir (str): Path to the experiment directory.
        """
        # Exclude 'cov' and sanitize values for directory names
        sanitized_params_for_path = {
            k: str(v).replace("/", "_").replace("\\", "_").replace(" ", "_")
            for k, v in params.items() if k not in ("run_id")
            # Add other keys to exclude from path if necessary
        }
    
        # Desired order of parameters for the directory structure
        # This list defines the specific order.
        if desired_order is None:
            desired_order = ["solver", "alpha_SNR", "noise_type", "orientation_type", "nnz",  "sensor_white_noise_var", "subject", "seed"]
        
        path_components = []
        
        # Add parameters in the desired order if they exist in the sanitized params
        for key in desired_order:
            if key in sanitized_params_for_path:
                path_components.append(f"{key}={sanitized_params_for_path[key]}")
                del sanitized_params_for_path[key] # Remove to avoid adding them again
                
        # Add any remaining parameters, sorted by key for consistent ordering
        for key, value in sorted(sanitized_params_for_path.items()):
            path_components.append(f"{key}={value}")
            
        # Create subdirectories for each parameter component
        experiment_dir = base_dir
        for component in path_components:
            experiment_dir = os.path.join(experiment_dir, component)
    
        try:
            # Create the directory structure if it doesn't exist
            os.makedirs(experiment_dir, exist_ok=True)
        except OSError as e:
            self.logger.error(f"Failed to create experiment directory: {experiment_dir}. Error: {e}")
            raise
    
        self.logger.info(f"Experiment directory created: {experiment_dir}")
        return experiment_dir

    def run(self, nruns: int = 2, fig_path: str = "results/figures"):
        """
        Run benchmarking by iterating over combinations of solver and data parameters.

        Returns:
        - results (pd.DataFrame): DataFrame containing the results for each parameter combination.
        """
        # -------------------------------------------------------------
        # 1. Generate seeds and initialize bookkeeping
        # -------------------------------------------------------------        
        rng = check_random_state(self.random_state)
        seeds = rng.randint(low=0, high=2 ** 32, size=nruns)
        results_list = []
                
        param_combinations = list(product(
            ParameterGrid(self.solver_param_grid),
            ParameterGrid(self.data_param_grid),
            ParameterGrid(self.noise_param_grid),
            seeds
        ))
         
        # -------------------------------------------------------------
        # 2. Iterate over solver and data parameter combinations
        # -------------------------------------------------------------
        self.logger.info(f"Starting benchmark with {len(param_combinations)} runs...")

        for run_id, (solver_params, data_params, noise_params, seed) in enumerate(param_combinations, start=1):
            solver_name = getattr(self.solver, "__name__", str(self.solver))
            orientation_type = data_params.get("orientation_type")
            n_orient = 3 if orientation_type == "free" else 1
            solver_params['fwd_path'] = get_data_path() / f"{data_params['subject']}-fwd.fif"
            
            self.logger.info(f"[Run {run_id}/{len(param_combinations)}] Seed: {seed}")
            self.logger.info(f"Solver: {solver_name} | Params: {solver_params}")
            self.logger.info(f"Noise Params: {noise_params}")
            self.logger.info(f"Data Params: {data_params}")
            
            this_result = {
                'run_id': run_id,
                "seed": seed,
                "solver": solver_name,
                'noise_type': noise_params['noise_type'],
                **{k: v for k, v in solver_params.items() if k != "fwd_path"},
                **data_params,
            }
            
            n_trials = 5
            
            global_source_rng = np.random.RandomState(seed)
            global_source_seeds = global_source_rng.randint(0, 2**32 - 1, n_trials)

            global_noise_rng = np.random.RandomState(seed + 123456)
            global_noise_seeds = global_noise_rng.randint(0, 2**32 - 1, size=n_trials)
            
            try:
                # -------------------------------------------------------------
                # 3. Create directory
                # -------------------------------------------------------------
                experiment_dir = self.create_experiment_directory(
                    base_dir=fig_path,
                    params=this_result,
                    desired_order = [
                        "solver", "noise_type", "alpha_SNR", "orientation_type", "nnz", "sensor_white_noise_var", "subject", "seed"
                    ]
                )
                
                # -------------------------------------------------------------
                # 4. Get leadfield matrix
                # -------------------------------------------------------------
                self.logger.info("Building leadfield matrix...")
                L = self.leadfield_builder.get_leadfield(
                    subject=data_params['subject'],
                    orientation_type=orientation_type,
                    retrieve_mode="load"
                )
                n_sensors, n_sources = L.shape
                sensor_units = self.leadfield_builder.sensor_units

                # -------------------------------------------------------------
                # 5. Simulate source and sensor data
                # -------------------------------------------------------------
                self.logger.info("Simulating source trials...")
                x_trials, x_active_indices_trials = self.source_simulator.simulate(
                    orientation_type=orientation_type,
                    n_sources=n_sources,
                    nnz=data_params['nnz'],
                    n_trials=n_trials,
                    global_seed=global_source_seeds,
                )
                source_units = self.source_simulator.source_units
                
                # -------------------------------------------------------------
                # 6. Simulate sensor and sensor data
                # -------------------------------------------------------------
                self.logger.info("Simulating sensor trials...")
                y_clean_trials, y_noisy_trials, noise_trials, noise_eta_trials =\
                    self.sensor_simulator.simulate(
                        x_trials=x_trials,
                        L=L,
                        orientation_type=orientation_type,
                        alpha_SNR=data_params['alpha_SNR'],
                        sensor_white_noise_var=data_params['sensor_white_noise_var'],
                        n_trials=n_trials,
                        global_seed=global_noise_seeds,
                )
                self.sensor_simulator.sensor_units = sensor_units  # Set units based on leadfield

                # -------------------------------------------------------------
                # 7. Slice the first trial for processing
                # -------------------------------------------------------------
                trial_idx = 0
                x_one_trial = x_trials[trial_idx]
                x_active_indices_one_trial = x_active_indices_trials[trial_idx]
                y_noisy_one_trial = y_noisy_trials[trial_idx]
                noise_eta_one_trial = noise_eta_trials[trial_idx] # noise scaling factor (eta)

                # -------------------------------------------------------------
                # 7a. Baseline noise variance estimate (pre-stimulus window)
                # -------------------------------------------------------------
                tmin = self.source_simulator.ERP_config['tmin']
                stim_onset = self.source_simulator.ERP_config['stim_onset']
                sfreq = self.source_simulator.ERP_config['sfreq']
                pre_stimulus_onset = int((stim_onset - tmin) * sfreq)
                if pre_stimulus_onset <= 0:
                    self.logger.warning(
                        "Computed pre_stimulus_onset <= 0; using full trial for baseline estimation"
                    )
                    y_pre = y_noisy_one_trial
                else:
                    y_pre = y_noisy_one_trial[:, :pre_stimulus_onset]

                try:
                    baseline_noise_var = float(np.mean(np.std(y_pre, axis=1) ** 2))
                except Exception:
                    baseline_noise_var = None
                if not baseline_noise_var or not np.isfinite(baseline_noise_var):
                    baseline_noise_var = 1.0  # safe fallback

                # -------------------------------------------------------------
                # 8. Determine noise variance based on noise_type
                # -------------------------------------------------------------
                # Validate noise_type and set initial noise variance / cv callable
                allowed_noise_types = {
                    "joint_learning",
                    "oracle",
                    "baseline",
                    "spatial_cv",
                    "temporal_cv",
                }
                noise_type = noise_params.get("noise_type")
                if noise_type not in allowed_noise_types:
                    raise ValueError(f"Invalid noise_type: {noise_type!r}. Allowed: {sorted(allowed_noise_types)}")

                if noise_type in ('spatial_cv', 'temporal_cv'):
                    grid_factors = np.logspace(-2, 2, 10)
                    alphas = baseline_noise_var * grid_factors
                    noise_variances = np.unique(alphas).tolist()
                    if not noise_variances:
                        noise_variances = [baseline_noise_var]
                    estimator = SpatialCVSolver if noise_type == 'spatial_cv' else TemporalCVSolver                    

                    # Create source estimator with CV for noise variance selection
                    source_estimator = estimator( 
                        solver=self.solver,
                        solver_params=solver_params,
                        n_orient=n_orient,
                        logger=self.logger,
                        # noise_variances=noise_params.get('default_alphas_grid', None),
                        noise_variances=noise_variances,
                        cv=noise_params.get('cv', 5),
                        n_jobs=noise_params.get('n_jobs', 1),
                    )
                else:                                    
                    if noise_type == 'oracle':
                        print("----- oracle noise variance estimation -----")
                        noise_var = float((data_params['sensor_white_noise_var'] * noise_eta_one_trial) ** 2)
                        
                    # Take the pre-stimulus data segment in sensor space, calculate the standard deviation (across time) for each channel and average them, then we will have a baseline sigma (for a single trial):
                    elif noise_type == 'baseline':
                        print("----- baseline noise variance estimation -----")
                        noise_var = baseline_noise_var
                        # record for debugging/inspection
                        self.logger.info(
                            f"Baseline noise variance (trial {run_id}): {noise_var:.3e}, eta: {noise_eta_one_trial:.3e}"
                        )

                    elif noise_type == 'joint_learning':
                        noise_var = None


                    # Create source estimator with fixed noise variance
                    source_estimator = SourceEstimator(
                        solver=self.solver,
                        solver_params=solver_params,
                        noise_var=noise_var,
                        n_orient=n_orient,
                        logger=self.logger
                    )
                    
                # -------------------------------------------------------------
                # 8. Fit the source estimator and predict posterior mean & cov
                # -------------------------------------------------------------
                self.logger.info(f"Fitting source estimator {self.solver.__name__}")
                source_estimator.fit(L, y_noisy_one_trial)
                solver_output = source_estimator.predict(y=y_noisy_one_trial)

                x_hat_one_trial = solver_output.get("posterior_mean")
                x_hat_active_indices_one_trial = solver_output.get("active_indices")
                posterior_cov = solver_output.get("posterior_cov")
                
                noise_var = solver_output.get("noise_var")
                gamma = solver_output.get("gamma")
                
                this_result['gamma'] = gamma
                this_result["noise_var"] = noise_var
                
                posterior_var = self.uncertainty_estimator.get_posterior_variance(
                    posterior_cov=posterior_cov,
                    orientation_type=orientation_type
                )
                this_result['mean_posterior_variance'] = np.mean(posterior_var)
                
                # -------------------------------------------------------------
                # 9. Estimate uncertainty (-> credible intervals)
                # -------------------------------------------------------------
                self.logger.info("Estimating uncertainty...")
                if solver_name == 'gamma_map':
                    self.logger.info("Skipping uncertainty estimation for gamma_map solver.")
                    results_list.append(this_result)
                    continue
                                
                x_one_trial_avg_time = np.mean(x_one_trial, axis=1, keepdims=True)
                x_hat_one_trial_avg_time = np.mean(x_hat_one_trial, axis=1, keepdims=True)
                
                # Scale posterior variance by number of time points (averaging over time reduces variance)
                n_time = x_one_trial.shape[1]
                posterior_var_avg_time = posterior_var / n_time
                    
                # Compute confidence intervals
                ci_lower_active, ci_upper_active, counts_within_ci_active, empirical_coverages_pre_cal = \
                    self.uncertainty_estimator.get_credible_intervals_data(
                        x=x_one_trial_avg_time,
                        x_hat=x_hat_one_trial_avg_time,
                        posterior_var=posterior_var_avg_time,
                        orientation_type=orientation_type
                    )

                # Calibrate using CV
                empirical_coverages_post_cal, fold_results = \
                    self.uncertainty_estimator.calibration_CV(
                    x=x_one_trial_avg_time,
                    x_hat=x_hat_one_trial_avg_time,
                    posterior_var=posterior_var_avg_time,
                    orientation_type='fixed', n_folds=5, random_state=42)
                    
                # -------------------------------------------------------------
                # 10. Evaluate metrics and merge them into this_result
                # -------------------------------------------------------------
                metric_kwargs = dict(
                    x=x_one_trial_avg_time,
                    x_hat=x_hat_one_trial_avg_time,
                    posterior_var=posterior_var_avg_time,
                    orientation_type=orientation_type,
                    nnz=data_params.get("nnz"),
                    subject=data_params.get("subject"),
                    fwd_path=self.leadfield_builder.leadfield_dir,
                )

                try:
                    metric_results_pre_cal = self.metric_evaluator.evaluate_metrics(
                        empirical_coverages=empirical_coverages_pre_cal,
                        **metric_kwargs
                    )
                    # store metrics with suffix to indicate pre-calibration
                    suffixed_pre = {f"pre_cal_{k}": v for k, v in metric_results_pre_cal.items()}
                    this_result.update(suffixed_pre)

                    metric_results_post_cal = self.metric_evaluator.evaluate_metrics(
                        empirical_coverages=empirical_coverages_post_cal,
                        **metric_kwargs
                    )
                    # store metrics with suffix to indicate post-calibration
                    suffixed_post = {f"post_cal_{k}": v for k, v in metric_results_post_cal.items()}
                    this_result.update(suffixed_post)

                    for k in self.metric_evaluator.metrics:
                        # compute improvement using the original metric keys, store as <metric>_improvement
                        improvement_key = f"improvement_{k}"
                        try:
                            pre = metric_results_pre_cal.get(k)
                            post = metric_results_post_cal.get(k)
                            if pre is None or post is None:
                                this_result[improvement_key] = None
                            else:
                                this_result[improvement_key] = (pre - post) / pre * 100
                        except Exception:
                            this_result[improvement_key] = None
                        
                except Exception as e:
                    self.logger.error(f"Error while evaluating metrics: {e}", exc_info=True)
                    this_result.update({"metric_evaluation_error": str(e)})

                # --------------------------------------------------------------
                # 11. Vizualization
                # --------------------------------------------------------------
                viz = Visualizer(base_save_path=experiment_dir, logger=self.logger)
                viz.plot_all(
                    x_trials=x_trials,
                    x_active_indices_trials=x_active_indices_trials,
                    x_hat_one_trial=x_hat_one_trial,
                    x_hat_active_indices_one_trial=x_hat_active_indices_one_trial,
                    y_clean_trials=y_clean_trials,
                    y_noisy_trials=y_noisy_trials,
                    trial_idx=trial_idx,
                    n_sources=n_sources,
                    subject=data_params.get("subject"),
                    subjects_dir=mne.datasets.sample.data_path() / 'subjects', #TODO: include MEG anatomical data
                    fwd_path=self.leadfield_builder.leadfield_dir,
                    nnz=data_params.get("nnz"),
                    ERP_config=self.ERP_config,
                    sample_idx=200,
                    source_units=source_units,
                    sensor_units=sensor_units,
                    confidence_levels=self.uncertainty_estimator.confidence_levels,
                    nominal_coverages=self.uncertainty_estimator.nominal_coverages,
                    empirical_coverages=empirical_coverages_pre_cal,
                    empirical_coverages_post_cal=empirical_coverages_post_cal,
                    ci_lower=ci_lower_active,
                    ci_upper=ci_upper_active,
                    orientation_type=orientation_type,
                    result=this_result,
                    experiment_dir=experiment_dir,
                )
                
                active_indices_size = (
                    len(x_hat_active_indices_one_trial)
                    if x_hat_active_indices_one_trial is not None
                    else 0
                )
                this_result['active_indices_size'] = active_indices_size

            except Exception as e:
                self.logger.error(f"Error during benchmarking run_id {this_result.get('run_id', 'N/A')}: {e}", exc_info=True)
                this_result["error_message"] = str(e) # More specific error key
            
            results_list.append(this_result)
            self.logger.info(f"Completed run_id {this_result.get('run_id', 'N/A')}")

        self.logger.info("Benchmarking completed.")
        return pd.DataFrame(results_list)
