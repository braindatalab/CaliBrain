import datetime
import os
import logging
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

from calibrain import EvaluationMetrics
from calibrain import SourceEstimator, DataSimulator, UncertaintyEstimator, gamma_map, eloreta

class Benchmark:
    def __init__(self, solver, solver_param_grid, data_param_grid, data_simulator, metrics, random_state=42, logger=None):
        """
        Initialize the Benchmark class.

        Parameters:
        - solver (callable): The solver function (e.g., gamma_map, eloreta).
        - solver_param_grid (dict): Grid of solver hyperparameters.
        - data_param_grid (dict): Grid of data generation hyperparameters.
        - data_simulator (DataSimulator): Instance of DataSimulator for generating data.
        - metrics (list of callables): List of metric functions to evaluate the results.
        - random_state (int, optional): Random seed for reproducibility.
        - logger (logging.Logger, optional): Logger instance for logging messages.
        """
        self.solver = solver
        self.solver_param_grid = solver_param_grid
        self.data_param_grid = data_param_grid
        self.data_simulator = data_simulator
        self.metrics = metrics
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
            for k, v in params.items() if k not in ("cov", "run_id")
            # Add other keys to exclude from path if necessary
        }
    
        # Desired order of parameters for the directory structure
        # This list defines the specific order.
        if desired_order is None:
            desired_order = ["subject", "solver", "gammas", "orientation_type", "nnz", "noise_type", "alpha_SNR", "seed"]
        
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
             
    def run(self, nruns=2, fig_path="results/figures/uncertainty_analysis_figures"):
        """
        Run benchmarking by iterating over combinations of solver and data parameters.

        Returns:
        - results (pd.DataFrame): DataFrame containing the results for each parameter combination.
        """
        rng = check_random_state(self.random_state)
        seeds = rng.randint(low=0, high=2 ** 32, size=nruns)
        results_list = []
        
        self.logger.info(f"Starting benchmark with {nruns} runs...")
        for solver_params, data_params, seed, in product(
            ParameterGrid(self.solver_param_grid),
            ParameterGrid(self.data_param_grid),
            seeds
        ):
            solver_name = getattr(self.solver, "__name__", str(self.solver))
            self.logger.info(f"Run {len(results_list) + 1}/{nruns*len(ParameterGrid(self.solver_param_grid))*len(ParameterGrid(self.data_param_grid))}, Seed: {seed}")
            self.logger.info(f"Evaluating solver: {solver_name}")
            self.logger.info(f"Solver parameters: {solver_params}")
            self.logger.info(f"Data parameters: {data_params}")
            
            this_result = {
                'run_id': len(results_list) + 1,
                "seed": seed,
                "solver": solver_name,
                **solver_params,
                **data_params,
            }
            experiment_dir = self.create_experiment_directory(
                base_dir=fig_path,
                params=this_result,
                desired_order = ["subject", "solver", "gammas", "orientation_type", "nnz", "noise_type", "alpha_SNR", "seed"]
            )
            
            self.data_simulator.__dict__.update(data_params)
            # self.data_simulator.noise_type = solver_params.get("noise_type")

            try:
                # Simulate data (with n_trials!)
                self.logger.info("Simulating data...")
                data = self.data_simulator.simulate(
                    data_params['subject'], 
                    seed,
                    visualize=True,
                    save_path=os.path.join(experiment_dir , "data_simulation")
                )
                y_noisy, L, x, x_active_indices, noise, noise_var = data
                trial_i = 0
                
                # if solver_params.get("noise_type") == 'oracle':
                #     solver_params["cov"] = cov_scaled
                
                n_orient = 3 if data_params.get("orientation_type") == "free" else 1 # TODO: put this in the LeadfieldSimulator class
                
                self.logger.info("Initializing source estimator...")
                source_estimator = SourceEstimator(
                    solver=self.solver,
                    solver_params=solver_params,
                    # cov=cov_scaled,
                    n_orient=n_orient,
                    logger=self.logger
                )            
                source_estimator.fit(L, y_noisy)            
                x_hat, x_hat_active_set, posterior_cov = source_estimator.predict(
                    y=y_noisy[trial_i], noise_var=noise_var[trial_i]
                )
                this_result['active_set_size'] = len(x_hat_active_set)
                
                # TODO: check wheter keepdims=True is needed later
                x_avg_time = np.mean(x, axis=2, keepdims=True)
                x_hat_avg_time = np.mean(x_hat, axis=1, keepdims=True)
                
                self.logger.info("Initializing uncertainty estimator...")
                uncert_est = UncertaintyEstimator(
                    orientation_type=self.data_simulator.orientation_type,
                    x=x_avg_time[trial_i],
                    x_hat=x_hat_avg_time,
                    n_totl_sources=x_avg_time[trial_i].shape[0],
                    active_set=x_hat_active_set,
                    x_active_indices=x_active_indices[trial_i],
                    posterior_cov=posterior_cov,
                    experiment_dir=experiment_dir,
                    confidence_levels= np.arange(0.0, 1.1, 0.1),
                    logger=self.logger,
                )
                full_posterior_cov = uncert_est.construct_full_covariance()
                # uncert_est.plot_sorted_posterior_variances(top_k=10)
                # uncert_est.visualize_sorted_covariances(top_k=10)
                # uncert_est.plot_posterior_covariance_matrix()
                
                
                # NOTE: since uncert_est.x = x_avg_time[trial_i], the following plots will have a single time step (and ofcourse one trial). So we can ignore the time_step parameter for now.
                # uncert_est.plot_active_sources_single_time_step(
                #     uncert_est.x[x_active_indices[trial_i]],
                #     uncert_est.x_hat[x_hat_active_set],
                # )
                

                # --- Evaluate metrics for ALL sources ---
                self.logger.info("Computing CI and empirical coverage for ALL sources...")
                # counts_within_ci: (n_condifence_levels, n_ori, n_times)
                ci_lower, ci_upper, counts_within_ci = \
                uncert_est.get_confidence_intervals_data(
                    uncert_est.x, #[x_hat_active_set],
                    uncert_est.x_hat, #[x_hat_active_set]
                    full_posterior_cov, #posterior_cov
                )
                # Normalize by the number of active sources and take the first orientation and time step
                empirical_coverage = (counts_within_ci / uncert_est.x.shape[0])[:, 0, 0]
                
                eval_metrics = EvaluationMetrics(
                    x=uncert_est.x,
                    x_hat=uncert_est.x_hat,
                    subject=data_params.get("subject"),
                    orientation_type=data_params.get("orientation_type"),
                    nnz=data_params.get("nnz"),
                    y=y_noisy[trial_i],
                    L=L,
                    cov=full_posterior_cov,
                    confidence_levels=uncert_est.confidence_levels,
                    empirical_coverage=empirical_coverage,
                    metrics_to_call=self.metrics,
                    logger=self.logger                   
                )
                eval_metrics.evaluate_and_store_metrics(this_result, metric_suffix="_all_sources")
                
            
                # --- Evaluate metrics for ACTIVE sources (estimated active set) ---
                self.logger.info("Computing CI and empirical coverage for ACTIVE sources...")

                _, _, counts_within_ci_active = \
                uncert_est.get_confidence_intervals_data(
                    uncert_est.x[x_hat_active_set],
                    uncert_est.x_hat[x_hat_active_set],
                    posterior_cov,
                )
                # Normalize by the number of active sources and take the first orientation and time step
                empirical_coverage_active = (counts_within_ci_active / uncert_est.x[x_hat_active_set].shape[0])[:, 0, 0]

                # Create EvaluationMetrics instance for active indices
                eval_metrics_active = EvaluationMetrics(
                    x=uncert_est.x[x_hat_active_set],
                    x_hat=uncert_est.x_hat[x_hat_active_set],
                    subject=data_params.get("subject"),
                    orientation_type=data_params.get("orientation_type"),
                    nnz=data_params.get("nnz"),
                    y=y_noisy[trial_i],
                    L=L,
                    cov=posterior_cov,
                    confidence_levels=uncert_est.confidence_levels,
                    empirical_coverage=empirical_coverage_active,
                    metrics_to_call=self.metrics,
                    logger=self.logger     
                )
                                    
                eval_metrics_active.evaluate_and_store_metrics(this_result, metric_suffix="_active_indices")
                
                
                # --- Plotting ---
                uncert_est.visualize_confidence_intervals(
                    ci_lower, ci_upper,                            
                    uncert_est.x[x_active_indices[trial_i]],
                    uncert_est.x_hat[x_hat_active_set],
                )
                # all sources  
                uncert_est.vizualise_calibration_curve(
                    empirical_coverage,
                    this_result,
                    which_legend="all_sources",
                    filename="Calibration_curve_all_sources"
                    
                )
                # active sources
                uncert_est.vizualise_calibration_curve(
                    empirical_coverage_active,
                    this_result,
                    which_legend="active_set",
                    filename="Calibration_curve_active_set"
                )
                # if self.data_param_grid.get("orientation_type") == "free":
                #     uncert_est.plot_top_relevant_CE_pairs(top_k=5, confidence_level=0.95)
            
            
            except Exception as e:
                self.logger.error(f"Error during benchmarking run_id {this_result.get('run_id', 'N/A')}: {e}", exc_info=True)
                this_result["error_message"] = str(e) # More specific error key
            
            results_list.append(this_result)
            self.logger.info(f"Completed run_id {this_result.get('run_id', 'N/A')}")

        self.logger.info("Benchmarking completed.")
        return pd.DataFrame(results_list)
