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

from calibrain import SourceEstimator, DataSimulator, UncertaintyEstimator, gamma_map, eloreta

class Benchmark:
    def __init__(self, solver, solver_param_grid, data_param_grid, data_simulator, metrics, logger=None):
        """
        Initialize the Benchmark class.

        Parameters:
        - solver (callable): The solver function (e.g., gamma_map, eloreta).
        - solver_param_grid (dict): Grid of solver hyperparameters.
        - data_param_grid (dict): Grid of data generation hyperparameters.
        - metrics (list of callables): List of metric functions to evaluate the results.
        - logger (logging.Logger, optional): Logger instance for logging messages.
        """
        self.solver = solver
        self.solver_param_grid = solver_param_grid
        self.data_param_grid = data_param_grid
        self.data_simulator = data_simulator
        self.metrics = metrics
        self.logger = logger if logger else logging.getLogger(__name__)
    
    def create_experiment_directory(self, base_dir, params):
        """
        Create a directory structure for the experiment, with subdirectories for each parameter in a specified order, followed by any remaining parameters.
    
        Parameters:
        - base_dir (str): Base directory for the experiment.
        - params (dict): Dictionary of parameters.
    
        Returns:
        - experiment_dir (str): Path to the experiment directory.
        """
        # Exclude 'cov' and sanitize values for directory names
        sanitized_params_for_path = {
            k: str(v).replace("/", "_").replace("\\", "_").replace(" ", "_")
            for k, v in params.items() if k != "cov" 
            # Add other keys to exclude from path if necessary, e.g. if k != "alpha_snr_db"
        }
    
        # Desired order of parameters for the directory structure
        # This list defines the specific order.
        desired_order = ["subject", "estimator", "gammas", "orientation_type", "nnz", "noise_type", "alpha_snr_db"]
        
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

    def run(self, nruns=1):
        """
        Run benchmarking by iterating over combinations of solver and data parameters.

        Returns:
        - results (pd.DataFrame): DataFrame containing the results for each parameter combination.
        """
        rng = check_random_state(42) 
        seeds = rng.randint(low=0, high=2 ** 32, size=nruns)

        results = []
        
        self.logger.info(f"Starting benchmark with {nruns} runs...")
        # Iterate over all combinations of solver and data parameters
        for solver_params, data_params, seed, in product(
            ParameterGrid(self.solver_param_grid),
            ParameterGrid(self.data_param_grid),
            seeds
        ):
            solver_name = getattr(self.solver, "__name__", str(self.solver))

            self.logger.info(f"Evaluating solver: {solver_name}")
            self.logger.info(f"Solver parameters: {solver_params}")
            self.logger.info(f"Data parameters: {data_params}")

            # Create experiment directory
            self.logger.info("Creating experiment directory...")
            
            experiment_dir = self.create_experiment_directory(
                base_dir="results/figures/uncertainty_analysis_figures",
                params={
                    "estimator": solver_name,
                    **solver_params, 
                    **data_params}
            )
            
            # Update data_simulator with data parameters
            self.data_simulator.seed = seed
            self.data_simulator.rng = rng
            # self.data_simulator.noise_type = solver_params.get("noise_type")
            self.data_simulator.__dict__.update(data_params)

            # Simulate data (with n_trials!)
            self.logger.info("Simulating data...")
            data = self.data_simulator.simulate(
                data_params['subject'], 
                visualize=True,
                save_path=os.path.join(experiment_dir , "data")
            )
            y_noisy, L, x, x_active_indices, noise, noise_power = data
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
                y=y_noisy[trial_i], noise_var=noise_power[trial_i])
            
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
            self.logger.info("Creating figures for uncertainty analysis...")
            
            full_posterior_cov = uncert_est.construct_full_covariance()
            # uncert_est.plot_sorted_posterior_variances(top_k=10)
            # uncert_est.visualize_sorted_covariances(top_k=10)
            # uncert_est.plot_posterior_covariance_matrix()
            
            
            # NOTE: since uncert_est.x = x_avg_time[trial_i], the following plots will have a single time step (and ofcourse one trial). So we can ignore the time_step parameter for now.
            uncert_est.plot_active_sources_single_time_step(
                uncert_est.x[x_active_indices[trial_i]],
                uncert_est.x_hat[x_hat_active_set],
            )
            
            # The counts_within_ci have the shape of (n_condifence_levels, n_ori, n_times)
            ci_lower, ci_upper, counts_within_ci = uncert_est.get_confidence_intervals_data(
                uncert_est.x, #[x_hat_active_set],
                uncert_est.x_hat, #[x_hat_active_set]
                full_posterior_cov,
            )

            uncert_est.visualize_confidence_intervals(
                ci_lower, ci_upper,                            
                uncert_est.x[x_active_indices[trial_i]],
                uncert_est.x_hat[x_hat_active_set],
            )
            if self.data_param_grid.get("orientation_type") == "free":
                uncert_est.plot_top_relevant_CE_pairs(top_k=5, confidence_level=0.95)
                
            # plot observed hit rate vs. confidence levels           
            uncert_est.vizualise_calibration_curve(
                counts_within_ci,
                normalize_hits=False,
            )
            
            uncert_est.vizualise_calibration_curve(
                counts_within_ci,
                normalize_hits=True, 
            )
            
            
            # uncert_est.visualize_confidence_intervals(confidence_levels, time_point=0)

            # TBC: Evaluate metrics
            self.logger.info("Evaluating metrics...")
            run_results = {
                "run": len(results) + 1,
                "seed": seed,
                "solver": solver_name,
                "solver_params": solver_params,
                "data_params": data_params,
                "active_set_size": len(x_hat_active_set),
            }
            
            try:
                for metric in self.metrics:
                    if metric is not None:
                        run_results[metric.__name__] = metric(x_hat, x)
            except Exception as e:
                self.logger.error(f"Error evaluating metrics: {e}")
                run_results["error"] = str(e)
                
            results.append(run_results)

        self.logger.info("Benchmarking completed.")

        return pd.DataFrame(results)
