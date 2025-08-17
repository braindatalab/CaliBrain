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
# from calibrain.leadfield_simulation import LeadfieldBuilder
from calibrain.utils import inspect_object
from mne.io.constants import FIFF

class Benchmark:
    def __init__(self, solver : callable, solver_param_grid : dict, data_param_grid : dict, ERP_config : dict, source_simulator : SourceSimulator, leadfield_builder : LeadfieldBuilder, sensor_simulator : SensorSimulator, uncertainty_estimator : UncertaintyEstimator, metric_evaluator : MetricEvaluator, random_state=42, logger=None):
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
            for k, v in params.items() if k not in ("cov", "run_id")
            # Add other keys to exclude from path if necessary
        }
    
        # Desired order of parameters for the directory structure
        # This list defines the specific order.
        if desired_order is None:
            desired_order = ["subject", "solver", "init_gamma", "orientation_type", "alpha_SNR", "noise_type", "nnz", "seed"]
        
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

    def run(self, nruns: int = 2, fig_path: str = "results/figures/uncertainty_analysis_figures"):
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
            seeds
        ))
         
        # -------------------------------------------------------------
        # 2. Iterate over solver and data parameter combinations
        # -------------------------------------------------------------
        self.logger.info(f"Starting benchmark with {len(param_combinations)} runs...")
        
        for run_id, (solver_params, data_params, seed) in enumerate(param_combinations, start=1):
            solver_name = getattr(self.solver, "__name__", str(self.solver))
            
            self.logger.info(f"[Run {run_id}/{len(param_combinations)}] Seed: {seed}")
            self.logger.info(f"Solver: {solver_name} | Params: {solver_params}")
            self.logger.info(f"Data Params: {data_params}")
            
            this_result = {
                'run_id': run_id,
                "seed": seed,
                "solver": solver_name,
                **solver_params,
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
                        "subject", "solver", "init_gamma", "orientation_type", "alpha_SNR", "noise_type", "nnz", "seed"
                    ]
                )
                
                # -------------------------------------------------------------
                # 4. Get leadfield matrix
                # -------------------------------------------------------------
                self.logger.info("Building leadfield matrix...")
                L = self.leadfield_builder.get_leadfield(
                    subject=data_params['subject'],
                    orientation_type=data_params['orientation_type'],
                    retrieve_mode="load"
                )
                n_sensors, n_sources = L.shape
                sensor_units = self.leadfield_builder.sensor_units

                # -------------------------------------------------------------
                # 5. Simulate source and sensor data
                # -------------------------------------------------------------
                self.logger.info("Simulating source trials...")

                x_trials, x_active_indices_trials = self.source_simulator.simulate(
                    orientation_type=data_params['orientation_type'],
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
                y_clean_trials, y_noisy_trials, noise_trials, noise_var_trials =\
                    self.sensor_simulator.simulate(
                        x_trials=x_trials,
                        L=L,
                        orientation_type=data_params['orientation_type'],
                        alpha_SNR=data_params['alpha_SNR'],
                        n_trials=n_trials,
                        global_seed=global_noise_seeds,
                )
                self.sensor_simulator.sensor_units = sensor_units  # Set units based on leadfield

                # -------------------------------------------------------------
                # 7. Fit the source estimator and predict posterior mean & covariance
                # -------------------------------------------------------------
                # Slice the first trial for processing
                trial_idx = 0
                x = x_trials[trial_idx]
                x_active_indices = x_active_indices_trials[trial_idx]
                y_noisy = y_noisy_trials[trial_idx]
                noise_var = noise_var_trials[trial_idx]

                n_orient = 3 if data_params.get("orientation_type") == "free" else 1

                self.logger.info("Fitting source estimator...")
                source_estimator = SourceEstimator(
                    solver=self.solver,
                    solver_params=solver_params,
                    n_orient=n_orient,
                    logger=self.logger
                )            
                source_estimator.fit(L, y_noisy)            
                x_hat, x_hat_active_indices, posterior_cov = source_estimator.predict(
                    y=y_noisy, noise_var=noise_var
                )
                this_result['active_indices_size'] = len(x_hat_active_indices)

                # -------------------------------------------------------------
                # 8. Estimate uncertainty (credible intervals)
                # -------------------------------------------------------------
                self.logger.info("Estimating uncertainty...")
                
                # TODO: check whether we still need to set keepdims=True.
                x_avg_time = np.mean(x, axis=1, keepdims=True)
                x_hat_avg_time = np.mean(x_hat, axis=1, keepdims=True)
                
                # full_posterior_cov = self.uncertainty_estimator.construct_full_covariance(
                #     x=x_avg_time,
                #     x_hat_active_indices=x_hat_active_indices,
                #     posterior_cov=posterior_cov,
                #     orientation_type=data_params.get("orientation_type"),
                # )
                # self.uncertainty_estimator.debug_covariance(posterior_cov, full_posterior_cov, "Posterior Covariance")

                # Evaluate metrics: ALL SOURCES
                # self.logger.debug("Computing CI and empirical coverage for all sources")

                # # counts_within_ci: (n_condifence_levels, n_orient, n_times)
                # ci_lower, ci_upper, counts_within_ci =\
                # self.uncertainty_estimator.get_confidence_intervals_data(
                #     x=x_avg_time, #uncert_est.x, #[x_hat_active_indices],
                #     x_hat=x_hat_avg_time, #uncert_est.x_hat, #[x_hat_active_indices]
                #     posterior_cov=full_posterior_cov, #posterior_cov
                #     orientation_type=data_params.get("orientation_type")
                # )                            

                # # Evaluate metrics: ACTIVE SOURCES
                # self.logger.debug("Computing CI and empirical coverage for active sources")
                # ci_lower_active, ci_upper_active, counts_within_ci_active = \
                # self.uncertainty_estimator.get_confidence_intervals_data(
                #     x=x_avg_time[x_hat_active_indices], 
                #     x_hat=x_hat_avg_time[x_hat_active_indices],
                #     posterior_cov=posterior_cov,
                #     orientation_type=data_params.get("orientation_type")
                # )

                # Get boolean mask for sources present in both sets
                matched_mask = np.isin(x_hat_active_indices, x_active_indices)
                
                if not np.any(matched_mask):
                    self.logger.warning(f"No intersection between true active sources and estimated active sources")

                    ci_lower_active = np.zeros(
                        (len(self.uncertainty_estimator.confidence_levels), n_orient, 1)
                    )
                    ci_upper_active = np.zeros(
                        (len(self.uncertainty_estimator.confidence_levels), n_orient, 1)
                    )
                    counts_within_ci_active = np.zeros(
                        (len(self.uncertainty_estimator.confidence_levels), n_orient, 1)
                    )

                    empirical_coverage_active = np.zeros(
                        (len(self.uncertainty_estimator.confidence_levels))
                    )

                    empirical_coverage_active = np.zeros(
                        (len(self.uncertainty_estimator.confidence_levels))
                    )
                else:
                    # Get relative indices within x_hat_active_indices (for posterior_cov slicing)
                    matched_relative_indices = np.where(matched_mask)[0]

                    # Get the actual source indices for data slicing
                    matched_absolute_indices = x_hat_active_indices[matched_mask]

                    # Slice data using absolute indices (both arrays are full-size)
                    x_matched = x_avg_time[matched_absolute_indices]  # Use absolute for x (full array)
                    x_hat_matched = x_hat_avg_time[matched_absolute_indices]  # Use absolute for x_hat (also full array)

                    # Slice posterior covariance using relative indices (only contains active components)
                    posterior_cov_matched = posterior_cov[np.ix_(matched_relative_indices, matched_relative_indices)]

                    # Verify dimensions
                    print(f"x_matched shape: {x_matched.shape}")
                    print(f"x_hat_matched shape: {x_hat_matched.shape}")  
                    print(f"posterior_cov_matched shape: {posterior_cov_matched.shape}")

                    # Now all should have consistent dimensions
                    assert x_matched.shape[0] == x_hat_matched.shape[0] == posterior_cov_matched.shape[0]
                    
                    # Compute confidence intervals
                    ci_lower_active, ci_upper_active, counts_within_ci_active, empirical_coverage_active = \
                        self.uncertainty_estimator.get_confidence_intervals_data(
                            x=x_matched,
                            x_hat=x_hat_matched,
                            posterior_cov=posterior_cov_matched,
                            orientation_type=data_params.get("orientation_type")
                        )
                        
                    # Normalize by the number of sources and take the first orientation and time step
                    # time_idx = 0 # As we average x across time, the time index is always 0.
                    # all_sources_empirical_coverage = (counts_within_ci / len(x_avg_time))[:, 0, time_idx]
                    
                    # empirical_coverage_active = (counts_within_ci_active / len(matched_absolute_indices))[:, 0, time_idx]
                
                    # empirical_coverage_active = (counts_within_ci_active / len(x_hat_active_indices))[:, 0, time_idx]    

                # -------------------------------------------------------------
                # 9. Evaluate metrics and store results
                # -------------------------------------------------------------
                
                # # --- Evaluate metrics: ALL SOURCES (full posterior covariance) ---
                # self.metric_evaluator.evaluate_and_store_metrics(
                #     this_result,
                #     metric_suffix="_all_sources",
                #     empirical_coverage=all_sources_empirical_coverage,
                #     cov=full_posterior_cov # not full posterior_cov since we use the diagonal elements for mean_posterior_std metric????
                # )

                # --- Evaluate metrics: ACTIVE SOURCES (estimated active set) ---
                # empirical_coverage_active = (counts_within_ci_active / len(x_hat_active_indices))[:, 0, time_idx]
                
                            
                self.metric_evaluator.evaluate_and_store_metrics(
                    this_result,
                    metric_suffix="_active_indices",
                    empirical_coverage=empirical_coverage_active,  
                    cov=posterior_cov, #_matched
                )

                # --------------------------------------------------------------
                # 10. Vizualization
                # --------------------------------------------------------------
                self.visualize(
                    x_trials=x_trials,
                    active_indices_trials=x_active_indices_trials,
                    x_avg_time=x_avg_time,
                    x_hat_avg_time=x_hat_avg_time,
                    x_active_indices=x_active_indices,
                    x_hat_active_indices=x_hat_active_indices,
                    n_sources=n_sources,
                    y_clean=y_clean_trials,
                    y_noisy=y_noisy_trials,
                    experiment_dir=experiment_dir,
                    trial_idx=trial_idx,
                    source_units=source_units,
                    sensor_units=sensor_units,
                    confidence_levels=self.uncertainty_estimator.confidence_levels,
                    empirical_coverages={
                        # 'all_sources': all_sources_empirical_coverage,
                        'active_indices': empirical_coverage_active
                    },
                    ci_lower=ci_lower_active,
                    ci_upper=ci_upper_active,
                    orientation_type=data_params.get("orientation_type"),
                    result=this_result
                )

                # self.uncertainty_estimator.plot_sorted_posterior_variances(top_k=10)
                # self.uncertainty_estimator.visualize_sorted_covariances(top_k=10)
                # self.uncertainty_estimator.plot_posterior_covariance_matrix()

                '''
                # --- Plotting ---

                # active sources
                self.uncertainty_estimator.visualize_calibration_curve(
                    empirical_coverage_active,
                    this_result,
                    which_legend="active_indices",
                    filename="Calibration_curve_active_indices"
                )
                # if self.data_param_grid.get("orientation_type") == "free":
                #     self.uncertainty_estimator.plot_top_relevant_CE_pairs(top_k=5, confidence_level=0.95)
                '''
            
            except Exception as e:
                self.logger.error(f"Error during benchmarking run_id {this_result.get('run_id', 'N/A')}: {e}", exc_info=True)
                this_result["error_message"] = str(e) # More specific error key
            
            results_list.append(this_result)
            self.logger.info(f"Completed run_id {this_result.get('run_id', 'N/A')}")

        self.logger.info("Benchmarking completed.")
        return pd.DataFrame(results_list)

    def visualize(
        self,
        x_trials: np.ndarray,
        active_indices_trials: np.ndarray,
        x_avg_time: np.ndarray,
        x_hat_avg_time: np.ndarray,
        x_active_indices: np.ndarray,
        x_hat_active_indices: np.ndarray,
        n_sources: int,
        y_clean: np.ndarray,
        y_noisy: np.ndarray,
        experiment_dir: str,
        trial_idx: int = 0,
        source_units: str = FIFF.FIFF_UNIT_AM,
        sensor_units: str = FIFF.FIFF_UNIT_V,
        confidence_levels: np.ndarray = None,
        empirical_coverages: dict = None,
        ci_lower: np.ndarray = None,
        ci_upper: np.ndarray = None,
        orientation_type: str = "fixed",
        result: dict = None
    ):

        viz = Visualizer(base_save_path=experiment_dir, logger=self.logger)

        # =========================
        # 1. Plot simulated data
        # =========================
        
        # Plot sources (all trials)
        viz.plot_source_signals(
            ERP_config=self.ERP_config,
            x=x_trials,
            active_indices=active_indices_trials,
            units=source_units,
            trial_idx=None,
            title="Source Trials (All)",
            save_dir="data_simulation",
            file_name="source_trials_all",
            show=False,
        )

        # Plot sensors (all trials) with selected channels: y_noisy
        viz.plot_sensor_signals(
            ERP_config=self.ERP_config,
            y_trials=y_noisy,
            channels="all",  # or "all"
            units=sensor_units,
            mode="stack",
            title="Sensor Signals (All Trials stacked)",
            save_dir="data_simulation",
            file_name="sensor_stack_trials_noisy",
            show=False
        )

        # Concatenated trials: y_noisy
        viz.plot_sensor_signals(
            ERP_config=self.ERP_config,
            y_trials=y_noisy,
            # channels=[0, 10],  # or "all"
            units=sensor_units,
            mode="concatenate",
            title="Sensor Signals (All trials concatenated)",
            save_dir="data_simulation",
            file_name="sensor_concatenate_trials_noisy",
            show=False
        )

        # Concatenated trials: y_clean
        viz.plot_sensor_signals(
            ERP_config=self.ERP_config,
            y_trials=y_clean,
            # channels=[0, 10],  # or "all"
            units=sensor_units,
            mode="concatenate",
            title="Sensor Signals (All trials concatenated)",
            save_dir="data_simulation",
            file_name="sensor_concatenate_trials_clean",
            show=False
        )
        
        # =========================
        # 2. Plot uncertainty analysis figures
        # =========================
        # Plot active sources
        viz.plot_active_sources(
            x=x_avg_time,
            x_hat=x_hat_avg_time,
            x_active_indices=x_active_indices,
            x_hat_active_indices=x_hat_active_indices,
            n_sources=n_sources,
            source_units=source_units,
            orientation_type= orientation_type,
            save_path="uncertainty_analysis",
            file_name="active_sources",
            show=False
        )
        # Plot confidence intervals - unshared y-axis
        viz.plot_ci(
            x=x_avg_time,
            x_hat=x_hat_avg_time,
            x_active_indices=x_active_indices,
            x_hat_active_indices=x_hat_active_indices,
            n_sources=n_sources,
            source_units=source_units,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_levels=confidence_levels,
            orientation_type=orientation_type,
            sharey=False,
            save_path="uncertainty_analysis",
            file_name="confidence_intervals",
            show=False,
            figsize=(18, 13)
        )

        # Plot confidence intervals - shared y-axis
        viz.plot_ci(
            x=x_avg_time,
            x_hat=x_hat_avg_time,
            x_active_indices=x_active_indices,
            x_hat_active_indices=x_hat_active_indices,
            n_sources=n_sources,
            source_units=source_units,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_levels=confidence_levels,
            orientation_type=orientation_type,
            sharey=True,
            save_path="uncertainty_analysis",
            file_name="confidence_intervals_yshare",
            show=False,
            figsize=(18, 13)
        )
        
        # plot calibration curve - all sources
        # viz.plot_calibration_curve(
        #     confidence_levels=confidence_levels,
        #     empirical_coverage=empirical_coverages['all_sources'],
        #     result=result, 
        #     which_legend="all_sources", # or "active_indices"
        #     filename='calibration_curve_all_sources',
        #     save_path='uncertainty_analysis',
        #     show=True,
        # )

        # plot calibration curve - active sources
        viz.plot_calibration_curve(
            confidence_levels=confidence_levels,
            empirical_coverage=empirical_coverages['active_indices'],
            result=result, 
            which_legend="active_indices", # or "all_sources"
            filename='calibration_curve_active_sources',
            save_path='uncertainty_analysis',
            show=True,
        )
