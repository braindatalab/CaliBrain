import datetime
import os
import logging
from typing import Optional
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.utils import check_random_state
from itertools import product
from joblib import Parallel, delayed
import mne
from itertools import combinations
from scipy.stats import wishart
from matplotlib.patches import Ellipse
from scipy.stats import chi2

from calibrain import MetricEvaluator
from calibrain import SourceEstimator, SensorSimulator, SourceSimulator, UncertaintyEstimator, Visualizer, LeadfieldBuilder, gamma_map, eloreta, UncertaintyCalibrator
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

    def __getstate__(self):
        state = self.__dict__.copy()
        state['logger'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.logger is None:
            self.logger = logging.getLogger(__name__)
    
    def _log_run_progress(
        self,
        run_in_config: int,
        nruns_local: int,
        config_index: int,
        num_configs: int,
        run_id: int,
        total_runs: int,
        solver_name: str,
        noise_type: Optional[str],
        nnz: Optional[int],
        alpha_snr: Optional[float],
        final: bool = False,
    ) -> None:
        """Log per-run progress and an optional separator summary when the
        current run is the final run of the current configuration.

        Parameters mirror the values computed in `_execute_single_run`.
        """
        try:
            # single-line progress message
            self.logger.info(
                "[%d/%d | config %d/%d | %d/%d] Estimator: %s | noise_type: %s | nnz: %s | alpha_SNR: %s",
                run_in_config,
                nruns_local,
                config_index,
                num_configs,
                run_id,
                total_runs,
                solver_name,
                noise_type,
                nnz,
                alpha_snr,
            )

            if final:
                # separator and short summary for completed configuration
                self.logger.info("%s", "=" * 80)
                self.logger.info(
                    "Finished %d runs for config %d/%d — Estimator=%s | noise_type=%s | nnz=%s | alpha_SNR=%s",
                    nruns_local,
                    config_index,
                    num_configs,
                    solver_name,
                    noise_type,
                    nnz,
                    alpha_snr,
                )
        except Exception:
            # tolerate logging errors — do not break benchmarking
            try:
                # fallback to basic debug message
                self.logger.debug("Progress logging failed for run %s", run_id)
            except Exception:
                pass
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
    
        self.logger.debug(f"Experiment directory created: {experiment_dir}")
        return experiment_dir

    def _prepare_run_data(self, data_params: dict, seed: int, n_trials: int) -> dict:
        data_params = dict(data_params)
        orientation_type = data_params.get("orientation_type")
        leadfield_data = self.leadfield_builder.get_leadfield(
            subject=data_params['subject'],
            orientation_type=orientation_type,
            retrieve_mode="load",
            return_metadata=True,
        )
        L = leadfield_data.leadfield
        if orientation_type == "fixed":
            n_sensors, n_sources = L.shape
        else:
            n_sensors, n_sources = L.shape[:2]

        self.sensor_simulator.set_sensor_metadata(
            kind=leadfield_data.sensor_kind,
            coil_type=leadfield_data.coil_type,
            units=leadfield_data.sensor_units,
            unitmult=leadfield_data.sensor_unitmult,
        )

        global_source_rng = np.random.RandomState(seed)
        global_source_seeds = global_source_rng.randint(0, 2 ** 32 - 1, n_trials)
        x_trials, x_active_indices_trials = self.source_simulator.simulate(
            orientation_type=orientation_type,
            n_sources=n_sources,
            nnz=data_params['nnz'],
            n_trials=n_trials,
            global_seed=global_source_seeds,
        )
        source_units = self.source_simulator.units
        source_unitmult = self.source_simulator.unitmult

        global_noise_rng = np.random.RandomState(seed + 123456)
        global_noise_seeds = global_noise_rng.randint(0, 2 ** 32 - 1, size=n_trials)
        y_clean_trials, y_noisy_trials, noise_trials, noise_eta_trials = self.sensor_simulator.simulate(
            x_trials=x_trials,
            L=L,
            orientation_type=orientation_type,
            alpha_SNR=data_params['alpha_SNR'],
            sensor_white_noise_var=data_params['sensor_white_noise_var'],
            n_trials=n_trials,
            global_seed=global_noise_seeds,
        )

        return {
            "leadfield": L,
            "sensor_metadata": {
                "kind": leadfield_data.sensor_kind,
                "coil_type": leadfield_data.coil_type,
                "units": leadfield_data.sensor_units,
                "unitmult": leadfield_data.sensor_unitmult,
            },
            "x_trials": x_trials,
            "x_active_indices_trials": x_active_indices_trials,
            "y_clean_trials": y_clean_trials,
            "y_noisy_trials": y_noisy_trials,
            "noise_eta_trials": noise_eta_trials,
            "source_units": source_units,
            "source_unitmult": source_unitmult,
        }

    def run(self, nruns: int = 2, fig_path: str = "results/figures", n_jobs: int = 1):
        """
        Run benchmarking by iterating over combinations of solver and data parameters.

        Parameters
        ----------
        nruns : int
            Number of seeds to evaluate for each parameter combination.
        fig_path : str
            Base directory where per-run visualizations will be saved.
        n_jobs : int
            Number of parallel workers to use. ``1`` (default) keeps the sequential behaviour.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the results for each parameter combination.
        """
        rng = check_random_state(self.random_state)
        seeds = rng.randint(low=0, high=2 ** 32, size=nruns)
        param_combinations = list(product(
            ParameterGrid(self.solver_param_grid),
            ParameterGrid(self.data_param_grid),
            ParameterGrid(self.noise_param_grid),
            seeds
        ))
        total_runs = len(param_combinations)
        # compute number of parameter configurations for logging
        num_solver = len(ParameterGrid(self.solver_param_grid))
        num_data = len(ParameterGrid(self.data_param_grid))
        num_noise = len(ParameterGrid(self.noise_param_grid))
        num_configs = num_solver * num_data * num_noise
        self.logger.info(
            "Starting benchmark with %d experiments (%d nruns x %d configurations)",
            total_runs,
            nruns,
            num_configs,
        )
        if total_runs == 0:
            return pd.DataFrame()

        n_jobs = max(1, int(n_jobs or 1))
        worker_args = [
            (run_id, nruns, total_runs, solver_params, data_params, noise_params, seed, fig_path)
            for run_id, (solver_params, data_params, noise_params, seed) in enumerate(param_combinations, start=1)
        ]

        if n_jobs == 1:
            results_list = [self._execute_single_run(*args) for args in worker_args]
        else:
            self.logger.debug(f"Executing benchmark with n_jobs={n_jobs} using joblib Parallel (loky backend)...")
            parallel = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)
            results_list = parallel(
                delayed(self._execute_single_run)(*args)
                for args in worker_args
            )

        self.logger.info("Benchmarking completed.")
        return pd.DataFrame(results_list)

    def _execute_single_run(
        self,
        run_id: int,
        nruns: int,
        total_runs: int,
        solver_params: dict,
        data_params: dict,
        noise_params: dict,
        seed: int,
        fig_path: str,
        n_trials: int = 5,
    ) -> dict:
        solver_params = dict(solver_params)
        data_params = dict(data_params)
        noise_params = dict(noise_params)
        solver_name = getattr(self.solver, "__name__", str(self.solver))
        orientation_type = data_params.get("orientation_type")
        n_orient = 3 if orientation_type == "free" else 1
        solver_params['fwd_path'] = get_data_path() / 'fwd'

        # Format human-friendly, 1-based progress counters and avoid division-by-zero
        nruns_local = max(1, int(nruns))
        # number of parameter configurations (ceil division)
        num_configs = max(1, (total_runs + nruns_local - 1) // nruns_local)
        config_index = (run_id - 1) // nruns_local + 1
        run_in_config = (run_id - 1) % nruns_local + 1

        # extract noise_type, nnz and alpha_SNR early for logging
        noise_type = noise_params.get("noise_type")
        nnz = data_params.get("nnz")
        alpha_snr = data_params.get("alpha_SNR")

        # delegate logging to helper (keeps formatting in one place)
        self._log_run_progress(
            run_in_config=run_in_config,
            nruns_local=nruns_local,
            config_index=config_index,
            num_configs=num_configs,
            run_id=run_id,
            total_runs=total_runs,
            solver_name=solver_name,
            noise_type=noise_type,
            nnz=nnz,
            alpha_snr=alpha_snr,
            final=False,
        )
        self.logger.debug(f"Solver params: {solver_params}")
        self.logger.debug(f"Noise params: {noise_params}")
        self.logger.debug(f"Data params: {data_params}")

        this_result = {
            'run_id': run_id,
            "seed": seed,
            "solver": solver_name,
            'noise_type': noise_params['noise_type'],
            **{k: v for k, v in solver_params.items() if k != "fwd_path"},
            **data_params,
        }

        try:
            experiment_dir = self.create_experiment_directory(
                base_dir=fig_path,
                params=this_result,
                desired_order=[
                    "solver", "noise_type", "alpha_SNR", "orientation_type", "nnz", "sensor_white_noise_var", "subject", "seed"
                ]
            )

            prepared_data = self._prepare_run_data(data_params, seed, n_trials)
            L = prepared_data["leadfield"]
            sensor_meta = prepared_data["sensor_metadata"]
            sensor_kind = sensor_meta.get("kind")
            sensor_coil_type = sensor_meta.get("coil_type")
            sensor_units = sensor_meta.get("units")
            sensor_unitmult = sensor_meta.get("unitmult")
            self.sensor_simulator.set_sensor_metadata(
                kind=sensor_kind,
                coil_type=sensor_coil_type,
                units=sensor_units,
                unitmult=sensor_unitmult,
            )

            if orientation_type == "fixed":
                n_sensors, n_sources = L.shape
            else:
                n_sensors, n_sources = L.shape[:2]

            x_trials = prepared_data["x_trials"]
            x_active_indices_trials = prepared_data["x_active_indices_trials"]
            y_clean_trials = prepared_data["y_clean_trials"]
            y_noisy_trials = prepared_data["y_noisy_trials"]
            noise_eta_trials = prepared_data["noise_eta_trials"]
            source_units = prepared_data["source_units"]
            source_unitmult = prepared_data["source_unitmult"]

            trial_idx = 0
            x_one_trial = x_trials[trial_idx]
            x_active_indices_one_trial = x_active_indices_trials[trial_idx]
            y_noisy_one_trial = y_noisy_trials[trial_idx]
            noise_eta_one_trial = noise_eta_trials[trial_idx]

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
                baseline_noise_var = 1.0

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
                grid_factors = np.logspace(-2, 2, 20)
                alphas = baseline_noise_var * grid_factors
                noise_variances = np.unique(alphas).tolist() or [baseline_noise_var]
                estimator = SpatialCVSolver if noise_type == 'spatial_cv' else TemporalCVSolver
                source_estimator = estimator(
                    solver=self.solver,
                    solver_params=solver_params,
                    n_orient=n_orient,
                    logger=self.logger,
                    noise_variances=noise_variances,
                    cv=noise_params.get('cv', 5),
                    n_jobs=noise_params.get('n_jobs', 1),
                )
            else:
                if noise_type == 'oracle':
                    self.logger.debug("Using oracle noise variance estimate")
                    noise_var = float((data_params['sensor_white_noise_var'] * noise_eta_one_trial) ** 2)
                elif noise_type == 'baseline':
                    self.logger.debug("Using baseline noise variance estimate")
                    noise_var = baseline_noise_var
                    self.logger.debug(
                        f"Baseline noise variance (trial {run_id}): {noise_var:.3e}, eta: {noise_eta_one_trial:.3e}"
                    )
                elif noise_type == 'joint_learning':
                    noise_var = None

                source_estimator = SourceEstimator(
                    solver=self.solver,
                    solver_params=solver_params,
                    noise_var=noise_var,
                    n_orient=n_orient,
                    logger=self.logger
                )

            self.logger.debug(f"Fitting source estimator {self.solver.__name__}")
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
            this_result['avg_posterior_variance'] = np.mean(posterior_var)
            this_result['std_posterior_variance'] = np.std(np.sqrt(posterior_var))
            this_result['avg_posterior_mean'] = np.mean(x_hat_one_trial)
            this_result['std_posterior_mean'] = np.std(x_hat_one_trial)

            self.logger.debug("Estimating uncertainty...")
            if solver_name == 'gamma_map':
                self.logger.debug("Skipping uncertainty estimation for gamma_map solver.")
                this_result['active_indices_size'] = (
                    len(x_hat_active_indices_one_trial)
                    if x_hat_active_indices_one_trial is not None
                    else 0
                )
                return this_result

            x_one_trial_avg_time = np.mean(x_one_trial, axis=1, keepdims=True)
            x_hat_one_trial_avg_time = np.mean(x_hat_one_trial, axis=1, keepdims=True)
            n_times = x_one_trial.shape[1]
            posterior_var_avg_time = posterior_var / n_times
            posterior_std_avg_time = np.sqrt(np.maximum(posterior_var_avg_time, 0.0))

            calibrator = UncertaintyCalibrator(
                uncertainty_estimator=self.uncertainty_estimator,
                metric_evaluator=self.metric_evaluator,
                n_folds=5,
                random_state=self.random_state,
            )
            calibration_results = calibrator.calibrate(
                x_true=x_one_trial_avg_time,
                x_hat=x_hat_one_trial_avg_time,
                posterior_std=posterior_std_avg_time,
            )
            pre_calibration = calibration_results['pre_calibration']
            post_calibration = calibration_results['post_calibration']

            metric_kwargs = dict(
                x=x_one_trial_avg_time,
                x_hat=x_hat_one_trial_avg_time,
                posterior_var=posterior_var_avg_time,
                orientation_type=orientation_type,
                nnz=data_params.get("nnz"),
                subject=data_params.get("subject"),
                fwd_path=solver_params['fwd_path'],
            )

            try:
                metric_results_pre_cal = self.metric_evaluator.evaluate_metrics(
                    which="evaluation",
                    empirical_coverages=pre_calibration['empirical_coverages'],
                    **metric_kwargs,
                )
                suffixed_pre = {f"pre_cal_{k}": v for k, v in metric_results_pre_cal.items()}
                this_result.update(suffixed_pre)

                metric_results_post_cal = self.metric_evaluator.evaluate_metrics(
                    which="evaluation",
                    empirical_coverages=post_calibration['empirical_coverages'],
                    **metric_kwargs,
                )
                suffixed_post = {f"post_cal_{k}": v for k, v in metric_results_post_cal.items()}
                this_result.update(suffixed_post)

                for k in self.metric_evaluator.evaluation_metrics:
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

            calibration_metric_names = tuple(
                getattr(self.metric_evaluator, "calibration_metrics", tuple())
            )
            pre_cal_metrics = pre_calibration.get('metrics', {})
            post_cal_metrics = post_calibration.get('metrics', {})
            for metric_name in calibration_metric_names:
                pre_value = pre_cal_metrics.get(metric_name)
                post_value = post_cal_metrics.get(metric_name)
                if pre_value is not None:
                    this_result[f"pre_cal_{metric_name}"] = pre_value
                if post_value is not None:
                    this_result[f"post_cal_{metric_name}"] = post_value
                improvement_key = f"improvement_{metric_name}"
                if (
                    pre_value is None
                    or post_value is None
                    or (isinstance(pre_value, (int, float, np.floating)) and np.isclose(pre_value, 0.0))
                ):
                    this_result[improvement_key] = None
                else:
                    this_result[improvement_key] = (pre_value - post_value) / pre_value * 100

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
                fwd_path=solver_params['fwd_path'],
                nnz=data_params.get("nnz"),
                ERP_config=self.ERP_config,
                sample_idx=200,
                source_units=source_units,
                source_unitmult=source_unitmult,
                sensor_units=sensor_units,
                confidence_levels=self.uncertainty_estimator.nominal_coverages,
                nominal_coverages=pre_calibration['nominal_coverages'],
                empirical_coverages=pre_calibration['empirical_coverages'],
                empirical_coverages_post_cal=post_calibration['empirical_coverages'],
                ci_lower=pre_calibration.get('ci_lowers'),
                ci_upper=pre_calibration.get('ci_uppers'),
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
            self.logger.error(
                f"Error during benchmarking run_id {this_result.get('run_id', 'N/A')}: {e}",
                exc_info=True,
            )
            this_result["error_message"] = str(e)

        # print a separator when we've finished the last run of a configuration
        try:
            if run_in_config == nruns_local:
                self._log_run_progress(
                    run_in_config=run_in_config,
                    nruns_local=nruns_local,
                    config_index=config_index,
                    num_configs=num_configs,
                    run_id=run_id,
                    total_runs=total_runs,
                    solver_name=solver_name,
                    noise_type=noise_type,
                    nnz=nnz,
                    alpha_snr=alpha_snr,
                    final=True,
                )
        except Exception:
            # tolerate logging errors — do not break benchmarking
            pass

        self.logger.debug(f"Completed run_id {this_result.get('run_id', 'N/A')}")
        return this_result
