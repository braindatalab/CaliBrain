import datetime
import json
import os
import logging
import h5py
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
from calibrain import SourceEstimator, SensorSimulator, SourceSimulator, UncertaintyEstimator, Visualizer, LeadfieldBuilder, UncertaintyCalibrator
from calibrain.utils import get_data_path, inspect_object
from mne.io.constants import FIFF
from calibrain.calibration_storage import save_calibration_record
from calibrain.calibration_dataset import EEG_COIL_TYPES, MEG_COIL_TYPES

# Suppress verbose MNE console output in worker processes (joblib spawns new interpreters)
logging.getLogger("mne").setLevel(logging.ERROR)
logging.getLogger("mne.utils").setLevel(logging.ERROR)

class DataGenerator:
    def __init__(
        self,
        solver: callable,
        solver_param_grid: dict,
        data_param_grid: dict,
        noise_param_grid: dict,
        ERP_config: dict,
        source_simulator: SourceSimulator,
        leadfield_builder: LeadfieldBuilder,
        sensor_simulator: SensorSimulator,
        save_posterior_stats: bool = True,
        posterior_dir: str | Path | None = None,
        random_state=42,
        logger=None,
    ):
        """
        Initialize the DataGenerator class.

        Parameters
        ----------
        solver : callable
            The solver function (e.g., ``gamma_map_sflex`` or ``BMN``).
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
        save_posterior_stats : bool, optional
            If True, persist per-run posterior summaries for later aggregation.
        posterior_dir : str or Path, optional
            Directory where posterior summary files should be stored. Defaults
            to the per-run experiment directory when omitted.
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
        self.save_posterior_stats = save_posterior_stats
        self.posterior_dir = Path(posterior_dir) if posterior_dir else None
        if self.posterior_dir is not None:
            self.posterior_dir.mkdir(parents=True, exist_ok=True)
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
        global_run_id: Optional[int],
        global_total_runs: Optional[int],
        solver_name: str,
        noise_type: Optional[str],
        nnz: Optional[int],
        alpha_snr: Optional[float],
    ) -> None:
        """Log per-run progress and an optional separator summary when the
        current run is the final run of the current configuration.

        Parameters mirror the values computed in `_execute_single_run`.
        """
        try:
            self.logger.info(
                "[run: %d/%d | config: %d/%d | total: %d/%d] %s | %s | %s NNZ | %s SNR",
                run_in_config,
                nruns_local,
                config_index,
                num_configs,
                global_run_id if global_run_id is not None else run_id,
                global_total_runs if global_total_runs is not None else total_runs,
                solver_name,
                noise_type,
                nnz,
                alpha_snr,
            )
        except Exception:
            # tolerate logging errors — do not break generation
            try:
                # fallback to basic debug message
                self.logger.debug("Progress logging failed for run %s", run_id)
            except Exception:
                pass
            
    def _create_experiment_directory(self, base_dir, params, desired_order):
        """
        Create a directory structure for the experiment, with subdirectories for each parameter in a specified order, followed by any remaining parameters.
    
        Parameters
        ----------
        base_dir : str
            Base directory for the experiment.
        params : dict
            Dictionary of parameters.
        desired_order : list
            List of parameter keys in the desired order for the directory structure.
    
        Returns
        -------
        experiment_dir : str
            Path to the experiment directory.
        """
        # Exclude 'cov' and sanitize values for directory names
        sanitized_params_for_path = {
            k: str(v).replace("/", "_").replace("\\", "_").replace(" ", "_")
            for k, v in params.items() if k not in ("run_id", "global_run_id")
            # Add other keys to exclude from path if necessary
        }
    
        # Desired order of parameters for the directory structure
        # This list defines the specific order.
        if desired_order is None:
            desired_order = ["solver", "noise_type", "orientation_type", "alpha_SNR", "subject", "nnz", "seed"]
        
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

    def _build_calibration_metadata(
        self,
        *,
        solver_name: str,
        solver_params: dict,
        data_params: dict,
        noise_params: dict,
        seed: int,
        experiment_dir: str | Path,
        global_run_id: int,
        config_index: int,
        run_in_config: int,
        nruns_local: int,
    ) -> dict:
        timestamp = datetime.datetime.utcnow().isoformat()
        return {
            "global_run_id": global_run_id,
            "config_index": config_index,
            "run_in_config": run_in_config,
            "nruns_per_config": nruns_local,
            "seed": seed,
            "timestamp_utc": timestamp,
            "solver": solver_name,
            "solver_params": dict(solver_params),
            "data_params": dict(data_params),
            "noise_params": dict(noise_params),
            "noise_type": noise_params.get("noise_type"),
            "subject": data_params.get("subject"),
            "nnz": data_params.get("nnz"),
            "alpha_SNR": data_params.get("alpha_SNR"),
            "orientation_type": data_params.get("orientation_type"),
            "experiment_dir": Path(experiment_dir).as_posix(),
        }

    def _persist_calibration_results(
        self,
        *,
        experiment_dir: str | Path,
        record_dir: str | Path,
        solver_name: str,
        solver_params: dict,
        data_params: dict,
        noise_params: dict,
        seed: int,
        global_run_id: int,
        config_index: int,
        run_in_config: int,
        nruns_local: int,
        pre_calibration: dict,
        post_calibration: dict,
    ) -> Optional[Path]:
        record_dir = Path(record_dir)
        record_name = f"calibration_run-{global_run_id:05d}"
        metadata = self._build_calibration_metadata(
            solver_name=solver_name,
            solver_params=solver_params,
            data_params=data_params,
            noise_params=noise_params,
            seed=seed,
            experiment_dir=experiment_dir,
            global_run_id=global_run_id,
            config_index=config_index,
            run_in_config=run_in_config,
            nruns_local=nruns_local,
        )
        try:
            return save_calibration_record(
                output_dir=record_dir,
                record_name=record_name,
                metadata=metadata,
                pre_calibration=pre_calibration,
                post_calibration=post_calibration,
            )
        except Exception as exc:
            self.logger.warning(
                "Failed to store calibration record for global_run_id %s: %s",
                global_run_id,
                exc,
            )
            return None

    @staticmethod
    def _sanitize_metadata(metadata: dict | None) -> dict:
        if not metadata:
            return {}

        def _convert(value):
            if isinstance(value, (np.generic,)):
                return value.item()
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, Path):
                return value.as_posix()
            if isinstance(value, dict):
                return {k: _convert(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_convert(v) for v in value]
            return value

        return {k: _convert(v) for k, v in metadata.items()}

    def _persist_posterior_summary(
        self,
        *,
        experiment_dir: str | Path,
        datasets: dict,
        metadata: dict | None = None,
        posterior_dir: str | Path | None = None,
        filename: str | None = None,
    ) -> Optional[Path]:
        base_dir = Path(posterior_dir) if posterior_dir else Path(experiment_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        summary_filename = filename or "posterior_summary.h5"
        summary_path = base_dir / summary_filename
        try:
            with h5py.File(summary_path, "w") as handle:
                for name, value in datasets.items():
                    dataset_kwargs = {"data": value}
                    if isinstance(value, np.ndarray):
                        if value.ndim == 0:
                            pass
                        else:
                            dataset_kwargs["compression"] = "gzip"
                    else:
                        value_arr = np.asarray(value)
                        if value_arr.ndim == 0:
                            dataset_kwargs["data"] = value_arr
                        else:
                            dataset_kwargs["data"] = value_arr
                            dataset_kwargs["compression"] = "gzip"
                    handle.create_dataset(name, **dataset_kwargs)
                safe_metadata = self._sanitize_metadata(metadata)
                if safe_metadata:
                    handle.attrs["metadata_json"] = json.dumps(safe_metadata)
            return summary_path
        except Exception as exc:
            self.logger.warning(
                "Failed to store posterior summary at %s: %s", summary_path, exc
            )
            return None

    def _prepare_run_data(self, data_params: dict, seed: int) -> dict:
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

        source_seed = seed
        sensor_seed = (seed * 26544) % (2**32)

        x, x_active_indices = self.source_simulator.simulate(
            n_sources=n_sources,
            nnz=data_params['nnz'],
            orientation_type=orientation_type,
            coil_type=leadfield_data.coil_type,
            seed=source_seed,
        )
        source_units = self.source_simulator.units
        source_unitmult = self.source_simulator.unitmult

        y_clean, y_noisy, noise, noise_eta = self.sensor_simulator.simulate(
            x=x,
            L=L,
            alpha_SNR=data_params['alpha_SNR'],
            sensor_white_noise_std=data_params['sensor_white_noise_std'],
            seed=sensor_seed,
        )

        return {
            "leadfield": L,
            "src_coords": leadfield_data.src_coords,
            "sensor_metadata": {
                "kind": leadfield_data.sensor_kind,
                "coil_type": leadfield_data.coil_type,
                "units": leadfield_data.sensor_units,
                "unitmult": leadfield_data.sensor_unitmult,
            },
            "Q_basis": leadfield_data.Q_basis,
            "x": x,
            "x_active_indices": x_active_indices,
            "y_clean": y_clean,
            "y_noisy": y_noisy,
            "noise": noise,
            "noise_eta": noise_eta,
            "source_units": source_units,
            "source_unitmult": source_unitmult,
        }

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
        global_run_id: Optional[int] = None,
        global_total_runs: Optional[int] = None,
    ) -> dict:
        # Ensure worker processes emit INFO logs to stdout/file (joblib workers start with default WARNING level)
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            log_handlers = [logging.StreamHandler()]
            log_file_env = os.environ.get("CALIBRAIN_LOG_FILE")
            if log_file_env:
                try:
                    log_handlers.insert(0, logging.FileHandler(log_file_env, mode="a"))
                except Exception:
                    pass
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                handlers=log_handlers,
            )
        root_logger.setLevel(logging.INFO)
        solver_params = dict(solver_params)
        data_params = dict(data_params)
        noise_params = dict(noise_params)
        solver_name = getattr(self.solver, "__name__", str(self.solver))
        orientation_type = data_params.get("orientation_type")
        solver_params['fwd_path'] = get_data_path() / '1284src_fwd' / data_params['subject']

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
        if global_run_id is None:
            global_run_id = run_id
        self._log_run_progress(
            run_in_config=run_in_config,
            nruns_local=nruns_local,
            config_index=config_index,
            num_configs=num_configs,
            run_id=run_id,
            total_runs=total_runs,
            global_run_id=global_run_id,
            global_total_runs=global_total_runs,
            solver_name=solver_name,
            noise_type=noise_type,
            nnz=nnz,
            alpha_snr=alpha_snr,
        )
        self.logger.debug(f"Solver params: {solver_params}")
        self.logger.debug(f"Noise params: {noise_params}")
        self.logger.debug(f"Data params: {data_params}")

        this_result = {
            'run_id': run_in_config,
            'global_run_id': global_run_id,
            "seed": seed,
            "solver": solver_name,
            'noise_type': noise_params['noise_type'],
            **{k: v for k, v in solver_params.items() if k != "fwd_path"},
            **data_params,
        }

        try:
            experiment_dir = self._create_experiment_directory(
                base_dir=fig_path,
                params=this_result,
                desired_order=[
                    "orientation_type", "solver", "noise_type", "nnz", "subject", "alpha_SNR", "seed"
                ]
            )

            prepared_data = self._prepare_run_data(data_params, seed)
            L = prepared_data["leadfield"]
            src_coords = prepared_data["src_coords"]
            sensor_meta = prepared_data["sensor_metadata"]
            sensor_kind = sensor_meta.get("kind")
            sensor_coil_type = sensor_meta.get("coil_type")
            sensor_units = sensor_meta.get("units")
            sensor_unitmult = sensor_meta.get("unitmult")
            # Persist sensor metadata in the per-run results so the CSV manifest
            # can be used for filtering without opening the H5 summaries.
            this_result["sensor_kind"] = sensor_kind
            this_result["coil_type"] = sensor_coil_type
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

            x = prepared_data["x"]
            x_active_indices = prepared_data["x_active_indices"]
            y_clean = prepared_data["y_clean"]
            y_noisy = prepared_data["y_noisy"]
            noise = prepared_data["noise"]
            noise_eta = prepared_data["noise_eta"]
            source_units = prepared_data["source_units"]
            source_unitmult = prepared_data["source_unitmult"]
            
            n_times = x.shape[-1]
            this_result["n_sources"] = int(n_sources)
            this_result["n_times"] = int(n_times)
            
            if orientation_type == "fixed":
                n_orient = 1
            else:
                if x.ndim != 3:
                    raise ValueError(
                        f"Free-orientation simulations must have shape (N, comps, T); got {x.shape}"
                    )
                n_orient = x.shape[1]

            tmin = self.source_simulator.ERP_config['tmin']
            stim_onset = self.source_simulator.ERP_config['stim_onset']
            sfreq = self.source_simulator.ERP_config['sfreq']
            pre_stimulus_onset = int((stim_onset - tmin) * sfreq)
            if pre_stimulus_onset <= 0:
                self.logger.warning(
                    "Computed pre_stimulus_onset <= 0; using full trial for baseline estimation"
                )
                y_pre = y_noisy
            else:
                y_pre = y_noisy[:, :pre_stimulus_onset]

            try:
                baseline_noise_var = float(np.mean(np.std(y_pre, axis=1) ** 2))
            except Exception:
                baseline_noise_var = None
            if not baseline_noise_var or not np.isfinite(baseline_noise_var):
                baseline_noise_var = 1.0

            allowed_noise_types = {
                "adaptive_joint_learning",
                "oracle",
                "baseline",
            }
            
            solver_params['src_coords'] = src_coords
            
            noise_type = noise_params.get("noise_type")
            if noise_type not in allowed_noise_types:
                raise ValueError(f"Invalid noise_type: {noise_type!r}. Allowed: {sorted(allowed_noise_types)}")

            if noise_type == 'oracle':
                self.logger.debug("Using oracle noise variance estimate")
                noise_var = float(np.var(noise))
            elif noise_type == 'baseline':
                self.logger.debug("Using baseline noise variance estimate")
                noise_var = baseline_noise_var
                self.logger.debug(
                    f"Baseline noise variance (global run {run_id}, config run {run_in_config}): {noise_var:.3e}, eta: {noise_eta:.3e}"
                )
            elif noise_type == 'adaptive_joint_learning':
                noise_var = None

            source_estimator = SourceEstimator(
                solver=self.solver,
                solver_params=solver_params,
                noise_var=noise_var,
                n_orient=n_orient,
                logger=self.logger
            )

            self.logger.debug(f"Fitting source estimator {self.solver.__name__}")
            source_estimator.fit(L, y_noisy)
            solver_output = source_estimator.predict(y=y_noisy)

            x_hat = solver_output.get("posterior_mean_reshaped")
            if x_hat is None:
                x_hat = solver_output.get("posterior_mean")
            x_hat_active_indices = solver_output.get("active_indices")
            posterior_cov = solver_output.get("posterior_cov")
            noise_var = solver_output.get("noise_var")
            gamma = solver_output.get("gamma")
            
            # TODO: remove temporary plotting code
            if noise_type == 'adaptive_joint_learning':
                plot_error_curves(
                    err_gamma=solver_output["err_gamma_hist"],
                    # err_lambda=solver_output["err_lambda_hist"],
                    title="Gamma errors (joint learning)",
                    save_path=os.path.join(experiment_dir, "gamma_lambda_errors.png"),
                )               

            this_result['gamma'] = gamma
            this_result["noise_var"] = noise_var

            # posterior_var = self.uncertainty_estimator.get_posterior_variance(
            #     posterior_cov=posterior_cov,
            #     orientation_type=orientation_type
            # )

            # x_avg_time = np.mean(x, axis=-1, keepdims=True)
            # x_hat_avg_time = np.mean(x_hat, axis=-1, keepdims=True)
            # n_times = x.shape[-1]
            # posterior_var_avg_time = posterior_var / n_times
            # posterior_std_avg_time = np.sqrt(np.maximum(posterior_var_avg_time, 0.0))
            
            # TODO: this is a temporary workaround to allow calibration of free orientation solvers using only the norm of the source estimates and their uncertainty.
            # for free orientation, reshape posterior covariance from (3N, 3N) to (N, N, 3, 3)
            # if orientation_type == "free":
            #     x_avg_time=np.linalg.norm(x_avg_time, axis=1, keepdims=False)
            #     x_hat_avg_time=np.linalg.norm(x_hat_avg_time, axis=1, keepdims=False)
            #     x = np.linalg.norm(x, axis=1, keepdims=False)
            #     x_hat = np.linalg.norm(x_hat, axis=1, keepdims=False)
            #     x_hat_active_indices = x_hat_active_indices[:x_hat_avg_time.shape[0]]

            active_indices_size = (
                len(x_hat_active_indices)
                if x_hat_active_indices is not None
                else 0
            )
            this_result['active_indices_size'] = active_indices_size

            summary_path = None
            if self.save_posterior_stats:
                posterior_base = self.posterior_dir if self.posterior_dir is not None else Path(experiment_dir)
                summary_filename = f"posterior_summary_run{global_run_id:08d}_seed{seed}.h5"

                summary_arrays = {
                    "x_true": x,
                    "x_hat": x_hat,
                    "posterior_cov": posterior_cov,
                    "Q_basis": np.asarray(prepared_data.get("Q_basis"), dtype=float),
                }

                summary_metadata = {
                    "global_run_id": global_run_id,
                    "run_id": run_in_config,
                    "nruns": nruns,
                    "solver": solver_name,
                    "noise_type": noise_params.get("noise_type"),
                    "subject": data_params.get("subject"),
                    "orientation_type": orientation_type,
                    "nnz": data_params.get("nnz"),
                    "alpha_SNR": data_params.get("alpha_SNR"),
                    "n_sources": n_sources,
                    "n_times": n_times,
                    "seed": seed,
                    "experiment_dir": Path(experiment_dir).as_posix(),
                    "posterior_dir": Path(posterior_base).as_posix(),
                    "posterior_filename": summary_filename,
                    "sensor_kind": sensor_kind,
                    "coil_type": sensor_coil_type,
                    "sensor_units": sensor_units,
                    "sensor_unitmult": sensor_unitmult,
                }
                summary_path = self._persist_posterior_summary(
                    experiment_dir=experiment_dir,
                    datasets=summary_arrays,
                    metadata=summary_metadata,
                    posterior_dir=posterior_base,
                    filename=summary_filename,
                )
                if summary_path is not None:
                    this_result["posterior_summary"] = summary_path.as_posix()

            return this_result

            # calibrator = UncertaintyCalibrator(
            #     uncertainty_estimator=self.uncertainty_estimator,
            #     metric_evaluator=self.metric_evaluator,
            # )
            # calibration_results = calibrator.calibrate(
            #     x_true=x_avg_time,
            #     x_hat=x_hat_avg_time,
            #     posterior_std=posterior_std_avg_time,
            # )
            # pre_calibration = calibration_results['pre_calibration']
            # post_calibration = calibration_results['post_calibration']
            # calibration_record_dir = Path("results") / "calibration_records"
            # record_path = self._persist_calibration_results(
            #     experiment_dir=experiment_dir,
            #     record_dir=calibration_record_dir,
            #     solver_name=solver_name,
            #     solver_params=solver_params,
            #     data_params=data_params,
            #     noise_params=noise_params,
            #     seed=seed,
            #     global_run_id=global_run_id,
            #     config_index=config_index,
            #     run_in_config=run_in_config,
            #     nruns_local=nruns_local,
            #     pre_calibration=pre_calibration,
            #     post_calibration=post_calibration,
            # )
            # if record_path is not None:
            #     this_result["calibration_record"] = record_path.as_posix()

            # metric_kwargs = dict(
            #     x=x_avg_time,
            #     x_hat=x_hat_avg_time,
            #     posterior_var=posterior_var_avg_time,
            #     orientation_type="fixed", # TODO: remove hardcoding
            #     nnz=data_params.get("nnz"),
            #     subject=data_params.get("subject"),
            #     fwd_path=solver_params['fwd_path'],
            # )

            # try:
            #     evaluation_metrics = self.metric_evaluator.evaluate_metrics(
            #         which="evaluation",
            #         empirical_coverages=pre_calibration['empirical_coverages'],
            #         **metric_kwargs,
            #     )
            #     this_result.update(evaluation_metrics)
            # except Exception as e:
            #     self.logger.error(f"Error while evaluating evaluation metrics: {e}", exc_info=True)
            #     this_result.update({"metric_evaluation_error": str(e)})

            # calibration_metric_names = tuple(
            #     getattr(self.metric_evaluator, "calibration_metrics", tuple())
            # )
            # pre_cal_metrics = pre_calibration.get('calibration_metrics', {})
            # post_cal_metrics = post_calibration.get('calibration_metrics', {})
            # for metric_name in calibration_metric_names:
            #     pre_value = pre_cal_metrics.get(metric_name)
            #     post_value = post_cal_metrics.get(metric_name)
            #     if pre_value is not None:
            #         this_result[f"pre_cal_{metric_name}"] = pre_value
            #     if post_value is not None:
            #         this_result[f"post_cal_{metric_name}"] = post_value
            #     improvement_key = f"improvement_{metric_name}"
            #     if (
            #         pre_value is None
            #         or post_value is None
            #         or (isinstance(pre_value, (int, float, np.floating)) and np.isclose(pre_value, 0.0))
            #     ):
            #         this_result[improvement_key] = None
            #     else:
            #         this_result[improvement_key] = (pre_value - post_value) / pre_value * 100

            # viz = Visualizer(base_save_path=experiment_dir, logger=self.logger)
            # viz.plot_all(
            #     x=x,
            #     x_active_indices=x_active_indices,
            #     x_hat=x_hat,
            #     x_hat_active_indices=x_hat_active_indices,
            #     y_clean=y_clean,
            #     y_noisy=y_noisy,
            #     n_sources=n_sources,
            #     subject=data_params.get("subject"),
            #     fwd_path=solver_params['fwd_path'],
            #     nnz=data_params.get("nnz"),
            #     ERP_config=self.ERP_config,
            #     sample_idx=200,
            #     source_units=source_units,
            #     source_unitmult=source_unitmult,
            #     sensor_units=sensor_units,
            #     sensor_unitmult=sensor_unitmult,
            #     confidence_levels=self.uncertainty_estimator.nominal_coverages,
            #     nominal_coverages=pre_calibration['nominal_coverages'],
            #     empirical_coverages=pre_calibration['empirical_coverages'],
            #     empirical_coverages_post_cal=post_calibration['empirical_coverages'],
            #     ci_lower=pre_calibration.get('ci_lowers'),
            #     ci_upper=pre_calibration.get('ci_uppers'),
            #     orientation_type="fixed", # TODO: remove
            #     result=this_result,
            #     experiment_dir=experiment_dir,
            # )

        except Exception as e:
            self.logger.error(
                f"Error during data generation global_run_id {this_result.get('global_run_id', 'N/A')} (config run {this_result.get('run_id', 'N/A')}): {e}",
                exc_info=True,
            )
            this_result["error_message"] = str(e)

        self.logger.debug(
            f"Completed global_run_id {this_result.get('global_run_id', 'N/A')} (config run {this_result.get('run_id', 'N/A')})"
        )
        return this_result

    def run(
        self,
        nruns: int = 2,
        fig_path: str = "results/figures",
        n_jobs: int = 1,
        run_offset: int = 0,
        global_total_runs: Optional[int] = None,
    ):
        """
        Run data generation by iterating over combinations of solver and data parameters.

        Parameters
        ----------
        nruns : int
            Number of seeds to evaluate for each parameter combination.
        fig_path : str
            Base directory where per-run visualizations will be saved.
        n_jobs : int
            Number of parallel workers to use. ``1`` (default) keeps the sequential behaviour.
        run_offset : int
            Number of experiments completed prior to this generator call. Used
            for global progress tracking when multiple estimators are run
            sequentially.
        global_total_runs : int, optional
            Total number of experiments planned across estimators. If provided,
            the logger also reports this aggregate figure.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the results for each parameter combination.
        """
        rng = check_random_state(self.random_state)
        
        # First, create all parameter combinations (without seeds)
        param_grids = list(product(
            ParameterGrid(self.solver_param_grid),
            ParameterGrid(self.data_param_grid),
            ParameterGrid(self.noise_param_grid),
        ))
        
        # Calculate total number of runs
        num_configs = len(param_grids)
        total_runs = num_configs * nruns
        
        # Generate unique seeds for EVERY experiment (not just nruns seeds)
        # This ensures each parameter combination gets different random data
        all_seeds = rng.randint(low=0, high=2 ** 32, size=total_runs)
        
        # Create parameter combinations with unique seeds
        param_combinations = []
        seed_idx = 0
        for solver_params, data_params, noise_params in param_grids:
            for _ in range(nruns):
                param_combinations.append((
                    solver_params, 
                    data_params, 
                    noise_params, 
                    all_seeds[seed_idx]
                ))
                seed_idx += 1
        
        self.logger.info(
            "%s\nStarting data generation for estimator %s with %d experiments (%d nruns x %d configurations)",
            "-" * 50,
            getattr(self.solver, "__name__", str(self.solver)),
            total_runs,
            nruns,
            num_configs,
        )
        if total_runs == 0:
            return pd.DataFrame()

        worker_args = [
            (
                run_id,
                nruns,
                total_runs,
                solver_params,
                data_params,
                noise_params,
                seed,
                fig_path,
                run_offset + run_id,
                global_total_runs,
            )
            for run_id, (solver_params, data_params, noise_params, seed) in enumerate(param_combinations, start=1)
        ]

        if n_jobs == 1:
            # sequential execution
            results_list = [self._execute_single_run(*args) for args in worker_args]
        else:
            # parallel execution
            self.logger.debug(f"Running data generation in parallel with n_jobs={n_jobs}")
            parallel = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)
            results_list = parallel(
                delayed(self._execute_single_run)(*args)
                for args in worker_args
            )

        self.logger.debug("Data generation completed.")
        return pd.DataFrame(results_list)


# TODO: move plotting functions to Visualizer class
def plot_error_curves(err_gamma, title="Gamma/Lambda errors", save_path=None):
    """
    Plot err_gamma and err_lambda vs iteration.

    Parameters
    ----------
    err_gamma : sequence of float
        Relative gamma errors per iteration.
    err_lambda : sequence of float
        Relative lambda errors per iteration.
    title : str
        Plot title.
    """
    iters = np.arange(len(err_gamma))
    plt.figure()
    plt.semilogy(iters, err_gamma, label="err_gamma")
    # plt.semilogy(iters, err_lambda, label="err_lambda")
    plt.xlabel("Iteration")
    plt.ylabel("Relative error")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

# TODO: remove temporary plotting code
def plot_error_curves_comparison(err_gamma_1, err_lambda_1,
                                err_gamma_2, err_lambda_2,
                                labels=("Method 1", "Method 2")):
    """
    Compare error curves of two methods (e.g. with sFLEX and without sFLEX).

    Parameters
    ----------
    err_gamma_1, err_lambda_1 : sequence of float
        Errors for method 1.
    err_gamma_2, err_lambda_2 : sequence of float
        Errors for method 2.
    labels : tuple of str
        Labels for the two methods.
    """
    iters1 = np.arange(len(err_gamma_1))
    iters2 = np.arange(len(err_gamma_2))

    plt.figure()
    plt.semilogy(iters1, err_gamma_1, label=f"{labels[0]}: err_gamma")
    plt.semilogy(iters1, err_lambda_1, label=f"{labels[0]}: err_lambda")
    plt.semilogy(iters2, err_gamma_2, label=f"{labels[1]}: err_gamma", linestyle="--")
    plt.semilogy(iters2, err_lambda_2, label=f"{labels[1]}: err_lambda", linestyle="--")
    plt.xlabel("Iteration")
    plt.ylabel("Relative error")
    plt.title("Gamma/Lambda Error Comparison")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
# TODO: remove temporary plotting code    
def plot_alphas_cv(alphas, grid_factors, baseline_noise_var, experiment_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid_factors, alphas, marker='o', linestyle='-', label=f'alphas (n={len(alphas)})')
    ax.axhline(
        baseline_noise_var,
        color='red',
        linestyle='--',
        label=f'baseline_noise_var = {baseline_noise_var:.3e}',
    )
    ax.set_xscale('log')
    ax.set_xlabel('grid factor (log scale)')
    ax.set_ylabel('alpha (noise variance)')
    ax.set_title('Spatial CV: Alpha grid vs baseline noise variance')
    ax.legend()
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    save_path = os.path.join(experiment_dir, "alphas_vs_baseline.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
