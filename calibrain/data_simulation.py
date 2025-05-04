import numpy as np
from numpy.random import Generator
from scipy.stats import wishart
from pathlib import Path
import logging
import os
from typing import Optional, Tuple, Union, Dict, List, Any

import mne
import matplotlib.pyplot as plt

from calibrain import LeadfieldSimulator
from calibrain.utils import load_config

"""
Module for simulating brain activity data.

Provides the `DataSimulator` class to generate synthetic brain activity,
including source time courses, leadfield matrices (via loading, simulation,
or random generation), and sensor-level measurements with controllable noise (SNR).
Supports both fixed and free source orientations and includes visualization tools.
"""

class DataSimulator:
    """
    Simulates brain activity data including source activity, leadfield, and sensor measurements.

    Handles different leadfield generation modes (random, load, simulate) and
    noise addition based on specified SNR. Supports fixed and free source orientations.

    Attributes
    ----------
    n_sensors : int
        Number of sensors. Updated based on the obtained leadfield.
    n_sources : int
        Number of sources. Updated based on the obtained leadfield.
    n_times : int
        Number of time points.
    nnz : int
        Number of non-zero (active) sources to simulate.
    orientation_type : str
        Orientation type ('fixed' or 'free').
    alpha_snr : float
        Target SNR defined as signal_norm / (signal_norm + noise_norm).
    noise_type : str
        Type of noise generation method (currently affects oracle covariance).
    seed : Optional[int]
        Random seed for reproducibility.
    logger : logging.Logger
        Logger instance.
    rng : np.random.Generator
        Random number generator instance.
    leadfield_mode : str
        Mode for leadfield generation ('load', 'simulate', 'random').
    leadfield_path : Optional[Path]
        Path to the leadfield file (used if mode='load').
    leadfield_config_path : Optional[Path]
        Path to config file (used if mode='simulate').
    """
    def __init__(
        self,
        n_sensors: int = 60,
        n_sources: int = 3000,
        n_times: int = 100,
        nnz: int = 3,
        orientation_type: str = "fixed",
        alpha_snr: float = 0.99,
        noise_type: str = "oracle",
        seed: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
        rng: Optional[Generator] = None,
        leadfield_mode: str = "random",
        leadfield_path: Optional[Union[str, Path]] = None,
        leadfield_config_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the DataSimulator.

        Parameters
        ----------
        n_sensors : int, optional
            Initial number of sensors, by default 60. May be updated by leadfield.
        n_sources : int, optional
            Initial number of sources, by default 3000. May be updated by leadfield.
        n_times : int, optional
            Number of time points, by default 100.
        nnz : int, optional
            Number of non-zero sources, by default 3.
        orientation_type : str, optional
            Orientation type ('fixed' or 'free'), by default "fixed".
        alpha_snr : float, optional
            Target SNR = signal_norm / (signal_norm + noise_norm), by default 0.99.
        noise_type : str, optional
            Type of noise ('random' or 'scaled_identity'), by default "oracle".
        seed : Optional[int], optional
            Random seed for reproducibility. If None, uses default seeding, by default None.
        logger : Optional[logging.Logger], optional
            Logger instance. If None, creates a default logger, by default None.
        rng : Optional[Generator], optional
            NumPy random number generator. If None, creates one based on seed, by default None.
        leadfield_mode : str, optional
            'load', 'simulate', or 'random', by default "random".
        leadfield_path : Optional[Union[str, Path]], optional
            Path to leadfield file (if mode='load'), by default None.
        leadfield_config_path : Optional[Union[str, Path]], optional
            Path to config file (if mode='simulate'), by default None.
        """
        self.n_sensors = n_sensors
        self.n_sources = n_sources
        self.n_times = n_times
        self.nnz = nnz
        self.orientation_type = orientation_type
        self.alpha_snr = alpha_snr
        self.noise_type = noise_type # Currently unused beyond init logging
        self.seed = seed
        self.logger = logger if logger else logging.getLogger(__name__)

        self.rng = rng if rng else np.random.default_rng(seed)

        self.leadfield_mode = leadfield_mode
        self.leadfield_path = Path(leadfield_path) if leadfield_path else None
        self.leadfield_config_path = Path(leadfield_config_path) if leadfield_config_path else None

        self.logger.info(f"DataSimulator initialized with orientation: {self.orientation_type}, leadfield mode: {self.leadfield_mode}")

    def _get_leadfield(self) -> np.ndarray:
        """
        Get or generate the leadfield matrix based on the specified mode.

        Updates self.n_sensors and self.n_sources based on the obtained leadfield.

        Returns
        -------
        np.ndarray
            The leadfield matrix (L). Shape depends on orientation_type:
            - 'fixed': (n_sensors, n_sources)
            - 'free': (n_sensors, n_sources, 3)

        Raises
        ------
        ValueError
            If leadfield_mode is invalid, required paths are missing,
            or loaded/simulated leadfield has unexpected dimensions/format.
        FileNotFoundError
            If leadfield_path does not exist when mode='load'.
        """
        expected_suffix = "-free.npz" if self.orientation_type == "free" else "-fixed.npz"
        expected_dimensions = 3 if self.orientation_type == "free" else 2

        leadfield: np.ndarray

        if self.leadfield_mode == "load":
            if not self.leadfield_path:
                raise ValueError("Path to the leadfield file (leadfield_path) must be provided when leadfield_mode='load'.")

            try:
                if not self.leadfield_path.exists():
                    raise FileNotFoundError(f"Leadfield file does not exist: {self.leadfield_path}")

                # Optional strict check:
                # if not self.leadfield_path.name.endswith(expected_suffix):
                #     self.logger.warning(f"Leadfield file name '{self.leadfield_path.name}' does not match expected suffix '{expected_suffix}' for orientation '{self.orientation_type}'.")

                self.logger.info(f"Loading leadfield matrix from file: {self.leadfield_path}")
                with np.load(self.leadfield_path) as data:
                    if "leadfield" not in data:
                        raise ValueError(f"File {self.leadfield_path} does not contain 'leadfield' key.")
                    leadfield = data["leadfield"]

                if leadfield.ndim != expected_dimensions:
                    raise ValueError(
                        f"Loaded leadfield matrix dimension mismatch for orientation '{self.orientation_type}': "
                        f"expected {expected_dimensions} dimensions, but got {leadfield.ndim}."
                    )
                self.logger.info(f"Leadfield loaded with shape {leadfield.shape}")

            except (FileNotFoundError, ValueError) as e:
                self.logger.error(f"Failed to load leadfield matrix: {e}")
                raise

        elif self.leadfield_mode == "simulate":
            if not self.leadfield_config_path:
                raise ValueError("Path to the configuration file (leadfield_config_path) must be provided when leadfield_mode='simulate'.")
            self.logger.info(f"Simulating leadfield matrix using LeadfieldSimulator with config: {self.leadfield_config_path}")

            try:
                config = load_config(Path(self.leadfield_config_path))
                L_simulator = LeadfieldSimulator(config=config, logger=self.logger)
                leadfield = L_simulator.simulate()
                self.logger.info(f"Simulated leadfield matrix with shape {leadfield.shape}")

                if leadfield.ndim != expected_dimensions:
                    raise ValueError(
                        f"Simulated leadfield matrix dimension mismatch for orientation '{self.orientation_type}': "
                        f"expected {expected_dimensions} dimensions, but got {leadfield.ndim}."
                    )
            except Exception as e:
                 self.logger.error(f"Failed to simulate leadfield matrix: {e}")
                 raise

        elif self.leadfield_mode == "random":
            self.logger.info(f"Generating a random leadfield matrix (n_sensors={self.n_sensors}, n_sources={self.n_sources}).")
            if self.orientation_type == "fixed":
                leadfield = self.rng.standard_normal((self.n_sensors, self.n_sources))
            else:
                leadfield = self.rng.standard_normal((self.n_sensors, self.n_sources, 3))
            self.logger.info(f"Random leadfield generated with shape {leadfield.shape}")

        else:
            raise ValueError(f"Invalid leadfield mode '{self.leadfield_mode}'. Options are 'load', 'simulate', or 'random'.")

        # Update n_sensors and n_sources based on the actual leadfield dimensions
        if leadfield.ndim == 2: # Fixed
            self.n_sensors, self.n_sources = leadfield.shape
        elif leadfield.ndim == 3: # Free
            self.n_sensors, self.n_sources, _ = leadfield.shape

        self.logger.info(f"Leadfield obtained. Updated n_sensors={self.n_sensors}, n_sources={self.n_sources}")
        return leadfield

    def _generate_source_time_courses(self) -> np.ndarray:
        """
        Generate synthetic source time courses in the source space.

        Only `self.nnz` sources will have non-zero activity, randomly chosen.

        Returns
        -------
        np.ndarray
            Source time courses (x). Shape depends on orientation_type:
            - 'fixed': (n_sources, n_times)
            - 'free': (n_sources, 3, n_times)

        Raises
        ------
        ValueError
            If `self.orientation_type` is unsupported.
        """
        if self.orientation_type == "fixed":
            x = np.zeros((self.n_sources, self.n_times))
            idx = self.rng.choice(self.n_sources, size=self.nnz, replace=False)
            x[idx] = self.rng.standard_normal((self.nnz, self.n_times))
        elif self.orientation_type == "free":
            n_orient = 3
            x = np.zeros((self.n_sources, n_orient, self.n_times))
            idx = self.rng.choice(self.n_sources, size=self.nnz, replace=False)
            x[idx] = self.rng.standard_normal((self.nnz, n_orient, self.n_times))
        else:
            raise ValueError(f"Unsupported orientation type: {self.orientation_type}")

        return x

    def _project_to_sensor_space(self, L: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Project the source activity to the sensor space using the leadfield matrix.

        Parameters
        ----------
        L : np.ndarray
            Leadfield matrix (µV / nAm).
            - 'fixed': Shape (n_sensors, n_sources).
            - 'free': Shape (n_sensors, n_sources, 3).
        x : np.ndarray (nAm)
            Source activity.
            - 'fixed': Shape (n_sources, n_times).
            - 'free': Shape (n_sources, 3, n_times).

        Returns
        -------
        np.ndarray
            Sensor measurements (y_clean). Shape: (n_sensors, n_times). => (µV / nAm) * nAm = µV

        Raises
        ------
        ValueError
            If `self.orientation_type` is unsupported.
        """
        # (µV / nAm) * nAm = µV
        if self.orientation_type == "fixed":
            # Matrix multiplication: (n_sensors, n_sources) @ (n_sources, n_times) -> (n_sensors, n_times)
            y = L @ x
        elif self.orientation_type == "free":
            # Einstein summation: Sum over source index 'm' and orientation index 'r'
            # (n_sensors, n_sources, 3) einsum (n_sources, 3, n_times) -> (n_sensors, n_times)
            y = np.einsum("nmr,mrt->nt", L, x) # Corrected einsum indices
        else:
            raise ValueError(f"Unsupported orientation type: {self.orientation_type}")

        return y

    def _add_noise(self, y_clean: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Add scaled Gaussian noise to clean sensor measurements based on alpha_snr.

        Parameters
        ----------
        y_clean : np.ndarray
            Clean sensor measurements, shape (n_sensors, n_times).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            - y_noisy : Noisy sensor measurements, shape (n_sensors, n_times).
            - cov_scaled : Scaled noise covariance matrix, shape (n_sensors, n_sensors).
            - noise_scaled : Scaled noise added, shape (n_sensors, n_times).
        """
        # Define a oracle noise covariance (identity matrix scaled)
        oracle_cov = 1e-2 * np.eye(self.n_sensors)
        # Generate multivariate normal noise based on the oracle covariance
        noise = self.rng.multivariate_normal(np.zeros(self.n_sensors), oracle_cov, size=self.n_times).T

        # Frobenius norm for signal and noise
        signal_norm = np.linalg.norm(y_clean, "fro")
        noise_norm = np.linalg.norm(noise, "fro")

        # Prevent division by zero if norms are zero
        if noise_norm == 0:
             self.logger.warning("Initial noise norm is zero. Cannot scale noise based on SNR. Returning clean signal.")
             return y_clean, oracle_cov, noise # Return unscaled noise and cov

        if signal_norm == 0:
             self.logger.warning("Clean signal norm is zero. Noise scaling might be arbitrary.")
             # Decide on behavior: maybe return zero noise or unscaled noise?
             # Returning unscaled noise for now.
             snr_scaling_factor = 1.0
        else:
             # Calculate SNR scaling factor to achieve target alpha_snr
             snr_scaling_factor = ((1 - self.alpha_snr) / self.alpha_snr) * (signal_norm / noise_norm)

        noise_scaled = noise * snr_scaling_factor
        cov_scaled = oracle_cov * snr_scaling_factor ** 2
        y_noisy = y_clean + noise_scaled

        if (signal_norm + np.linalg.norm(noise_scaled, "fro")) > 0:
            actual_alpha = signal_norm / (signal_norm + np.linalg.norm(noise_scaled, "fro")) 
        else:
            actual_alpha = 0
        self.logger.info(f"Target alpha_snr: {self.alpha_snr:.4f}, Actual alpha_snr: {actual_alpha:.4f}")


        return y_noisy, cov_scaled, noise_scaled

    def simulate(self, visualize: bool = True, save_path: str = "results/figures/data_sim/") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the full data simulation pipeline.

        Steps:
        1. Get leadfield matrix (load, simulate, or random).
        2. Generate source time courses.
        3. Project sources to sensor space (clean measurements).
        4. Add noise based on SNR.
        5. Optionally visualize results.

        Parameters
        ----------
        visualize : bool, optional
            Whether to generate and save visualization plots, by default True.
            Requires `leadfield_mode` to be 'simulate' or a valid `leadfield_config_path`
            to be provided for obtaining MNE info.
        save_path : str, optional
            Base directory to save visualization figures,
            by default "results/figures/data_sim/".

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            - y_noisy : Noisy sensor measurements (n_sensors, n_times).
            - L : Leadfield matrix. Shape depends on orientation type, potentially reshaped for 'free'.
            - x : Ground truth source activity. Shape depends on orientation type.
            - cov_scaled : Scaled noise covariance matrix (n_sensors, n_sensors).
            - noise_scaled : Scaled noise added (n_sensors, n_times).

        Raises
        ------
        ValueError
            If visualization is requested but MNE info cannot be obtained
            (e.g., `leadfield_mode` is not 'simulate' and `leadfield_config_path` is missing).
        """
        L_orig = self._get_leadfield()
        x = self._generate_source_time_courses()
        y_clean = self._project_to_sensor_space(L_orig, x)
        y_noisy, cov_scaled, noise_scaled = self._add_noise(y_clean)

        if visualize:
            info = None
            try:
                # Attempt to get info using LeadfieldSimulator setup
                config = load_config(Path(self.leadfield_config_path))
                info = LeadfieldSimulator(config=config, logger=self.logger).handle_info()
            except Exception as e:
                self.logger.warning(f"Could not load info from config {self.leadfield_config_path} for visualization: {e}")

            save_path_obj = Path(save_path)
            save_path_obj.mkdir(parents=True, exist_ok=True)

            self.visualize_signals(
                x=x,
                y_clean=y_clean,
                y_noisy=y_noisy,
                nnz_to_plot=self.nnz,
                sfreq=info["sfreq"] if info else 100.0, # Default sfreq if info missing
                max_sensors=3,
                plot_sensors_together=False,
                show=False,
                save_path=os.path.join(save_path, "data_sim.png"),
            )

            self.visualize_leadfield(
                 L_orig,
                 orientation_type=self.orientation_type,
                 save_path=os.path.join(save_path, "leadfield_matrix.png"),
                 show=False
            )

            if info:
                self.visualize_leadfield_topomap(
                    leadfield_matrix=L_orig,
                    info=info,
                    x=x,
                    orientation_type=self.orientation_type,
                    title="Leadfield Topomap for Active (Nonzero) Sources",
                    save_path=os.path.join(save_path, "leadfield_topomap.png"),
                    show=False,
                )
            else:
                 self.logger.info("Skipping leadfield topomap visualization due to missing MNE info.")

        # Reshape leadfield matrix for free orientation if needed by downstream estimators.
        # This might be a workaround depending on estimator expectations.
        L = L_orig
        if self.orientation_type == "free":
            self.logger.debug("Reshaping free orientation leadfield from (sensors, sources, 3) to (sensors, sources*3)")
            L = L_orig.reshape(L_orig.shape[0], -1)

        return y_noisy, L, x, cov_scaled, noise_scaled

    def visualize_signals(
        self,
        x: np.ndarray,
        y_clean: np.ndarray,
        y_noisy: np.ndarray,
        active_sources: Optional[np.ndarray] = None,
        nnz_to_plot: int = -1,
        sfreq: float = 100.0,
        max_sensors: int = 3,
        plot_sensors_together: bool = False,
        shift: float = 20.0,
        figsize: Tuple[float, float] = (14, 10),
        save_path: Optional[str] = 'results/figures/data_sim.png',
        show: bool = False
    ) -> None:
        """
        Visualize source activity and sensor measurements.

        Plots active source time courses and compares clean vs. noisy sensor signals.

        Parameters
        ----------
        x : np.ndarray
            Source activity. Shape depends on orientation type.
        y_clean : np.ndarray
            Clean sensor measurements (n_sensors, n_times).
        y_noisy : np.ndarray
            Noisy sensor measurements (n_sensors, n_times).
        active_sources : Optional[np.ndarray], optional
            Indices of non-zero (active) sources. If None, they are determined from x, by default None.
        nnz_to_plot : int, optional
            Number of non-zero sources to plot. If -1, plot all non-zero sources found, by default -1.
        sfreq : float, optional
            Sampling frequency in Hz, by default 100.0.
        max_sensors : int, optional
            Maximum number of sensors to plot, by default 3.
        plot_sensors_together : bool, optional
            If True, plot all sensors on the same subplot. If False, stack plots vertically, by default False.
        shift : float, optional
            Vertical shift between sensors when plotting together, by default 20.0.
        figsize : Tuple[float, float], optional
            Figure size for the plot, by default (14, 10).
        save_path : Optional[str], optional
            Path to save the figure. If None, the figure is not saved, by default 'results/figures/data_sim.png'.
        show : bool, optional
            If True, display the plot, by default False.
        """
        n_times = y_clean.shape[1]
        times = np.linspace(0, (n_times - 1) / sfreq, n_times) if n_times > 1 else np.array([0])

        if active_sources is None:
            if self.orientation_type == "fixed":
                active_sources = np.where(np.any(x != 0, axis=-1))[0]
            elif self.orientation_type == "free":
                self.logger.info("Calculating norm of source activity to find active sources for free orientation.")
                # Check if any component (X, Y, Z) at any time point is non-zero for a source
                active_sources = np.where(np.any(x != 0, axis=(1, 2)))[0]
            else:
                 raise ValueError(f"Unsupported orientation type: {self.orientation_type}")


        if nnz_to_plot != -1 and len(active_sources) > nnz_to_plot:
             plot_indices = self.rng.choice(active_sources, nnz_to_plot, replace=False)
             self.logger.info(f"Plotting {nnz_to_plot} randomly selected active sources out of {len(active_sources)}.")
        else:
             plot_indices = active_sources
             nnz_to_plot = len(plot_indices) # Update actual number plotted

        y_min = min(y_clean.min(), y_noisy.min())
        y_max = max(y_clean.max(), y_noisy.max())
        y_range = y_max - y_min if y_max > y_min else 1.0 # Avoid zero range

        num_sensors_to_plot = min(max_sensors, y_clean.shape[0])
        total_plots = 1 + (1 if plot_sensors_together else num_sensors_to_plot)
        fig, axes = plt.subplots(
            total_plots,
            1,
            figsize=figsize,
            gridspec_kw={"height_ratios": [1] * total_plots}, # Equal height for now
            sharex=True # Share x-axis
        )
        # Ensure axes is always an array
        if total_plots == 1:
            axes = [axes]

        ax_sources = axes[0]
        if self.orientation_type == "fixed":
            for i in plot_indices:
                ax_sources.plot(times, x[i].T, label=f"Source {i}")
        elif self.orientation_type == "free":
            for i in plot_indices:
                # Plot norm or individual components? Plotting norm for simplicity.
                source_norm = np.linalg.norm(x[i], axis=0)
                ax_sources.plot(times, source_norm, label=f"Source {i} (Norm)")
                # Alternatively, plot components:
                # for j, orient in enumerate(["X", "Y", "Z"]):
                #     ax_sources.plot(times, x[i, j], label=f"Source {i} ({orient})", alpha=0.7)

        ax_sources.set_title(f"{nnz_to_plot} Active Simulated Source Activity")
        ax_sources.set_ylabel("Amplitude (a.u.)") # Arbitrary units for sources
        ax_sources.grid(True)
        ax_sources.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        sensor_axes = axes[1:]
        if plot_sensors_together:
            ax_sensors = sensor_axes[0]
            current_shift = 0
            for i in range(num_sensors_to_plot):
                ax_sensors.plot(times, y_clean[i] + current_shift, label=f"Clean (Sensor {i})", linewidth=1.5)
                ax_sensors.plot(times, y_noisy[i] + current_shift, label=f"Noisy (Sensor {i})", alpha=0.8, linewidth=1)
                current_shift += shift # Use provided shift relative to previous signal
            ax_sensors.set_title("Sensor Measurements")
            ax_sensors.set_ylabel("Amplitude (a.u.)") # Arbitrary units for sensors
            ax_sensors.grid(True)
            ax_sensors.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            for idx, ax_sens in enumerate(sensor_axes):
                ax_sens.plot(times, y_clean[idx], label=f"Clean", linewidth=1.5)
                ax_sens.plot(times, y_noisy[idx], label=f"Noisy", alpha=0.8, linewidth=1)
                ax_sens.set_title(f"Sensor {idx}")
                ax_sens.set_ylabel("Amplitude (a.u.)")
                ax_sens.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range) # Consistent ylim
                ax_sens.grid(True)
                ax_sens.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        axes[-1].set_xlabel("Time (s)")

        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legends

        if save_path:
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
            self.logger.info(f"Signals visualization saved to {save_path}")

        if show:
            plt.show()
        plt.close(fig)

    def visualize_leadfield(
        self,
        leadfield_matrix: np.ndarray,
        orientation_type: str = "fixed",
        save_path: Optional[str] = None,
        show: bool = False
    ) -> None:
        """
        Visualize the leadfield matrix as a heatmap.

        Parameters
        ----------
        leadfield_matrix : np.ndarray
            The leadfield matrix.
            - 'fixed': Shape (n_sensors, n_sources).
            - 'free': Shape (n_sensors, n_sources, 3).
        orientation_type : str, optional
            Orientation type ('fixed' or 'free'), by default "fixed".
        save_path : Optional[str], optional
            Path to save the figure. If None, not saved, by default None.
        show : bool, optional
            If True, display the plot, by default False.

        Raises
        ------
        ValueError
            If leadfield_matrix is invalid or orientation_type is unsupported.
        """
        if leadfield_matrix is None or not isinstance(leadfield_matrix, np.ndarray) or leadfield_matrix.size == 0:
            self.logger.error("Invalid leadfield matrix provided for visualization.")
            return

        fig = None # Initialize fig
        try:
            if orientation_type == "fixed":
                if leadfield_matrix.ndim != 2:
                     raise ValueError(f"Expected 2D leadfield for fixed orientation, got {leadfield_matrix.ndim}D")
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(leadfield_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
                fig.colorbar(im, ax=ax, label="Amplitude")
                ax.set_title("Leadfield Matrix (Fixed Orientation)")
                ax.set_xlabel("Sources")
                ax.set_ylabel("Sensors")
            elif orientation_type == "free":
                if leadfield_matrix.ndim != 3 or leadfield_matrix.shape[-1] != 3:
                     raise ValueError(f"Expected 3D leadfield (..., 3) for free orientation, got shape {leadfield_matrix.shape}")
                n_orient = leadfield_matrix.shape[-1]
                fig, axes = plt.subplots(1, n_orient, figsize=(15, 5), sharey=True)
                if n_orient == 1: axes = [axes] # Ensure axes is iterable
                orientations = ["X", "Y", "Z"]
                images = []
                for i in range(n_orient):
                    im = axes[i].imshow(leadfield_matrix[:, :, i], aspect='auto', cmap='viridis', interpolation='nearest')
                    images.append(im)
                    axes[i].set_title(f"Leadfield Matrix ({orientations[i]})")
                    axes[i].set_xlabel("Sources")
                axes[0].set_ylabel("Sensors")
                fig.colorbar(images[0], ax=axes, location="right", label="Amplitude", fraction=0.05, pad=0.04)
            else:
                raise ValueError("Invalid orientation type. Must be 'fixed' or 'free'.")

            plt.tight_layout()

            if save_path:
                save_dir = Path(save_path).parent
                save_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, bbox_inches="tight")
                self.logger.info(f"Leadfield matrix visualization saved to {save_path}")
            if show:
                plt.show()

        except Exception as e:
             self.logger.error(f"Failed during leadfield visualization: {e}")
        finally:
             if fig:
                 plt.close(fig)


    def visualize_leadfield_topomap(
        self,
        leadfield_matrix: np.ndarray,
        info: mne.Info,
        x: np.ndarray,
        orientation_type: str = "fixed",
        save_path: Optional[str] = None,
        title: Optional[str] = None,
        show: bool = False
    ) -> None:
        """
        Visualize leadfield patterns as topomaps for active sources.

        Parameters
        ----------
        leadfield_matrix : np.ndarray
            The leadfield matrix.
            - 'fixed': Shape (n_sensors, n_sources).
            - 'free': Shape (n_sensors, n_sources, 3).
        info : mne.Info
            MNE info object containing sensor locations.
        x : np.ndarray
            Source activity matrix to determine active sources.
            - 'fixed': Shape (n_sources, n_times).
            - 'free': Shape (n_sources, 3, n_times).
        orientation_type : str, optional
            Orientation type ('fixed' or 'free'), by default "fixed".
        save_path : Optional[str], optional
            Path to save the figure. If None, not saved, by default None.
        title : Optional[str], optional
            Title for the entire figure, by default None.
        show : bool, optional
            If True, display the plot, by default False.

        Raises
        ------
        ValueError
            If inputs are invalid or orientation_type is unsupported.
        """
        if leadfield_matrix is None or not isinstance(leadfield_matrix, np.ndarray) or leadfield_matrix.size == 0:
            self.logger.error("Invalid leadfield matrix provided for topomap visualization.")
            return
        if x is None or not isinstance(x, np.ndarray) or x.size == 0:
            self.logger.error("Invalid source activity matrix provided for topomap visualization.")
            return
        if info is None or not isinstance(info, mne.Info):
             self.logger.error("Invalid MNE info object provided for topomap visualization.")
             return

        fig = None # Initialize fig
        try:
            if orientation_type == "fixed":
                if leadfield_matrix.ndim != 2:
                     raise ValueError(f"Expected 2D leadfield for fixed orientation, got {leadfield_matrix.ndim}D")
                active_sources = np.where(np.any(x != 0, axis=-1))[0]
            elif orientation_type == "free":
                if leadfield_matrix.ndim != 3 or leadfield_matrix.shape[-1] != 3:
                     raise ValueError(f"Expected 3D leadfield (..., 3) for free orientation, got shape {leadfield_matrix.shape}")
                self.logger.info("Calculating norm of source activity to find active sources for free orientation.")
                active_sources = np.where(np.any(x != 0, axis=(1, 2)))[0]
            else:
                raise ValueError("Invalid orientation type. Must be 'fixed' or 'free'.")

            if len(active_sources) == 0:
                 self.logger.warning("No active sources found to visualize topomaps.")
                 return

            n_active = len(active_sources)
            n_cols = min(5, n_active) # Max 5 columns
            n_rows = int(np.ceil(n_active / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3), squeeze=False)
            axes_flat = axes.flatten()

            # Determine global color limits for consistency
            all_leadfield_values = []
            for i, source_idx in enumerate(active_sources):
                if orientation_type == "fixed":
                    leadfield_values = leadfield_matrix[:, source_idx]
                else: # free
                    # Visualize the norm of the 3 components for simplicity
                    leadfield_values = np.linalg.norm(leadfield_matrix[:, source_idx, :], axis=-1)
                all_leadfield_values.append(leadfield_values)

            if not all_leadfield_values:
                 self.logger.warning("Could not extract leadfield values for any active source.")
                 return

            vmax = np.max(np.abs(all_leadfield_values))
            vmin = -vmax

            for i, source_idx in enumerate(active_sources):
                leadfield_values = all_leadfield_values[i]
                im, _ = mne.viz.plot_topomap(
                    leadfield_values, info, axes=axes_flat[i], cmap="RdBu_r", # Use diverging colormap
                    vlim=(vmin, vmax), show=False, contours=6
                )
                axes_flat[i].set_title(f"Source {source_idx}")

            # Add a single colorbar
            fig.colorbar(im, ax=axes.ravel().tolist(), label='Leadfield Amplitude', shrink=0.6, aspect=10)

            # Hide unused subplots
            for j in range(n_active, len(axes_flat)):
                axes_flat[j].axis("off")

            if title:
                fig.suptitle(title, fontsize=16) # Removed weight="bold"

            plt.tight_layout(rect=[0, 0, 1, 0.95] if title else [0, 0, 1, 1]) # Adjust for suptitle

            if save_path:
                save_dir = Path(save_path).parent
                save_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, bbox_inches="tight")
                self.logger.info(f"Leadfield topomap visualization saved to {save_path}")

            if show:
                plt.show()

        except Exception as e:
             self.logger.error(f"Failed during leadfield topomap visualization: {e}")
        finally:
             if fig:
                 plt.close(fig)