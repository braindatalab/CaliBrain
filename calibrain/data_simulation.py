import os
from pathlib import Path
import logging
from typing import Optional, Tuple, Union, Dict, List, Any

import numpy as np
from numpy.random import Generator
from scipy.stats import wishart
from scipy.signal import butter, filtfilt
import mne

import matplotlib.pyplot as plt
import matplotlib.cm as cm # Import colormap functionality
from matplotlib.lines import Line2D # Import for custom legend
import matplotlib.gridspec as gridspec
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable # For better colorbar placement

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
    """

    def __init__(
        self,
        n_sensors: int = None,
        n_sources: int = None,
        n_times: int = 100,
        tmin: float = -0.5,
        tmax: float = 0.5,
        stim_onset: float = 0.0,
        sfreq: float = 250,
        fmin: int = 1,
        fmax: int = 5,
        nnz: int = 3,
        alpha_snr_db: float = 6.0,
        amplitude: float = 5.0, # nAm
        randomize_erp_timing_within_post_stim: bool = True,
        min_erp_duration_samples : int = None,
        n_trials: int = 3,
        orientation_type: str = "fixed",
        n_orient: int = 1,
        # noise_type: str = "oracle",
        seed: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
        rng: Optional[Generator] = None,
        leadfield_mode: str = "random",
        leadfield_dir: Optional[Union[str, Path]] = None,
        leadfield_config_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the DataSimulator.
        Parameters
        ----------
        n_sensors : int, optional
            Number of sensors, by default None.
        n_sources : int, optional
            Number of sources, by default None.
        n_times : int, optional
            Number of time points, by default 100.
        tmin : float, optional
            Start time in seconds, by default -0.5.
        tmax : float, optional
            End time in seconds, by default 0.5.
        stim_onset : float, optional
            The sample index within n_times at or after which ERP activity should occur. Defaults to 0.
        sfreq : float, optional
            Sampling frequency in Hz, by default 250.
        fmin : int, optional
            Minimum frequency for bandpass filter, by default 1.
        fmax : int, optional
            Maximum frequency for bandpass filter, by default 5.
        nnz : int, optional
            Number of active sources, by default 3.
        alpha_snr_db : float, optional
            Signal-to-noise ratio in dB, by default 6.0.
        amplitude : float, optional
            Amplitude of the ERP, by default 5.0.
        randomize_erp_timing_within_post_stim : bool, optional
            If True, randomizes the ERP's exact start and duration within the post-onset_sample window, by default True.
        min_erp_duration_samples : int, optional
            Minimum number of samples for the ERP activity segment. If None, uses an internal default. This is used as the lower bound for randomized duration.
        n_trials : int, optional
            Number of trials to simulate, by default 3.
        orientation_type : str, optional
            Source orientation type ("fixed" or "free"), by default "fixed".
        n_orient : int, optional
            Number of orientations for free sources (1 for fixed, 3 for free), by default 1.
        # noise_type : str, optional
            # Noise type, by default "oracle".
        seed : Optional[int], optional
            Random seed, by default None.
        logger : Optional[logging.Logger], optional
            Logger instance, by default None.
        rng : Optional[Generator], optional
            Random number generator, by default None.
        leadfield_mode : str, optional
            Leadfield generation mode ("random", "load", "simulate"), by default "random".
        leadfield_dir : Optional[Union[str, Path]], optional
            Path to leadfield files, by default None.
        leadfield_config_path : Optional[Union[str, Path]], optional
            Path to leadfield config file, by default None.
        
        Raises
        ------
        ValueError
            If leadfield_mode is invalid or required paths are missing.    
        """
        self.n_sensors = n_sensors
        self.n_sources = n_sources
        self.nnz = nnz
        self.n_times = n_times
        self.tmin = tmin
        self.tmax = tmax
        self.stim_onset = stim_onset
        self.sfreq = sfreq
        self.fmin = fmin
        self.fmax = fmax
        self.alpha_snr_db = alpha_snr_db
        self.amplitude = amplitude
        self.randomize_erp_timing_within_post_stim = randomize_erp_timing_within_post_stim
        self.min_erp_duration_samples = min_erp_duration_samples
        self.n_trials = n_trials
        self.orientation_type = orientation_type
        self.n_orient = n_orient
        # self.noise_type = noise_type
        self.seed = seed
        self.logger = logger if logger else logging.getLogger(__name__)
        self.rng = rng if rng else np.random.default_rng(seed)
        self.leadfield_mode = leadfield_mode
        self.leadfield_dir = Path(leadfield_dir) if leadfield_dir else None
        self.leadfield_config_path = Path(leadfield_config_path) if leadfield_config_path else None

        self.logger.info(f"DataSimulator initialized with orientation: {self.orientation_type}, leadfield mode: {self.leadfield_mode}")

    def _get_leadfield(self, subject) -> np.ndarray:
        """
        Get or generate the leadfield matrix based on the specified mode.

        Updates self.n_sensors and self.n_sources based on the obtained leadfield.

        Parameters
        ----------
        subject : str
        
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
            If leadfield_dir does not exist when mode='load'.
        """
        if self.orientation_type == "fixed":
            self.n_orient = 1 
            expected_dimensions = 2 # Fixed orientation leads to 2D leadfield (n_sensors, n_sources)
            expected_suffix = "-fixed.npz"
        elif self.orientation_type == "free":
            expected_suffix = "-free.npz"
            self.n_orient = 3
            expected_dimensions = 3 # Free orientation leads to 3D leadfield (n_sensors, n_sources, 3)
        else:
            raise ValueError(f"Invalid orientation_type '{self.orientation_type}'. Choose 'fixed' or 'free'.")
    

        if self.leadfield_mode == "load":
            # Define the two specific patterns you were trying to match:
            # Pattern 1: Includes orientation_type in the filename
            path_option1 = self.leadfield_dir / f"lead_field_{self.orientation_type}_{subject}.npz"
            
            # Pattern 2: Excludes orientation_type from the filename
            path_option2 = self.leadfield_dir / f"lead_field_{subject}.npz"

            try:
                if path_option1.exists():
                    self.leadfield_path = path_option1
                elif path_option2.exists():
                    self.leadfield_path = path_option2
                else:
                    self.logger.warning(
                        f"Leadfield file not found for subject '{subject}' with orientation '{self.orientation_type}' "
                        f"in directory '{self.leadfield_dir}'.\n"
                        f"Checked specific patterns:\n"
                        f"  - {path_option1}\n"
                        f"  - {path_option2}")
                    raise FileNotFoundError(
                        f"Leadfield file not found for subject '{subject}' in directory '{self.leadfield_dir}'. "
                    )

                self.logger.info(f"Loading leadfield matrix from file: {self.leadfield_path}")
                with np.load(self.leadfield_path) as data:
                    if "leadfield" not in data and "lead_field" not in data:
                        raise ValueError(f"File {self.leadfield_path} does not contain 'leadfield' or 'lead_field' key.")
                    leadfield = data["leadfield"] if "leadfield" in data else data["lead_field"]

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
        if leadfield.ndim == 2 == expected_dimensions: # Fixed
            self.n_sensors, self.n_sources = leadfield.shape
        elif leadfield.ndim == 3 == expected_dimensions: # Free
            self.n_sensors, self.n_sources, _ = leadfield.shape

        self.logger.info(f"Leadfield obtained. Updated n_sensors={self.n_sensors}, n_sources={self.n_sources}")
        return leadfield

    def _generate_erp_signal(self, seed, onset_sample):
        """
        Generate a smoothed ERP-like signal using bandpass-filtered noise and a Hanning window.
        
        The actual ERP waveform (noise, filter, Hanning window) is applied to a
        segment of the signal. The signal before `onset_sample` will be zero.
        The Hanning window provides a smooth onset and offset for the activity.

        If `randomize_erp_timing_within_post_stim` is True, the exact start time
        (offset from `onset_sample`) and duration of the ERP waveform within the
        post-`onset_sample` window are randomized. The ERP will still be contained
        entirely within the `onset_sample` to `n_times` interval.

        If `randomize_erp_timing_within_post_stim` is False, the ERP waveform
        spans the entire duration from `onset_sample` to `n_times`.
        
        Parameters:
        ----------
        - seed (int or np.random.RandomState): Random seed or state for reproducibility.
        - onset_sample (int): The sample index at which the ERP activity starts.
        
        Returns:
        -------
        - np.ndarray: The generated ERP signal of length n_times.
        """
        _DEFAULT_MIN_ERP_LEN = 82  # For filter stability (filtfilt butter order 4) & meaningful Hanning window

        if not isinstance(seed, np.random.RandomState):
            rng = np.random.RandomState(seed)
        else:
            rng = seed
        
        output_signal = np.zeros(self.n_times)
        
        current_min_erp_len = self.min_erp_duration_samples if self.min_erp_duration_samples is not None else _DEFAULT_MIN_ERP_LEN

        # Maximum available duration for ERP activity after onset_sample
        max_available_post_stim_duration = self.n_times - onset_sample

        if max_available_post_stim_duration < current_min_erp_len:
            # Not enough samples in the post-stimulus window for a meaningful ERP
            return output_signal

        actual_placement_start_sample: int
        n_times_for_erp_activity: int

        if self.randomize_erp_timing_within_post_stim:
            # Randomize ERP duration: from current_min_erp_len up to max_available_post_stim_duration (inclusive)
            actual_erp_duration = rng.randint(current_min_erp_len, max_available_post_stim_duration + 1)
            
            # Randomize ERP start offset within the available post-stimulus window
            # Max possible start offset (from onset_sample) for the chosen actual_erp_duration
            max_start_offset_from_onset = max_available_post_stim_duration - actual_erp_duration
            start_offset_from_onset = rng.randint(0, max_start_offset_from_onset + 1)
                
            actual_placement_start_sample = onset_sample + start_offset_from_onset
            n_times_for_erp_activity = actual_erp_duration
        else:
            # ERP spans the entire available post-stimulus duration
            n_times_for_erp_activity = max_available_post_stim_duration
            actual_placement_start_sample = onset_sample

        # Safeguard, though preceding logic should ensure this
        if n_times_for_erp_activity < current_min_erp_len:
            return output_signal

        # Generate noise only for the determined duration of the ERP activity
        white_noise_for_erp = rng.randn(n_times_for_erp_activity)
        
        # Design a Butterworth bandpass filter
        low = self.fmin / (self.sfreq / 2)
        high = self.fmax / (self.sfreq / 2)
        
        epsilon = 1e-9
        low = max(epsilon, low)
        high = min(1.0 - epsilon, high)
        if low >= high:
            return output_signal # Invalid frequency band

        try:
            b, a = butter(4, [low, high], btype='band')
        except ValueError as e:
            return output_signal # Filter design failed

        # Filter the noise segment
        erp_segment = filtfilt(b, a, white_noise_for_erp)
        
        # Apply Hanning window over the ERP segment
        erp_segment *= np.hanning(n_times_for_erp_activity) 
        
        std_erp_segment = np.std(erp_segment)
        if std_erp_segment < 1e-9: # Check if standard deviation is effectively zero
            return output_signal # Avoid division by zero; segment is flat
            
        erp_segment /= std_erp_segment # Normalize
        erp_segment *= self.amplitude      # Scale
        
        # Place the generated ERP segment into the output signal at the determined start
        end_sample_for_erp_segment = actual_placement_start_sample + len(erp_segment)
        
        # Ensure placement is within bounds (should be guaranteed by earlier logic)
        if actual_placement_start_sample < self.n_times and end_sample_for_erp_segment <= self.n_times:
            output_signal[actual_placement_start_sample : end_sample_for_erp_segment] = erp_segment
        
        return output_signal

    def _generate_source_time_courses(self, seed):
        """ Generates true source activity time courses. """
        if not isinstance(seed, np.random.RandomState):
            rng = np.random.RandomState(seed)
        else:
            rng = seed
            
        # n_sensors = leadfield.shape[0] # Not directly needed here anymore for y_clean
        times = np.arange(self.tmin, self.tmax, 1.0 / self.sfreq)
        self.n_times = len(times)

        # Determine the sample index for stimulus onset
        stim_indices = np.where(times >= self.stim_onset)[0]
        if len(stim_indices) == 0:
            # Stimulus onset is at or after tmax, effectively no ERP in this epoch
            stim_idx_for_erp_onset = self.n_times
        else:
            stim_idx_for_erp_onset = stim_indices[0]

        # x: np.ndarray # Define x type
        # active_indices: np.ndarray

        if self.orientation_type == "fixed":
            active_indices = rng.choice(self.n_sources, size=self.nnz, replace=False)
            x = np.zeros((self.n_sources, self.n_times))
            for i, src_idx in enumerate(active_indices):
                # Generate ERP signal with specified onset
                # The seed for _generate_erp_signal should be handled carefully if rng is passed
                source_rng_seed = rng.randint(0, 2**32 -1) # Derive a new seed for this source
                erp_waveform = self._generate_erp_signal(
                    seed=source_rng_seed, # Pass a derived seed
                    onset_sample=stim_idx_for_erp_onset
                )
                x[src_idx, :] = erp_waveform # Assign the full waveform (includes leading zeros)
        elif self.orientation_type == "free":
            n_orientations_free = 3 # TODO: Make this configurable
            active_indices = rng.choice(self.n_sources, size=self.nnz, replace=False)
            x = np.zeros((self.n_sources, self.n_orient, self.n_times))
            for i, src_idx in enumerate(active_indices):
                source_rng_seed = rng.randint(0, 2**32 -1)
                erp_waveform = self._generate_erp_signal(
                    seed=source_rng_seed,
                    onset_sample=stim_idx_for_erp_onset
                )
                orient_coeffs = rng.randn(n_orientations_free)
                norm_orient = np.linalg.norm(orient_coeffs)
                if norm_orient < 1e-9: # Avoid division by zero
                    orient_coeffs = np.array([1.0, 0.0, 0.0]) # Default orientation
                else:
                    orient_coeffs /= norm_orient
                
                for j_orient in range(n_orientations_free):
                    x[src_idx, j_orient, :] = orient_coeffs[j_orient] * erp_waveform

                # Alternatively, if you want to assign the same waveform to all orientations
                # for j_orient in range(n_orientations_free):
                #     x[src_idx, j_orient, :] = erp_waveform
        else:
            raise ValueError("Invalid orientation_type. Choose 'fixed' or 'free'.")

        return x, active_indices

    def _project_sources_to_sensors(self, x: np.ndarray, L: np.ndarray) -> np.ndarray:
        """
        Project the source activity to the sensor space using the leadfield matrix.

        Parameters
        ----------
        x : np.ndarray (nAm)
            Source activity.
            - 'fixed': Shape (n_sources, n_times).
            - 'free': Shape (n_sources, 3, n_times).
        L : np.ndarray
            Leadfield matrix (µV / nAm).
            - 'fixed': Shape (n_sensors, n_sources).
            - 'free': Shape (n_sensors, n_sources, 3).

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

    def _add_noise(self, y_clean, rng=None):
        """
        Adds Homoscedastic  (uniform variance across channels) and uncorrelated (white) Gaussian noise to a clean signal based on a desired SNR level.
        
        Parameters:
        - y_clean (np.ndarray): The clean signal array (e.g., channels x times).
        - alpha_snr_db (float): The desired signal-to-noise ratio in decibels (dB).
        - rng (np.random.RandomState, optional): A random number generator state
        for reproducible noise generation. If None, uses default numpy random state.

        Returns:
        - tuple: A tuple containing:
            - y_noisy (np.ndarray): The signal with added noise.
            - noise (np.ndarray): The generated noise array.
            - noise_power (float): The calculated variance of the added noise.
        """
        if rng is None:
            rng = np.random # Use default numpy random state if none provided

        signal_power = np.mean(y_clean ** 2)
        if signal_power == 0:
            print("Warning: Clean signal power is zero. Cannot add noise based on SNR.")
            noise_power = 0
            noise = np.zeros_like(y_clean)
        else:
            snr_linear = 10 ** (self.alpha_snr_db / 10.0)
            # Homoscedastic case: The standard deviation and thus the variance (noise_power) is the same for all sensors and all time points.
            noise_power = signal_power / snr_linear # Variance of the noise
            noise_std = np.sqrt(noise_power)
            
            # Draw noise from Gaussian distribution (independenet noise at each sensor and at each time point). -> The noise covariance matrix is diagonal. 
            noise = rng.normal(0, noise_std, size=y_clean.shape) # White noise (uncorrelated across sensors and time) with a uniform power across sensors. shape: n_sensors x n_times. 

        y_noisy = y_clean + noise
        return y_noisy, noise, noise_power

    def simulate(self, subject, visualize: bool = True, save_path: str = "results/figures/data_sim/") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        subject : str
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
        L = self._get_leadfield(subject)
        
        # --- Simulate Multiple Trials (Clean Data First) ---
        print(f"Simulating {self.n_trials} trials (clean data)...")
        y_clean_all_trials = [] # Store clean sensor data per trial
        x_all_trials = [] # Store source data per trial
        active_indices_all_trials = [] # Store active indices per trial

        for i_trial in range(self.n_trials):
            print(f"  Simulating clean trial {i_trial + 1}/{self.n_trials}")
            trial_seed = self.seed + i_trial if self.seed is not None else None
            
            x_trial, active_indices = self._generate_source_time_courses(trial_seed)
            y_clean_trial = self._project_sources_to_sensors(x=x_trial, L=L)

            x_all_trials.append(x_trial)
            active_indices_all_trials.append(active_indices)
            y_clean_all_trials.append(y_clean_trial)
            
        print("Clean simulation complete.")

        y_clean_all_trials = np.array(y_clean_all_trials) # (n_trials, n_channels, n_times)
        x_all_trials = np.array(x_all_trials) # (n_trials, n_sources, [n_orient,] n_times)
        active_indices_all_trials = np.array(active_indices_all_trials) # (n_trials, nnz)
        
        time_vector = np.arange(self.tmin, self.tmax, 1.0 / self.sfreq)

        # --- Add Noise Separately ---
        y_noisy_all_trials = [] # To store noisy data
        noise_all_trials = [] # Optional: store noise itself
        noise_power_all_trials = [] # Optional: store noise power per trial

        noise_rng = np.random.RandomState(self.seed + self.n_trials) # TODO: Use a separate seed/state for noise

        for i_trial in range(self.n_trials):
            # Get the clean data for this trial
            y_clean_trial = y_clean_all_trials[i_trial]

            # Add noise using the dedicated function
            # Use a trial-specific RNG state derived from noise_rng for reproducibility per trial
            trial_noise_rng = np.random.RandomState(noise_rng.randint(0, 2**32 - 1))
            y_noisy_trial, noise_trial, noise_power_trial = self._add_noise(
                y_clean_trial,
                rng=trial_noise_rng
            )

            y_noisy_all_trials.append(y_noisy_trial)
            noise_all_trials.append(noise_trial) 
            noise_power_all_trials.append(noise_power_trial)

        # Convert noisy data list to NumPy array
        y_noisy_all_trials = np.array(y_noisy_all_trials) # Shape: (n_trials, n_channels, n_times)
        noise_all_trials = np.array(noise_all_trials) # Shape: (n_trials, n_channels, n_times)
        noise_power_all_trials = np.array(noise_power_all_trials) # Shape: (n_trials,)
                
        print("Noise addition complete.")


        if visualize:
            # Plot data from the first trial
            first_trial_idx = 0
            sensor_subplots_indices = [0, 10, 20] # Indices for the subplot sensor plot

            info = None
            try:
                # Attempt to get info using LeadfieldSimulator setup
                config = load_config(Path(self.leadfield_config_path))
                info = LeadfieldSimulator(config=config, logger=self.logger).handle_info()
            except Exception as e:
                self.logger.warning(f"Could not load info from config {self.leadfield_config_path} for visualization: {e}")

            self.visualize_signals(
                x=x_all_trials[first_trial_idx],
                y_clean=y_clean_all_trials[first_trial_idx],
                y_noisy=y_noisy_all_trials[first_trial_idx],
                nnz_to_plot=self.nnz,
                sfreq=info["sfreq"] if info else 250.0, # Default sfreq if info missing
                max_sensors=3,
                plot_sensors_together=False,
                show=False,
                save_path=os.path.join(save_path, "data_sim.png"),
            )
            
            # evoked_first_trial = mne.EvokedArray(
            #     data=y_noisy_all_trials[first_trial_idx],
            #     info=info,
            #     tmin=self.tmin
            # ) 

            # fig = mne.viz.plot_evoked_topo(evoked_first_trial, show=False, )
            # fig_path = os.path.join(save_path, "evoked_topo.png")
            # fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            # plt.close(fig)
            
            # mne.viz.plot_evoked(evoked_first_trial)
            
            self.visualize_leadfield_summary(
                L,
                orientation_type=self.orientation_type,
                bins=100,
                sensor_indices_to_plot=list(range(self.n_sensors)),
                # max_sensors_to_plot=10, # Let the function select if sensor_indices_to_plot is None
                save_path=os.path.join(save_path, "leadfield_summary.png"),
                show=False
            )
                    
            # self.visualize_leadfield_sensor_boxplot(
            #     L,
            #     orientation_type=self.orientation_type, 
            #     sensor_indices_to_plot=list(range(self.n_sensors)), 
            #     max_sensors_to_plot=20,
            #     save_path=os.path.join(save_path, "leadfield_sensor_boxplot.png"),
            #     show=False
            # )
            
            # self.visualize_leadfield_distribution(
            #     L,
            #     orientation_type=self.orientation_type,
            #     bins=100,
            #     save_path=os.path.join(save_path, "leadfield_distribution.png"),
            #     show=False
            # )

            # self.visualize_leadfield(
            #      L,
            #      orientation_type=self.orientation_type,
            #      save_path=os.path.join(save_path, "leadfield_matrix.png"),
            #      show=False
            # )

            if info:
                self.visualize_leadfield_topomap(
                    leadfield_matrix=L,
                    info=info,
                    x=x_all_trials[first_trial_idx],
                    orientation_type=self.orientation_type,
                    title="Leadfield Topomap for Some Active (Nonzero) Sources",
                    save_path=os.path.join(save_path, "leadfield_topomap.png"),
                    show=False,
                )
            else:
                 self.logger.info("Skipping leadfield topomap visualization due to missing MNE info.")






            print(f"\nPlotting results for trial {first_trial_idx + 1}...")

            # Now plot_sensor_signals uses the clean and noisy data generated separately
            self.plot_sensor_signals( 
                y_clean=y_clean_all_trials[first_trial_idx], # Use stored clean data
                y_noisy=y_noisy_all_trials[first_trial_idx],       # Use stored noisy data
                sensor_indices=sensor_subplots_indices,
                times=time_vector,
                save_dir=save_path,
                figure_name=f"specific_sensor_signals_subplots_trial{first_trial_idx}",
                trial_idx=first_trial_idx
            )

            self.plot_all_active_sources_single_figure(
                x=x_all_trials[first_trial_idx],
                times=time_vector,
                active_indices=active_indices_all_trials[first_trial_idx],
                stim_onset=self.stim_onset,
                save_dir=save_path,
                figure_name=f"active_sources_single_figure_trial{first_trial_idx}",
                trial_idx=first_trial_idx
            )

            # Plot data from the first trial (y_noisy_all_trials[first_trial_idx] is 2D)
            # average_epochs=False tells the function to treat the 2D data as a single trial
            self.plot_all_sensor_signals_single_figure(
                y_data=y_noisy_all_trials[first_trial_idx], # Pass single trial noisy data (2D)
                times=time_vector,
                save_dir=save_path,
                figure_name=f"all_sensor_signals_single_figure_trial{first_trial_idx}",
                trial_idx=first_trial_idx,
                average_epochs=False
            )

            # Plot average across trials (y_noisy_all_trials is 3D)
            # average_epochs=True tells the function to average the 3D data across trials
            self.plot_all_sensor_signals_single_figure(
                y_data=y_noisy_all_trials, # Pass all trial noisy data (3D array)
                times=time_vector,
                save_dir=save_path,
                figure_name="all_sensor_signals_average_trials",
                average_epochs=True
            )    


            self.plot_active_sources(
                x=x_all_trials[first_trial_idx],
                times=time_vector,
                active_indices=active_indices_all_trials[first_trial_idx],
                stim_onset=self.stim_onset,
                nnz=self.nnz,
                save_dir=save_path,
                figure_name=f"active_sources_subplots_trial{first_trial_idx}",
                trial_idx=first_trial_idx
            )



        # Reshape leadfield matrix for free orientation if needed by downstream estimators.
        # This might be a workaround depending on estimator expectations.
        if self.orientation_type == "free":
            self.logger.debug("Reshaping free orientation leadfield from (sensors, sources, 3) to (sensors, sources*3)")
            L = L.reshape(L.shape[0], -1)

        return y_noisy_all_trials, L, x_all_trials, active_indices_all_trials, noise_all_trials, noise_power_all_trials


    def visualize_signals(
        self,
        x: np.ndarray,
        y_clean: np.ndarray,
        y_noisy: np.ndarray,
        active_sources: Optional[np.ndarray] = None,
        nnz_to_plot: int = -1,
        sfreq: float = 100.0, # This sfreq is passed, consider using self.sfreq if more consistent
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
        Includes a line indicating stimulus onset.
        Uses self.tmin and self.tmax for the time axis.

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
            Sampling frequency in Hz, by default self.sfreq.
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
        # Use self.sfreq if the passed sfreq is the default placeholder, otherwise use passed sfreq
        current_sfreq = self.sfreq if sfreq == 100.0 and hasattr(self, 'sfreq') else sfreq
        
        n_times_from_data = y_clean.shape[1]
        
        # Generate time vector using self.tmin, self.tmax, and current_sfreq
        # Ensure n_times matches the data provided. If tmin/tmax/sfreq imply a different
        # n_times than the data, prioritize the data's n_times for indexing.
        times_from_params = np.arange(self.tmin, self.tmax, 1.0 / current_sfreq)
        
        if len(times_from_params) != n_times_from_data:
            self.logger.warning(
                f"Mismatch between n_times from data ({n_times_from_data}) and "
                f"n_times from tmin/tmax/sfreq ({len(times_from_params)}). "
                f"Using time axis derived from data length and tmin, sfreq."
            )
            times = np.linspace(self.tmin, self.tmin + (n_times_from_data - 1) / current_sfreq, n_times_from_data)
        else:
            times = times_from_params


        if active_sources is None:
            if self.orientation_type == "fixed":
                active_sources = np.where(np.any(x != 0, axis=-1))[0]
            elif self.orientation_type == "free":
                self.logger.info("Calculating norm of source activity to find active sources for free orientation.")
                active_sources = np.where(np.any(x != 0, axis=(1, 2)))[0]
            else:
                 raise ValueError(f"Unsupported orientation type: {self.orientation_type}")


        if nnz_to_plot != -1 and len(active_sources) > nnz_to_plot:
             plot_indices = self.rng.choice(active_sources, nnz_to_plot, replace=False)
             self.logger.info(f"Plotting {nnz_to_plot} randomly selected active sources out of {len(active_sources)}.")
        else:
             plot_indices = active_sources
             nnz_to_plot = len(plot_indices)

        y_min = min(y_clean.min(), y_noisy.min())
        y_max = max(y_clean.max(), y_noisy.max())
        y_range = y_max - y_min if y_max > y_min else 1.0

        num_sensors_to_plot = min(max_sensors, y_clean.shape[0])
        total_plots = 1 + (1 if plot_sensors_together else num_sensors_to_plot)
        fig, axes = plt.subplots(
            total_plots,
            1,
            figsize=figsize,
            gridspec_kw={"height_ratios": [1] * total_plots}, 
            sharex=True
        )
        if total_plots == 1:
            axes = [axes]

        ax_sources = axes[0]
        if self.orientation_type == "fixed":
            for i in plot_indices:
                ax_sources.plot(times, x[i].T, label=f"Source {i}")
        elif self.orientation_type == "free":
            for i in plot_indices:
                source_norm = np.linalg.norm(x[i], axis=0)
                ax_sources.plot(times, source_norm, label=f"Source {i} (Norm)")

        ax_sources.axvline(self.stim_onset, color='k', linestyle='--', linewidth=1, label='Stimulus Onset')
        ax_sources.set_title(f"{nnz_to_plot} Active Simulated Source Activity")
        ax_sources.set_ylabel("Amplitude (nAm)")
        ax_sources.grid(True)
        ax_sources.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        sensor_axes = axes[1:]
        if plot_sensors_together:
            ax_sensors = sensor_axes[0]
            current_shift = 0
            for i in range(num_sensors_to_plot):
                ax_sensors.plot(times, y_clean[i] + current_shift, label=f"Clean (Sensor {i})", linewidth=1.5)
                ax_sensors.plot(times, y_noisy[i] + current_shift, label=f"Noisy (Sensor {i})", linewidth=1.5)
                current_shift += shift 
            ax_sensors.axvline(self.stim_onset, color='k', linestyle='--', linewidth=1, label='Stimulus Onset')
            ax_sensors.set_title("Sensor Measurements")
            ax_sensors.set_ylabel("Amplitude (μV)") 
            ax_sensors.grid(True)
            # Consolidate legend for "Stimulus Onset" if it's plotted multiple times
            handles, labels = ax_sensors.get_legend_handles_labels()
            by_label = dict(zip(labels, handles)) # Remove duplicate labels
            ax_sensors.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            for idx, ax_sens in enumerate(sensor_axes):
                ax_sens.plot(times, y_clean[idx], label=f"Clean", linewidth=1.5)
                ax_sens.plot(times, y_noisy[idx], label=f"Noisy", linewidth=1)
                ax_sens.axvline(self.stim_onset, color='k', linestyle='--', linewidth=1, label='Stimulus Onset')
                ax_sens.set_title(f"Sensor {idx}")
                ax_sens.set_ylabel("Amplitude (μV)")
                ax_sens.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range) 
                ax_sens.grid(True)
                handles, labels = ax_sens.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax_sens.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))

        axes[-1].set_xlabel("Time (s)")

        plt.tight_layout(rect=[0, 0, 0.85, 1]) 

        if save_path:
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
            self.logger.info(f"Signals visualization saved to {save_path}")

        if show:
            plt.show()
        plt.close(fig)

    # def visualize_leadfield(
    #     self,
    #     leadfield_matrix: np.ndarray,
    #     orientation_type: str = "fixed",
    #     save_path: Optional[str] = None,
    #     show: bool = False
    # ) -> None:
    #     """
    #     Visualize the leadfield matrix as a heatmap.

    #     Parameters
    #     ----------
    #     leadfield_matrix : np.ndarray
    #         The leadfield matrix.
    #         - 'fixed': Shape (n_sensors, n_sources).
    #         - 'free': Shape (n_sensors, n_sources, 3).
    #     orientation_type : str, optional
    #         Orientation type ('fixed' or 'free'), by default "fixed".
    #     save_path : Optional[str], optional
    #         Path to save the figure. If None, not saved, by default None.
    #     show : bool, optional
    #         If True, display the plot, by default False.

    #     Raises
    #     ------
    #     ValueError
    #         If leadfield_matrix is invalid or orientation_type is unsupported.
    #     """
    #     if leadfield_matrix is None or not isinstance(leadfield_matrix, np.ndarray) or leadfield_matrix.size == 0:
    #         self.logger.error("Invalid leadfield matrix provided for visualization.")
    #         return

    #     fig = None # Initialize fig
    #     try:
    #         if orientation_type == "fixed":
    #             if leadfield_matrix.ndim != 2:
    #                  raise ValueError(f"Expected 2D leadfield for fixed orientation, got {leadfield_matrix.ndim}D")
    #             fig, ax = plt.subplots(figsize=(10, 8))
    #             im = ax.imshow(leadfield_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    #             fig.colorbar(im, ax=ax, label="Amplitude (µV / nAm)", fraction=0.05, pad=0.04)
    #             ax.set_title("Leadfield Matrix (Fixed Orientation)")
    #             ax.set_xlabel("Sources")
    #             ax.set_ylabel("Sensors")
    #         elif orientation_type == "free":
    #             if leadfield_matrix.ndim != 3 or leadfield_matrix.shape[-1] != 3:
    #                  raise ValueError(f"Expected 3D leadfield (..., 3) for free orientation, got shape {leadfield_matrix.shape}")
    #             n_orient = leadfield_matrix.shape[-1]
    #             fig, axes = plt.subplots(1, n_orient, figsize=(15, 5), sharey=True)
    #             if n_orient == 1: axes = [axes] # Ensure axes is iterable
    #             orientations = ["X", "Y", "Z"]
    #             images = []
    #             for i in range(n_orient):
    #                 im = axes[i].imshow(leadfield_matrix[:, :, i], aspect='auto', cmap='viridis', interpolation='nearest')
    #                 images.append(im)
    #                 axes[i].set_title(f"Leadfield Matrix ({orientations[i]})")
    #                 axes[i].set_xlabel("Sources")
    #             axes[0].set_ylabel("Sensors")
    #             fig.colorbar(images[0], ax=axes, location="right", label="Amplitude (µV / nAm)", fraction=0.05, pad=0.04)
    #         else:
    #             raise ValueError("Invalid orientation type. Must be 'fixed' or 'free'.")

    #         plt.tight_layout()

    #         if save_path:
    #             save_dir = Path(save_path).parent
    #             save_dir.mkdir(parents=True, exist_ok=True)
    #             plt.savefig(save_path, bbox_inches="tight")
    #             self.logger.info(f"Leadfield matrix visualization saved to {save_path}")
    #         if show:
    #             plt.show()

    #     except Exception as e:
    #          self.logger.error(f"Failed during leadfield visualization: {e}")
    #     finally:
    #          if fig:
    #              plt.close(fig)

    # def visualize_leadfield_distribution(
    #     self,
    #     leadfield_matrix: np.ndarray,
    #     orientation_type: str = "fixed",
    #     bins: int = 100,
    #     save_path: Optional[str] = None,
    #     title: Optional[str] = None,
    #     show: bool = False
    # ) -> None:
    #     """
    #     Visualize the distribution of leadfield amplitude values using a histogram.

    #     Parameters
    #     ----------
    #     leadfield_matrix : np.ndarray
    #         The leadfield matrix.
    #         - 'fixed': Shape (n_sensors, n_sources).
    #         - 'free': Shape (n_sensors, n_sources, 3).
    #     orientation_type : str, optional
    #         Orientation type ('fixed' or 'free'), by default "fixed".
    #         This mainly affects the title and interpretation.
    #     bins : int, optional
    #         Number of bins for the histogram, by default 100.
    #     save_path : Optional[str], optional
    #         Path to save the figure. If None, not saved, by default None.
    #     title : Optional[str], optional
    #         Custom title for the plot. If None, a default title is generated.
    #     show : bool, optional
    #         If True, display the plot, by default False.
    #     """
    #     if leadfield_matrix is None or not isinstance(leadfield_matrix, np.ndarray) or leadfield_matrix.size == 0:
    #         self.logger.error("Invalid leadfield matrix provided for distribution visualization.")
    #         return

    #     fig = None # Initialize fig
    #     try:
    #         fig, ax = plt.subplots(figsize=(10, 6))

    #         # Flatten the leadfield matrix to get all values for the histogram
    #         # For 'free' orientation, this will include values from all X, Y, Z components.
    #         leadfield_values_flat = leadfield_matrix.flatten()

    #         ax.hist(leadfield_values_flat, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)

    #         if title is None:
    #             default_title = f"Distribution of Leadfield Amplitudes ({orientation_type.capitalize()} Orientation)"
    #             ax.set_title(default_title, fontsize=14)
    #         else:
    #             ax.set_title(title, fontsize=14)

    #         ax.set_xlabel("Leadfield Amplitude (µV / nAm)", fontsize=12)
    #         ax.set_ylabel("Frequency", fontsize=12)
    #         ax.grid(True, linestyle='--', alpha=0.7)

    #         # Add some statistics to the plot
    #         mean_val = np.mean(leadfield_values_flat)
    #         std_val = np.std(leadfield_values_flat)
    #         median_val = np.median(leadfield_values_flat)
    #         min_val = np.min(leadfield_values_flat)
    #         max_val = np.max(leadfield_values_flat)

    #         stats_text = (
    #             f"Mean: {mean_val:.2e}\nStd: {std_val:.2e}\nMedian: {median_val:.2e}\n"
    #             f"Min: {min_val:.2e}\nMax: {max_val:.2e}\nN Values: {len(leadfield_values_flat)}"
    #         )
    #         # Position the text box in the upper right corner
    #         ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
    #                 verticalalignment='top', horizontalalignment='right',
    #                 bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))


    #         plt.tight_layout()

    #         if save_path:
    #             save_dir = Path(save_path).parent
    #             save_dir.mkdir(parents=True, exist_ok=True)
    #             plt.savefig(save_path, bbox_inches="tight")
    #             self.logger.info(f"Leadfield distribution visualization saved to {save_path}")
    #         if show:
    #             plt.show()

    #     except Exception as e:
    #          self.logger.error(f"Failed during leadfield distribution visualization: {e}")
    #     finally:
    #          if fig:
    #              plt.close(fig)
   


    # def visualize_leadfield_summary(
    #     self,
    #     leadfield_matrix: np.ndarray,
    #     orientation_type: str = "fixed",
    #     bins: int = 100,
    #     sensor_indices_to_plot: Optional[List[int]] = None,
    #     max_sensors_to_plot: int = 10,
    #     main_title: Optional[str] = None,
    #     save_path: Optional[str] = None,
    #     show: bool = False
    # ) -> None:
    #     """
    #     Visualize a summary of the leadfield matrix in a single figure:
    #     1. Top: Heatmap of the leadfield (norm for 'free' orientation).
    #     2. Bottom-Left: Box plots of leadfield amplitudes for selected sensors.
    #     3. Bottom-Right: Rotated histogram of all leadfield amplitudes (marginal to boxplots).

    #     Parameters
    #     ----------
    #     leadfield_matrix : np.ndarray
    #         The leadfield matrix.
    #         - 'fixed': Shape (n_sensors, n_sources).
    #         - 'free': Shape (n_sensors, n_sources, 3).
    #     orientation_type : str, optional
    #         Orientation type ('fixed' or 'free'), by default "fixed".
    #     bins : int, optional
    #         Number of bins for the histogram subplot, by default 100.
    #     sensor_indices_to_plot : Optional[List[int]], optional
    #         Specific list of sensor indices for the box plot. If None, a subset is chosen.
    #     max_sensors_to_plot : int, optional
    #         Maximum number of sensors for the box plot if sensor_indices_to_plot is None.
    #     main_title : Optional[str], optional
    #         Overall title for the figure.
    #     save_path : Optional[str], optional
    #         Path to save the figure.
    #     show : bool, optional
    #         If True, display the plot.
    #     """
    #     if leadfield_matrix is None or not isinstance(leadfield_matrix, np.ndarray) or leadfield_matrix.size == 0:
    #         self.logger.error("Invalid leadfield matrix provided for summary visualization.")
    #         return

    #     fig = None
    #     try:
    #         # Define the layout using GridSpec
    #         # Figure will have 2 main rows. The second row is split into 2 columns.
    #         # Heatmap takes more vertical space.
    #         fig = plt.figure(figsize=(15, 18)) # Adjusted figsize
    #         gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1], width_ratios=[3, 1])

    #         ax_heatmap = fig.add_subplot(gs[0, :])  # Heatmap spans both columns of the first row
    #         ax_boxplot = fig.add_subplot(gs[1, 0])  # Boxplot in the second row, first column
    #         ax_hist_y = fig.add_subplot(gs[1, 1], sharey=ax_boxplot) # Rotated histogram, shares y-axis with boxplot

    #         if main_title is None:
    #             default_main_title = f"Leadfield Matrix Summary ({orientation_type.capitalize()} Orientation)"
    #             fig.suptitle(default_main_title, fontsize=18, y=0.99)
    #         elif main_title:
    #             fig.suptitle(main_title, fontsize=18, y=0.99)

    #         # --- Subplot 1: Leadfield Heatmap (ax_heatmap) ---
    #         if orientation_type == "fixed":
    #             if leadfield_matrix.ndim != 2:
    #                 raise ValueError(f"Heatmap: Expected 2D leadfield for fixed, got {leadfield_matrix.ndim}D")
    #             lf_to_plot = leadfield_matrix
    #             heatmap_title = "Leadfield Matrix (Fixed Orientation)"
    #         elif orientation_type == "free":
    #             if leadfield_matrix.ndim != 3 or leadfield_matrix.shape[-1] != 3:
    #                 raise ValueError(f"Heatmap: Expected 3D leadfield (..., 3) for free, got {leadfield_matrix.shape}")
    #             lf_to_plot = np.linalg.norm(leadfield_matrix, axis=-1)
    #             heatmap_title = "Leadfield Matrix (Free Orientation - Norm)"
    #         else:
    #             raise ValueError("Heatmap: Invalid orientation type.")
            
    #         im = ax_heatmap.imshow(lf_to_plot, aspect='auto', cmap='viridis', interpolation='nearest')
    #         # Add colorbar to the heatmap subplot
    #         cbar = fig.colorbar(im, ax=ax_heatmap, label="Amplitude (µV / nAm)", fraction=0.046, pad=0.04, orientation='vertical')
    #         ax_heatmap.set_title(heatmap_title, fontsize=14)
    #         ax_heatmap.set_xlabel("Sources", fontsize=12)
    #         ax_heatmap.set_ylabel("Sensors", fontsize=12)

    #         # --- Data for Histogram and Boxplot ---
    #         leadfield_values_flat = leadfield_matrix.flatten() # For overall distribution
    #         num_total_sensors = leadfield_matrix.shape[0]
    #         actual_sensor_indices_to_plot: np.ndarray

    #         if sensor_indices_to_plot is None:
    #             if num_total_sensors > max_sensors_to_plot:
    #                 actual_sensor_indices_to_plot = np.linspace(0, num_total_sensors - 1, max_sensors_to_plot, dtype=int)
    #             else:
    #                 actual_sensor_indices_to_plot = np.arange(num_total_sensors)
    #         else:
    #             actual_sensor_indices_to_plot = np.array(sensor_indices_to_plot, dtype=int)
    #             if np.any(actual_sensor_indices_to_plot < 0) or np.any(actual_sensor_indices_to_plot >= num_total_sensors):
    #                 self.logger.error("Boxplot: Invalid sensor_indices_to_plot.")
    #                 ax_boxplot.text(0.5, 0.5, "Error: Invalid sensor indices.", ha='center', va='center', color='red')
    #                 actual_sensor_indices_to_plot = np.array([])

    #         # --- Subplot 2: Leadfield Sensor Box Plots (ax_boxplot) ---
    #         if len(actual_sensor_indices_to_plot) > 0:
    #             data_for_boxplot = []
    #             labels_for_boxplot = []
    #             for sensor_idx in actual_sensor_indices_to_plot:
    #                 if orientation_type == "fixed":
    #                     sensor_values = leadfield_matrix[sensor_idx, :]
    #                 elif orientation_type == "free":
    #                     sensor_values_3d = leadfield_matrix[sensor_idx, :, :]
    #                     sensor_values = np.linalg.norm(sensor_values_3d, axis=-1)
    #                 else:
    #                     raise ValueError("Boxplot: Invalid orientation type.")
    #                 data_for_boxplot.append(sensor_values)
    #                 labels_for_boxplot.append(str(sensor_idx))
                
    #             bp = ax_boxplot.boxplot(data_for_boxplot, patch_artist=True, medianprops=dict(color="black", linewidth=1.5), vert=True)
    #             try:
    #                 colors_list = cm.get_cmap('viridis', len(data_for_boxplot))
    #                 for i, patch in enumerate(bp['boxes']):
    #                     patch.set_facecolor(colors_list(i / len(data_for_boxplot)))
    #             except AttributeError:
    #                  self.logger.warning("Boxplot: Could not apply distinct colors.")
                
    #             ax_boxplot.set_title("Leadfield Amplitude per Sensor", fontsize=14)
    #             ax_boxplot.set_xlabel("Sensor Index", fontsize=12)
    #             ax_boxplot.set_ylabel("Leadfield Amplitude (µV / nAm)", fontsize=12)
    #             ax_boxplot.set_xticklabels(labels_for_boxplot, rotation=45, ha="right" if len(labels_for_boxplot) > 5 else "center")
    #             ax_boxplot.grid(True, linestyle='--', alpha=0.6, axis='y')
    #         elif not (np.any(actual_sensor_indices_to_plot < 0) or np.any(actual_sensor_indices_to_plot >= num_total_sensors)):
    #             ax_boxplot.text(0.5, 0.5, "No sensors for boxplot.", ha='center', va='center')
    #             ax_boxplot.set_xlabel("Sensor Index", fontsize=12)
    #             ax_boxplot.set_ylabel("Leadfield Amplitude (µV / nAm)", fontsize=12)


    #         # --- Subplot 3: Rotated Histogram (ax_hist_y) ---
    #         # This histogram shows the distribution of ALL leadfield values
    #         ax_hist_y.hist(leadfield_values_flat, bins=bins, color='skyblue', edgecolor='black', alpha=0.7, orientation='horizontal')
    #         ax_hist_y.set_title("Overall Distribution", fontsize=14)
    #         ax_hist_y.set_xlabel("Frequency", fontsize=12)
    #         # Remove y-tick labels for the histogram as it shares y-axis with boxplot
    #         plt.setp(ax_hist_y.get_yticklabels(), visible=False)
    #         ax_hist_y.grid(True, linestyle='--', alpha=0.7, axis='x')

    #         mean_val = np.mean(leadfield_values_flat)
    #         std_val = np.std(leadfield_values_flat)
    #         median_val = np.median(leadfield_values_flat)
    #         stats_text = (
    #             f"Mean: {mean_val:.2e}\nStd: {std_val:.2e}\nMedian: {median_val:.2e}"
    #         )
    #         # Add stats text to the histogram plot, adjusting position for horizontal orientation
    #         ax_hist_y.text(0.95, 0.95, stats_text, transform=ax_hist_y.transAxes, fontsize=9,
    #                        verticalalignment='top', horizontalalignment='right',
    #                        bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))


    #         # Adjust layout
    #         gs.tight_layout(fig, rect=[0, 0, 1, 0.96] if main_title else [0,0,1,1]) # Use GridSpec's tight_layout

    #         if save_path:
    #             save_dir = Path(save_path).parent
    #             save_dir.mkdir(parents=True, exist_ok=True)
    #             plt.savefig(save_path, bbox_inches="tight", dpi=150) # Added dpi
    #             self.logger.info(f"Leadfield summary visualization saved to {save_path}")
    #         if show:
    #             plt.show()

    #     except Exception as e:
    #          self.logger.error(f"Failed during leadfield summary visualization: {e}", exc_info=True) # Added exc_info
    #     finally:
    #          if fig:
    #              plt.close(fig)
   

    # def visualize_leadfield_sensor_boxplot(
    #     self,
    #     leadfield_matrix: np.ndarray,
    #     orientation_type: str = "fixed",
    #     sensor_indices_to_plot: Optional[List[int]] = None,
    #     max_sensors_to_plot: int = 20,
    #     save_path: Optional[str] = None,
    #     custom_title: Optional[str] = None,
    #     show: bool = False
    # ) -> None:
    #     """
    #     Visualize the distribution of leadfield amplitudes for selected sensors using box plots.
    #     Each box plot represents one sensor, showing the distribution of its leadfield
    #     values across all sources. For 'free' orientation, the norm of the 3 components
    #     is used for each source-sensor pair.

    #     Parameters
    #     ----------
    #     leadfield_matrix : np.ndarray
    #         The leadfield matrix.
    #         - 'fixed': Shape (n_sensors, n_sources).
    #         - 'free': Shape (n_sensors, n_sources, 3).
    #     orientation_type : str, optional
    #         Orientation type ('fixed' or 'free'), by default "fixed".
    #     sensor_indices_to_plot : Optional[List[int]], optional
    #         Specific list of sensor indices to plot. If None, a subset is chosen
    #         based on max_sensors_to_plot, by default None.
    #     max_sensors_to_plot : int, optional
    #         Maximum number of sensors to create box plots for if sensor_indices_to_plot
    #         is None, by default 20.
    #     save_path : Optional[str], optional
    #         Path to save the figure. If None, not saved, by default None.
    #     custom_title : Optional[str], optional
    #         Custom title for the plot. If None, a default title is generated.
    #     show : bool, optional
    #         If True, display the plot, by default False.
    #     """
    #     if leadfield_matrix is None or not isinstance(leadfield_matrix, np.ndarray) or leadfield_matrix.size == 0:
    #         self.logger.error("Invalid leadfield matrix provided for box plot visualization.")
    #         return

    #     fig = None # Initialize fig
    #     try:
    #         num_total_sensors = leadfield_matrix.shape[0]

    #         if sensor_indices_to_plot is None:
    #             if num_total_sensors > max_sensors_to_plot:
    #                 # Select evenly spaced sensors
    #                 selected_indices = np.linspace(0, num_total_sensors - 1, max_sensors_to_plot, dtype=int)
    #                 self.logger.info(f"Plotting box plots for {max_sensors_to_plot} selected sensors out of {num_total_sensors}.")
    #             else:
    #                 selected_indices = np.arange(num_total_sensors)
    #         else:
    #             selected_indices = np.array(sensor_indices_to_plot, dtype=int)
    #             if np.any(selected_indices < 0) or np.any(selected_indices >= num_total_sensors):
    #                 self.logger.error("Invalid sensor_indices_to_plot: indices out of bounds.")
    #                 return
            
    #         if len(selected_indices) == 0:
    #             self.logger.info("No sensors selected for box plot visualization.")
    #             return

    #         data_for_boxplot = []
    #         labels_for_boxplot = []

    #         for sensor_idx in selected_indices:
    #             if orientation_type == "fixed":
    #                 if leadfield_matrix.ndim != 2:
    #                     raise ValueError(f"Expected 2D leadfield for fixed orientation, got {leadfield_matrix.ndim}D shape {leadfield_matrix.shape}")
    #                 sensor_values = leadfield_matrix[sensor_idx, :]
    #             elif orientation_type == "free":
    #                 if leadfield_matrix.ndim != 3 or leadfield_matrix.shape[-1] != 3:
    #                     raise ValueError(f"Expected 3D leadfield (..., 3) for free orientation, got shape {leadfield_matrix.shape}")
    #                 sensor_values_3d = leadfield_matrix[sensor_idx, :, :] # Shape (n_sources, 3)
    #                 sensor_values = np.linalg.norm(sensor_values_3d, axis=-1) # Shape (n_sources,)
    #             else:
    #                 raise ValueError(f"Invalid orientation_type '{orientation_type}'. Choose 'fixed' or 'free'.")
                
    #             data_for_boxplot.append(sensor_values)
    #             labels_for_boxplot.append(str(sensor_idx))

    #         # Adjust figure width based on the number of boxplots, with a max width
    #         fig_width = min(max(10, len(selected_indices) * 0.7), 25)
    #         fig, ax = plt.subplots(figsize=(fig_width, 7))
            
    #         bp = ax.boxplot(data_for_boxplot, patch_artist=True, medianprops=dict(color="black", linewidth=1.5))

    #         # Optional: Color the boxes using a colormap
    #         # Ensure you have `import matplotlib.cm as cm`
    #         try:
    #             colors_list = cm.get_cmap('viridis', len(data_for_boxplot))
    #             for i, patch in enumerate(bp['boxes']):
    #                 patch.set_facecolor(colors_list(i / len(data_for_boxplot))) # Normalize index for colormap
    #         except AttributeError: # Fallback if get_cmap with number of colors is not supported (older matplotlib)
    #              self.logger.warning("Could not apply distinct colors to boxplots; using default or single color.")


    #         if custom_title is None:
    #             default_title = f"Leadfield Amplitude Distribution per Sensor ({orientation_type.capitalize()} Orientation)"
    #             ax.set_title(default_title, fontsize=14, pad=15)
    #         else:
    #             ax.set_title(custom_title, fontsize=14, pad=15)

    #         ax.set_xlabel("Sensor Index", fontsize=12)
    #         ax.set_ylabel("Leadfield Amplitude (µV / nAm)", fontsize=12)
    #         ax.set_xticklabels(labels_for_boxplot, rotation=45, ha="right" if len(labels_for_boxplot) > 10 else "center")
    #         ax.grid(True, linestyle='--', alpha=0.6, axis='y')

    #         plt.tight_layout()

    #         if save_path:
    #             save_dir = Path(save_path).parent
    #             save_dir.mkdir(parents=True, exist_ok=True)
    #             plt.savefig(save_path, bbox_inches="tight")
    #             self.logger.info(f"Leadfield sensor box plot visualization saved to {save_path}")
    #         if show:
    #             plt.show()

    #     except Exception as e:
    #          self.logger.error(f"Failed during leadfield sensor box plot visualization: {e}")
    #     finally:
    #          if fig:
    #              plt.close(fig)
          
          
    def visualize_leadfield_summary(
        self,
        leadfield_matrix: np.ndarray,
        orientation_type: str = "fixed",
        bins: int = 100,
        sensor_indices_to_plot: Optional[List[int]] = None,
        max_sensors_to_plot: int = 10,
        main_title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = False
    ) -> None:
        # ... (initial parameter validation and actual_sensor_indices_to_plot logic remains the same) ...
        if leadfield_matrix is None or not isinstance(leadfield_matrix, np.ndarray) or leadfield_matrix.size == 0:
            self.logger.error("Invalid leadfield matrix provided for summary visualization.")
            return

        fig = None
        try:
            num_total_sensors_in_lf = leadfield_matrix.shape[0]
            actual_sensor_indices_to_plot: np.ndarray

            if sensor_indices_to_plot is None:
                if num_total_sensors_in_lf > max_sensors_to_plot:
                    actual_sensor_indices_to_plot = np.linspace(0, num_total_sensors_in_lf - 1, max_sensors_to_plot, dtype=int)
                else:
                    actual_sensor_indices_to_plot = np.arange(num_total_sensors_in_lf)
            else:
                actual_sensor_indices_to_plot = np.array(sensor_indices_to_plot, dtype=int)
                if np.any(actual_sensor_indices_to_plot < 0) or np.any(actual_sensor_indices_to_plot >= num_total_sensors_in_lf):
                    self.logger.error("Summary Plot: Invalid sensor_indices_to_plot: indices out of bounds.")
                    if num_total_sensors_in_lf > 0 :
                        actual_sensor_indices_to_plot = np.arange(min(num_total_sensors_in_lf, max_sensors_to_plot))
                        self.logger.warning(f"Defaulting to plotting first {len(actual_sensor_indices_to_plot)} sensors for heatmap/boxplot.")
                    else:
                        actual_sensor_indices_to_plot = np.array([])

            fig = plt.figure(figsize=(16, 18)) # Adjust figsize as needed

            # Main GridSpec: 2 rows, 1 column. Each row will be further divided.
            gs_rows = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1]) # Adjust height_ratios if needed

            # --- Top Row: Heatmap and its Colorbar ---
            # To make the heatmap image wider, increase the first ratio (e.g., 0.95)
            # and decrease the second (e.g., 0.03), ensuring they make sense for the space.
            gs_top_row = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_rows[0],
                                                          width_ratios=[0.50, 0.03], # Example: Heatmap image gets 93%, colorbar 5% of top row width
                                                          wspace=0.5) # Adjust space between heatmap image and its colorbar
            ax_heatmap_img = fig.add_subplot(gs_top_row[0, 0])
            cax_heatmap_cb = fig.add_subplot(gs_top_row[0, 1])

            # --- Bottom Row: Boxplot and Histogram ---
            gs_bottom_row = gridspec.GridSpecFromSubplotSpec(1, 2,
                                                             subplot_spec=gs_rows[1],
                                                             width_ratios=[0.75, 0.25], # Example: Boxplot 75%, histogram 25% of bottom row width
                                                             wspace=0.02) # Adjust space between boxplot and histogram
            ax_boxplot = fig.add_subplot(gs_bottom_row[0, 0], sharex=ax_heatmap_img) # Boxplot shares X with heatmap IMAGE
            ax_hist_y = fig.add_subplot(gs_bottom_row[0, 1], sharey=ax_boxplot)  # Histogram shares Y with boxplot

            if main_title is None:
                default_main_title = f"Leadfield Matrix Summary ({orientation_type.capitalize()} Orientation)"
                fig.suptitle(default_main_title, fontsize=18, y=0.99)
            elif main_title:
                fig.suptitle(main_title, fontsize=18, y=0.99)

            # ... (rest of the plotting logic for heatmap, boxplot, histogram remains the same as the previous version) ...
            # --- Prepare data for heatmap (lf_for_heatmap: sources on Y, selected sensors on X) ---
            if orientation_type == "fixed":
                if leadfield_matrix.ndim != 2:
                    raise ValueError(f"Heatmap: Expected 2D leadfield for fixed, got {leadfield_matrix.ndim}D")
                lf_norm_for_heatmap = leadfield_matrix
                heatmap_title_suffix = "(Fixed Orientation)"
            elif orientation_type == "free":
                if leadfield_matrix.ndim != 3 or leadfield_matrix.shape[-1] != 3:
                    raise ValueError(f"Heatmap: Expected 3D leadfield (..., 3) for free, got {leadfield_matrix.shape}")
                lf_norm_for_heatmap = np.linalg.norm(leadfield_matrix, axis=-1)
                heatmap_title_suffix = "(Free Orientation - Norm)"
            else:
                raise ValueError("Heatmap: Invalid orientation type.")

            if len(actual_sensor_indices_to_plot) > 0:
                lf_selected_sensors = lf_norm_for_heatmap[actual_sensor_indices_to_plot, :]
                data_for_heatmap_display = lf_selected_sensors.T
            else:
                data_for_heatmap_display = np.array([[]])
                ax_heatmap_img.text(0.5, 0.5, "No sensors for heatmap.", ha='center', va='center')

            # --- Subplot 1: Flipped Leadfield Heatmap (ax_heatmap_img) & Colorbar (cax_heatmap_cb) ---
            if data_for_heatmap_display.size > 0 :
                im = ax_heatmap_img.imshow(data_for_heatmap_display, aspect='auto', cmap='viridis', interpolation='nearest')
                fig.colorbar(im, cax=cax_heatmap_cb, label="Amplitude (µV / nAm)")
                ax_heatmap_img.set_title(f"Leadfield Matrix {heatmap_title_suffix}", fontsize=14)
                ax_heatmap_img.set_ylabel("Sources", fontsize=12)
                ax_heatmap_img.set_xlabel("Sensor Index", fontsize=12)
            else:
                ax_heatmap_img.set_title(f"Leadfield Matrix {heatmap_title_suffix}", fontsize=14)
                ax_heatmap_img.set_ylabel("Sources", fontsize=12)
                ax_heatmap_img.set_xlabel("Sensor Index", fontsize=12) # Fallback if no data

            # --- Data for Histogram (Overall Distribution) ---
            leadfield_values_flat = leadfield_matrix.flatten()

            # --- Subplot 2: Leadfield Sensor Box Plots (ax_boxplot) ---
            labels_for_boxplot = [str(idx) for idx in actual_sensor_indices_to_plot]
            all_q1_values_for_boxplot_sensors = [] 
            all_q2_values_for_boxplot_sensors = [] 
            all_min_no_outliers_per_sensor = [] # Store min (no outliers) for each sensor's boxplot data
            all_max_no_outliers_per_sensor = [] # Store max (no outliers) for each sensor's boxplot data

            if len(actual_sensor_indices_to_plot) > 0:
                data_for_boxplot = []
                for sensor_idx in actual_sensor_indices_to_plot:
                    current_sensor_data = None
                    if orientation_type == "fixed":
                        current_sensor_data = leadfield_matrix[sensor_idx, :]
                    elif orientation_type == "free":
                        sensor_values_3d = leadfield_matrix[sensor_idx, :, :]
                        current_sensor_data = np.linalg.norm(sensor_values_3d, axis=-1)
                    else: 
                        self.logger.error(f"Boxplot: Invalid orientation type '{orientation_type}' encountered unexpectedly. Raising ValueError.")
                        raise ValueError("Boxplot: Invalid orientation type.")
                    data_for_boxplot.append(current_sensor_data)

                    if current_sensor_data.size > 0:
                        all_q1_values_for_boxplot_sensors.append(np.percentile(current_sensor_data, 25))
                        all_q2_values_for_boxplot_sensors.append(np.percentile(current_sensor_data, 50))

                        # Calculate min/max without outliers for THIS sensor's data
                        q1_sensor = np.percentile(current_sensor_data, 25)
                        q3_sensor = np.percentile(current_sensor_data, 75)
                        iqr_sensor = q3_sensor - q1_sensor
                        lower_bound_sensor = q1_sensor - 1.5 * iqr_sensor
                        upper_bound_sensor = q3_sensor + 1.5 * iqr_sensor
                        
                        sensor_data_no_outliers = current_sensor_data[
                            (current_sensor_data >= lower_bound_sensor) &
                            (current_sensor_data <= upper_bound_sensor)
                        ]
                        
                        if sensor_data_no_outliers.size > 0:
                            all_min_no_outliers_per_sensor.append(np.min(sensor_data_no_outliers))
                            all_max_no_outliers_per_sensor.append(np.max(sensor_data_no_outliers))
                        else:
                            # If all data for a sensor are outliers or it's empty after filtering
                            all_min_no_outliers_per_sensor.append(np.nan)
                            all_max_no_outliers_per_sensor.append(np.nan)
                    else: # current_sensor_data.size == 0
                        all_min_no_outliers_per_sensor.append(np.nan)
                        all_max_no_outliers_per_sensor.append(np.nan)
                
                boxprops = dict(facecolor='skyblue', alpha=0.7, edgecolor='black')
                medianprops = dict(color="navy", linewidth=1.5)
                
                bp = ax_boxplot.boxplot(data_for_boxplot, patch_artist=True, labels=labels_for_boxplot,
                                        boxprops=boxprops, medianprops=medianprops, vert=True)
                
                ax_boxplot.set_title("Leadfield Amplitude per Sensor", fontsize=14)
                ax_boxplot.set_ylabel("Leadfield Amplitude (µV / nAm)", fontsize=12)
                ax_boxplot.grid(True, linestyle='--', alpha=0.6, axis='y')
                ax_boxplot.set_xlabel("Selected Sensor Index", fontsize=12) # This label will be visible
                plt.setp(ax_boxplot.get_xticklabels(), rotation=45, ha="right" if len(labels_for_boxplot) > 5 else "center")
            else:
                ax_boxplot.text(0.5, 0.5, "No sensors for boxplot.", ha='center', va='center')
                ax_boxplot.set_title("Leadfield Amplitude per Sensor", fontsize=14)
                ax_boxplot.set_xlabel("Selected Sensor Index", fontsize=12)
                ax_boxplot.set_ylabel("Leadfield Amplitude (µV / nAm)", fontsize=12)
                self.logger.info("No boxplots generated as no sensors were selected.")

            # Configure shared X-axis: Heatmap image X-ticks are based on boxplot's
            if len(actual_sensor_indices_to_plot) > 0 and data_for_heatmap_display.size > 0:
                ax_heatmap_img.set_xticks(np.arange(len(actual_sensor_indices_to_plot)))
                plt.setp(ax_heatmap_img.get_xticklabels(), visible=False)
            # ax_heatmap_img.set_xlabel("") # This was commented out in the provided context, keeping it so

            # --- Subplot 3: Rotated Histogram (ax_hist_y) ---
            ax_hist_y.hist(leadfield_values_flat, bins=bins, color='lightcoral', edgecolor='black', alpha=0.7, orientation='horizontal')
            ax_hist_y.set_title("Overall Distribution", fontsize=14)
            ax_hist_y.set_xlabel("Frequency", fontsize=12)
            plt.setp(ax_hist_y.get_yticklabels(), visible=False)
            ax_hist_y.grid(True, linestyle='--', alpha=0.7, axis='x')

            mean_val = np.mean(leadfield_values_flat)
            median_val = np.median(leadfield_values_flat)
            mean_abs_val = np.mean(np.abs(leadfield_values_flat))
            std_val = np.std(leadfield_values_flat)
            min_val_flat = np.min(leadfield_values_flat) # Overall min (with outliers)
            max_val_flat = np.max(leadfield_values_flat) # Overall max (with outliers)

            # Calculate mean of Q1 and Q2 values from the boxplot data
            mean_of_boxplot_q1s = np.nanmean(all_q1_values_for_boxplot_sensors) if all_q1_values_for_boxplot_sensors else np.nan
            mean_of_boxplot_q2s = np.nanmean(all_q2_values_for_boxplot_sensors) if all_q2_values_for_boxplot_sensors else np.nan
            
            # Calculate mean of sensor-wise min/max (no outliers)
            mean_of_sensor_mins_no_outliers = np.nanmean(all_min_no_outliers_per_sensor) if all_min_no_outliers_per_sensor else np.nan
            mean_of_sensor_maxs_no_outliers = np.nanmean(all_max_no_outliers_per_sensor) if all_max_no_outliers_per_sensor else np.nan
            
            self.logger.info(f"Leadfield overall flat data stats: N_values={len(leadfield_values_flat)}, Mean={mean_val:.2e}, Std={std_val:.2e}, Median={median_val:.2e}, Min={min_val_flat:.2e}, Max={max_val_flat:.2e}, Mean Abs={mean_abs_val:.2e}")
            self.logger.info(f"Leadfield boxplot sensors stats: Mean of Q1s={mean_of_boxplot_q1s:.2e}, Mean of Q2s (Medians)={mean_of_boxplot_q2s:.2e} (for {len(all_q1_values_for_boxplot_sensors)} sensors)")
            self.logger.info(f"Leadfield boxplot sensors (no outliers): Mean of Mins={mean_of_sensor_mins_no_outliers:.2e}, Mean of Maxs={mean_of_sensor_maxs_no_outliers:.2e}")
            
            stats_text = (f"Overall Mean: {mean_val:.2e}\n"
                          f"Overall Median: {median_val:.2e}\n"
                          f"Overall Std: {std_val:.2e}\n"
                          f"Overall Min: {min_val_flat:.2e}\n"
                          f"Overall Max: {max_val_flat:.2e}\n"
                          f"Mean Abs: {mean_abs_val:.2e}\n"
                          f"Mean Boxplot Q1s: {mean_of_boxplot_q1s:.2e}\n"
                          f"Mean Boxplot Q2s: {mean_of_boxplot_q2s:.2e}\n"
                          f"Mean Sensor Min (no outliers): {mean_of_sensor_mins_no_outliers:.2e}\n"
                          f"Mean Sensor Max (no outliers): {mean_of_sensor_maxs_no_outliers:.2e}")
            
            ax_hist_y.text(0.95, 0.95, stats_text, transform=ax_hist_y.transAxes, fontsize=9,verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

            
            fig.tight_layout(rect=[0, 0, 1, 0.97] if main_title else [0,0,1,1])

            if save_path:
                save_dir = Path(save_path).parent
                save_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, bbox_inches="tight", dpi=150)
                self.logger.info(f"Leadfield summary visualization saved to {save_path}")
            if show:
                plt.show()

        except Exception as e:
             self.logger.error(f"Failed during leadfield summary visualization: {e}", exc_info=True)
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

            vmax = np.max(all_leadfield_values)
            vmin = np.min(all_leadfield_values)

            for i, source_idx in enumerate(active_sources):
                leadfield_values = all_leadfield_values[i]
                im, _ = mne.viz.plot_topomap(
                    leadfield_values, info, axes=axes_flat[i], cmap="RdBu_r", # Use diverging colormap
                    # vlim=(vmin, vmax), 
                    show=False,
                    contours=6
                )
                axes_flat[i].set_title(f"Source {source_idx}")

            # Add a single colorbar
            fig.colorbar(im, ax=axes.ravel().tolist(), label='Leadfield Amplitude (µV / nAm)', shrink=0.6, aspect=10)

            # Hide unused subplots
            for j in range(n_active, len(axes_flat)):
                axes_flat[j].axis("off")

            if title:
                fig.suptitle(title, fontsize=16) # Removed weight="bold"

            # plt.tight_layout(rect=[0, 0, 1, 0.95] if title else [0, 0, 1, 1])

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
                 

    def inspect_matrix_values(self, matrix, matrix_name="Matrix"):
        """
        Prints summary statistics and checks for invalid values in a NumPy array.

        Parameters:
        - matrix (np.ndarray): The matrix to inspect.
        - matrix_name (str): A name for the matrix used in print statements.
        """
        print(f"--- Inspecting {matrix_name} Values ---")
        if not isinstance(matrix, np.ndarray):
            print(f"Error: Input is not a NumPy array.")
            return
        if matrix.size == 0:
            print(f"Warning: {matrix_name} is empty.")
            return

        try:
            min_val = np.min(matrix)
            max_val = np.max(matrix)
            mean_val = np.mean(matrix)
            mean_abs_val = np.mean(np.abs(matrix))
            std_val = np.std(matrix)

            print(f"{matrix_name} mean: {mean_val:.2e}, std: {std_val:.2e}")
            print(f"{matrix_name} min: {min_val:.2e}, max: {max_val:.2e}")
            # print(f"{matrix_name} std: {std_val:.1e}") # Redundant with first line
            print(f"{matrix_name} mean abs: {mean_abs_val:.2e}")

            nan_check = np.isnan(matrix).any()
            inf_check = np.isinf(matrix).any()

            if nan_check:
                print(f"WARNING: {matrix_name} contains NaN values!")
            if inf_check:
                print(f"WARNING: {matrix_name} contains Inf values!")
            if not nan_check and not inf_check:
                print(f"{matrix_name} contains valid numbers (no NaNs or Infs detected).")

        except Exception as e:
            print(f"Error during inspection of {matrix_name}: {e}")
        print(f"--- End {matrix_name} Inspection ---")

    def load_and_validate_leadfield(self, leadfield_file_path, orientation_type):
        """
        Loads a leadfield matrix from an .npz file and validates its shape
        based on the expected orientation type. Includes value inspection.

        Parameters:
        - leadfield_file_path (str or Path): Path to the .npz file containing the leadfield.
        - orientation_type (str): The expected orientation type ("fixed" or "free").

        Returns:
        - np.ndarray: The loaded and validated leadfield matrix.

        Raises:
        - FileNotFoundError: If the leadfield file does not exist.
        - KeyError: If the expected key is not found in the .npz file.
        - ValueError: If the loaded leadfield matrix shape is inconsistent with the orientation_type.
        - Exception: For other potential loading errors.
        """
        print(f"Loading leadfield from: {leadfield_file_path}")
        try:
            with np.load(leadfield_file_path) as data:
                # ... (loading logic as before) ...
                if 'lead_field' in data:
                    leadfield_matrix = data["lead_field"]
                elif 'lead_field_fixed' in data and orientation_type == "fixed":
                    leadfield_matrix = data['lead_field_fixed']
                elif 'lead_field_free' in data and orientation_type == "free":
                    leadfield_matrix = data['lead_field_free']
                elif 'lead_field' in data:
                    print("Warning: Loading generic 'lead_field' key. Ensure it matches orientation type.")
                    leadfield_matrix = data["lead_field"]
                else:
                    keys_found = list(data.keys())
                    raise KeyError(f"Could not find a suitable leadfield key ('lead_field', 'lead_field_fixed', 'lead_field_free') in .npz file. Found keys: {keys_found}")

            print(f"Leadfield loaded successfully. Initial Shape: {leadfield_matrix.shape}", "dtype:", leadfield_matrix.dtype)

            # --- Validate leadfield shape against orientation_type ---
            # ... (validation logic as before) ...
            if orientation_type == "fixed":
                if leadfield_matrix.ndim != 2:
                    raise ValueError(f"Expected 2D leadfield for fixed orientation, got shape {leadfield_matrix.shape}")
            elif orientation_type == "free":
                if leadfield_matrix.ndim == 3:
                    if leadfield_matrix.shape[2] != 3:
                        raise ValueError(f"Expected 3 components in last dimension for free orientation, got shape {leadfield_matrix.shape}")
                elif leadfield_matrix.ndim == 2:
                    if leadfield_matrix.shape[1] % 3 == 0:
                        print("Warning: Reshaping potentially flattened free orientation leadfield.")
                        n_sensors, n_sources_x_3 = leadfield_matrix.shape
                        n_sources = n_sources_x_3 // 3
                        leadfield_matrix = leadfield_matrix.reshape(n_sensors, n_sources, 3)
                        print(f"Reshaped leadfield to {leadfield_matrix.shape}")
                    else:
                        raise ValueError(f"Cannot reshape 2D leadfield (shape {leadfield_matrix.shape}) to 3D free orientation.")
                else:
                    raise ValueError(f"Expected 2D or 3D leadfield for free orientation, got {leadfield_matrix.ndim} dimensions with shape {leadfield_matrix.shape}")
            else:
                raise ValueError(f"Invalid orientation_type specified: {orientation_type}. Choose 'fixed' or 'free'.")


            print(f"Leadfield validated successfully. Final Shape: {leadfield_matrix.shape}")

            # --- Inspect Leadfield Matrix Values using the function ---
            self.inspect_matrix_values(leadfield_matrix, matrix_name="Leadfield")
            # --- End Inspection ---

            return leadfield_matrix

        except FileNotFoundError:
            print(f"Error: Leadfield file not found at {leadfield_file_path}")
            raise # Re-raise the exception
        except (KeyError, ValueError) as e:
            print(f"Error loading or validating leadfield: {e}")
            raise # Re-raise the specific error
        except Exception as e:
            print(f"An unexpected error occurred during leadfield loading: {e}")
            raise 


    # --- Plotting Functions ---
    def plot_sensor_signals(self, y_clean, y_noisy, sensor_indices=None, times=None, save_dir=None, figure_name=None, trial_idx=None):
        """ Plot clean and noisy sensor signals for specific sensors for a specific trial. """
        if sensor_indices is None:
            sensor_indices = [0]
        if times is None:
            times = np.arange(y_clean.shape[1])

        n_sensors_to_plot = len(sensor_indices)
        fig, axes = plt.subplots(n_sensors_to_plot, 1, figsize=(10, n_sensors_to_plot * 3), sharex=True, sharey=True)
        title_suffix = f" (Trial {trial_idx+1})" if trial_idx is not None else ""
        fig.suptitle(f"Specific Sensor Signals{title_suffix}", fontsize=16)

        if n_sensors_to_plot == 1:
            axes = [axes]

        for i, sensor_idx in enumerate(sensor_indices):
            axes[i].plot(times, y_clean[sensor_idx], label="y_clean", linewidth=2)
            axes[i].plot(times, y_noisy[sensor_idx], label="y_noise")
            axes[i].set_title(f"Sensor {sensor_idx}")
            axes[i].set_xlabel("Time (s)")
            axes[i].set_ylabel("Amplitude  (µV)")
            axes[i].legend()
            axes[i].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_dir and figure_name:
            output_dir = Path(save_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / f"{figure_name}.png"
            plt.savefig(save_path, dpi=300)
            print(f"Sensor subplots figure saved to {save_path}")
        
        plt.close(fig)

    def plot_all_active_sources_single_figure(self, x, times, active_indices, stim_onset, save_dir=None, figure_name=None, trial_idx=None):
        """ Plot all specified active source signals on a single figure for a specific trial. """
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        title_suffix = f" (Trial {trial_idx+1})" if trial_idx is not None else ""
        fig.suptitle(f"All Active Source Signals{title_suffix}", fontsize=16)
        colors = cm.viridis(np.linspace(0, 1, len(active_indices)))

        # Handle potential free orientation source data shape
        if x.ndim == 3:
            x_plot = np.linalg.norm(x, axis=1) # Plot magnitude
        else:
            x_plot = x

        for i, src_idx in enumerate(active_indices):
            ax.plot(times, x_plot[src_idx], label=f"Source {src_idx}", linewidth=1.5, color=colors[i])

        ax.axvline(x=stim_onset, linestyle="--", color="gray", label="Stimulus Onset")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (nAm)")
        ax.legend(loc='best', fontsize='small')
        ax.grid(True, alpha=0.6)
        ax.set_title("Active Sources")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_dir and figure_name:
            output_dir = Path(save_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / f"{figure_name}.png"
            plt.savefig(save_path, dpi=300)
            print(f"Single figure source plot saved to {save_path}")
        plt.close(fig)

    def plot_all_sensor_signals_single_figure(self, y_data, times, sensor_indices=None, save_dir=None, figure_name=None, trial_idx=None, average_epochs=False):
        """
        Plot sensor signals (overlay) for selected sensors.
        If average_epochs is True and y_data is 3D, plots the average across epochs for each channel.
        If average_epochs is False and y_data is 2D, plots the single trial data.
        Does NOT average across channels.

        Parameters:
        - y_data (np.ndarray): Sensor measurements. Can be 2D (n_channels, n_times) for a single trial
                            or 3D (n_trials, n_channels, n_times) for multiple trials.
        - times (np.ndarray): Time vector corresponding to the signals.
        - sensor_indices (list or np.ndarray, optional): Indices of sensors to plot. If None, plots all sensors.
        - save_dir (str or Path, optional): Directory to save the figure.
        - figure_name (str, optional): Name of the figure file (without extension).
        - trial_idx (int, optional): Index of the trial being plotted (used for title if y_data is 2D and average_epochs is False).
        - average_epochs (bool): If True and y_data is 3D, plot the average across trials.
                                If False and y_data is 3D, raises an error.
                                If y_data is 2D, this primarily affects the title.
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        title_suffix = ""
        plot_individual_epochs = False # Flag to control plotting individual trials (currently always False)

        if y_data.ndim == 2: # Input is single trial or already averaged data
            y_plot = y_data # This is the data to plot (n_channels, n_times)
            if not average_epochs and trial_idx is not None:
                title_suffix = f" (Trial {trial_idx+1})"
            elif average_epochs: # Assume 2D input might be an average if flag is set
                title_suffix = " (Average across Trials)"
            # If 2D and not average_epochs and no trial_idx, title is generic
        elif y_data.ndim == 3: # Input is multi-trial data
            if average_epochs:
                y_plot = np.mean(y_data, axis=0) # Calculate average across trials (axis 0) -> shape (n_channels, n_times)
                title_suffix = " (Average across Trials)"
                # Do not plot individual epochs if averaging is requested
                plot_individual_epochs = False
            else:
                # If 3D data is passed but averaging is not requested, it's ambiguous.
                raise ValueError("Input y_data is 3D, but average_epochs is False. "
                                "Provide 2D data (single trial) or set average_epochs=True.")
        else:
            raise ValueError("Input y_data must be 2D or 3D")

        # Select specific sensors if requested from the data to be plotted (y_plot)
        if sensor_indices is None:
            sensor_indices_to_plot = np.arange(y_plot.shape[0]) # Use all channels
            y_plot_selected = y_plot
        else:
            # Ensure indices are valid for the potentially averaged data
            sensor_indices_to_plot = np.array(sensor_indices)[np.array(sensor_indices) < y_plot.shape[0]]
            if len(sensor_indices_to_plot) != len(sensor_indices):
                print("Warning: Some requested sensor_indices are out of bounds for the provided data.")
            y_plot_selected = y_plot[sensor_indices_to_plot, :]

        n_plot_sensors = y_plot_selected.shape[0]

        fig.suptitle(f"Sensor Signals {title_suffix}", fontsize=16)
        colors = cm.turbo(np.linspace(0, 1, n_plot_sensors))

        # --- Plotting Logic ---
        # Plot the main traces (either single trial or trial-averaged)
        for i in range(n_plot_sensors):
            actual_sensor_idx = sensor_indices_to_plot[i] # Get original index
            ax.plot(times, y_plot_selected[i, :], linewidth=1.0, color=colors[i], alpha=0.8, label=f"Ch {actual_sensor_idx}" if n_plot_sensors <= 15 else None)

        # Optional: Plot individual epoch traces lightly in the background (currently disabled)
        if plot_individual_epochs and y_data.ndim == 3:
            y_plot_all_selected = y_data[:, sensor_indices_to_plot, :] # Select sensors from original 3D data
            for i_trial in range(y_data.shape[0]):
                for i_ch in range(n_plot_sensors):
                    ax.plot(times, y_plot_all_selected[i_trial, i_ch, :], linewidth=0.2, color=colors[i_ch], alpha=0.1)


        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (µV)")
        ax.grid(True, alpha=0.6)
        ax.set_title(f"{n_plot_sensors} channels")

        # Update legend
        if n_plot_sensors <= 15: # Show legend only for fewer channels
            ax.legend(loc='best', fontsize='small')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_dir and figure_name:
            output_dir = Path(save_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / f"{figure_name}.png"
            plt.savefig(save_path, dpi=300)
        plt.close(fig)

    def plot_active_sources(self, x, times, active_indices, stim_onset, nnz, save_dir=None, figure_name=None, trial_idx=None):
        """ Plot active sources for a specific trial. """
        n_cols = 3
        n_rows = int(np.ceil(nnz / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), constrained_layout=True, sharex=True, sharey=True)
        title_suffix = f" (Trial {trial_idx+1})" if trial_idx is not None else ""
        fig.suptitle(f"Active Source Signals{title_suffix}", fontsize=16)
        axes = axes.flatten()

        # Handle potential free orientation source data shape (n_sources, n_orient, n_times)
        # Plot the norm or the first component for simplicity
        if x.ndim == 3:
            x_plot = np.linalg.norm(x, axis=1) # Plot magnitude for free orientation
            # Or plot first component: x_plot = x[:, 0, :]
        else:
            x_plot = x

        for i, src_idx in enumerate(active_indices):
            axes[i].plot(times, x_plot[src_idx], label=f"Source {src_idx}", linewidth=2)
            axes[i].axvline(x=stim_onset, linestyle="--", color="gray", label="Stimulus Onset")
            axes[i].set_xlabel("Time (s)")
            axes[i].set_ylabel("Amplitude (nAm)")
            axes[i].set_title(f"Active Source {src_idx}")
            axes[i].legend()
            axes[i].grid(True)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        if save_dir and figure_name:
            output_dir = Path(save_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / f"{figure_name}.png"
            plt.savefig(save_path, dpi=300)
            print(f"Subplots figure saved to {save_path}")
        plt.close(fig)
