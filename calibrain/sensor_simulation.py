"""
sensor_simulation.py
Module for simulating synthetic brain activity data for sensor-level measurements.
"""

import os
from pathlib import Path
import logging
from pdb import main
from typing import Optional, Tuple, Union, Dict, List, Any

import numpy as np
from numpy.random import Generator
from scipy.stats import wishart
from scipy.signal import butter, filtfilt
import mne
from mne.io.constants import FIFF

import matplotlib.pyplot as plt
import matplotlib.cm as cm # Import colormap functionality
from matplotlib.lines import Line2D # Import for custom legend
import matplotlib.gridspec as gridspec
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable # For better colorbar placement

from calibrain import LeadfieldBuilder
from calibrain.utils import load_config

class SensorSimulator:
    """Simulates synthetic brain activity data for sensor-level measurements.
    """
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the SensorSimulator.
        Parameters
        ----------
        logger : Optional[logging.Logger]
            Logger instance for logging messages. If None, a default logger will be created.
        """
        self.logger = logger if logger else logging.getLogger(__name__)
        
        #  Prameters that will be set during simulation
        self.sensor_units = FIFF.FIFF_UNIT_T  # Default unit for MEG (mag) sensors (T)
                
    def _project_sources_to_sensors(self, x: np.ndarray, L: np.ndarray, orientation_type: str) -> np.ndarray:
        """
        Project the source activity to the sensor space using the leadfield matrix.

        Parameters
        ----------
        x : np.ndarray (nAm)
            Source activity.
            - 'fixed': Shape (n_sources, n_times).
            - 'free': Shape (n_sources, 3, n_times).
        L : np.ndarray
            Leadfield matrix (µV / nAm for EEG or ft / nAm for MEG).
            - 'fixed': Shape (n_sensors, n_sources).
            - 'free': Shape (n_sensors, n_sources, 3).            
        orientation_type : str
            Orientation type of the sources ('fixed' or 'free').

        Returns
        -------
        np.ndarray
            Sensor measurements (y_clean). Shape: (n_sensors, n_times). The units depends on the channel type:
                - 'eeg': µV
                - 'meg': fT
        """
        if orientation_type == "fixed":
            # Matrix multiplication: (n_sensors, n_sources) @ (n_sources, n_times) -> (n_sensors, n_times)
            y = L @ x
        elif orientation_type == "free":
            # Einstein summation: Sum over source index 'm' and orientation index 'r'
            # (n_sensors, n_sources, 3) einsum (n_sources, 3, n_times) -> (n_sensors, n_times)
            y = np.einsum("nmr,mrt->nt", L, x) # Corrected einsum indices
        else:
            raise ValueError(f"Unsupported orientation type: {orientation_type}")

        return y

    def _add_noise(self, y_clean, alpha_SNR, noise_seed):
        """
        Adds Homoscedastic (uniform variance across channels) and uncorrelated (white) Gaussian noise to a clean signal based on a desired SNR level.
        
        Parameters:
        y_clean : np.ndarray
            The clean signal array (e.g., channels x times).
        alpha_SNR : float
            Desired signal-to-noise ratio between 0 and 1.
            - 0.0 means full noise, no signal.
            - 1.0 means no noise, only signal.
        noise_seed : int
            Seed for the random number generator to ensure reproducibility.

        Returns:
        - tuple[np.ndarray, np.ndarray, float]
            - y_noisy : np.ndarray
                The noisy signal with added Gaussian noise.
            - noise : np.ndarray
                The noise added to the clean signal.
            - noise_var : float
                The variance of the noise added to the clean signal.
        """
        if not (0.0 <= alpha_SNR <= 1.0):
            raise ValueError("alpha_SNR must be in [0, 1].")

        noise_rng = np.random.RandomState(noise_seed)
        
        # Generate base noise with variance = noise_std^2
        noise_std = 0.1 # 0.01 # Standard deviation of the noise

        if alpha_SNR == 1.0:
            # No noise added
            eps = np.zeros_like(y_clean)
            return y_clean.copy(), eps, 0.0

        # Sample noise from a Gaussian distribution
        eps = noise_rng.normal(loc=0.0, scale=noise_std, size=y_clean.shape)
        
        # Compute frobenius norms
        signal_norm = np.linalg.norm(y_clean, ord='fro')
        eps_norm = np.linalg.norm(eps, ord='fro')
        
        if alpha_SNR == 0.0:
            # Avoid division by zero, treat as full noise
            # Pure noise with same norm as clean signal
            eta = signal_norm / eps_norm
            eps_scaled = eta * eps
            return eps_scaled.copy(), eps_scaled, (eta * noise_std) ** 2
            
        # General case: scaled noise added to clean signal
        # Scale the noise to achieve the desired SNR
        eta = ((1 - alpha_SNR) / alpha_SNR) * (signal_norm / eps_norm)
        eps_scaled = eta * eps # Scaled noise
        y_noisy = y_clean + eps_scaled # Add noise to signal
        noise_var = (eta * noise_std) ** 2
    
        # OLD APPROACH:
        # signal_power = np.mean(y_clean ** 2)
        
        # if signal_power == 0:
        #     print("Warning: Clean signal power is zero. Cannot add noise based on SNR.")
        #     noise_power = 0
        #     noise = np.zeros_like(y_clean)
        # else:
        #     snr_linear = 10 ** (self.alpha_SNR / 10.0)
        #     # Homoscedastic case: The standard deviation and thus the variance (noise_power) is the same for all sensors and all time points.
        #     noise_power = signal_power / snr_linear # Variance of the noise
        #     noise_std = np.sqrt(noise_power)
            
        #     # Draw noise from Gaussian distribution (independenet noise at each sensor and at each time point). -> The noise covariance matrix is diagonal. 
        #     noise = noise_rng.normal(0, noise_std, size=y_clean.shape) # White noise (uncorrelated across sensors and time) with a uniform power across sensors. shape: n_sensors x n_times. 
        #     self.logger.info(f"Noise: {noise.shape}, noise power: {noise_power:.4f}, noise std: {noise_std:.4f}")

        # y_noisy = y_clean + noise
        # return y_noisy, noise, noise_power
        
        return y_noisy, eps_scaled, noise_var

    def simulate(
        self,
        x_trials: List[np.ndarray],
        L: np.ndarray,
        orientation_type: str = "fixed",
        alpha_SNR: float = 0.5,
        n_trials: int = 1,
        global_seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate sensor trials by projecting source data to sensor space and adding noise to the clean sensor data.
        
        Parameters
        ----------
        x_trials : List[np.ndarray]
            List of source time courses for each trial. Each element should be an array of shape (n_sources, n_times) for fixed orientation or (n_sources, 3, n_times) for free orientation.
        L : np.ndarray
            Leadfield matrix (µV / nAm for EEG or fT / nAm for MEG).
            - 'fixed': Shape (n_sensors, n_sources).
            - 'free': Shape (n_sensors, 3, n_sources).
        orientation_type : str
            Orientation type of the sources ('fixed' or 'free').
        alpha_SNR : float
            Desired signal-to-noise ratio between 0 and 1.
            - 0.0 means full noise, no signal.
            - 1.0 means no noise, only signal.
        n_trials : int
            Number of trials to simulate. Default is 1.
        global_seed : int
            Global seed for random number generation to ensure reproducibility across trials.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            - y_clean_all_trials : np.ndarray
                Clean sensor data for all trials. Shape: (n_trials, n_sensors, n_times).
            - y_noisy_all_trials : np.ndarray
                Noisy sensor data for all trials. Shape: (n_trials, n_sensors, n_times).
            - noise_all_trials : np.ndarray
                Noise added to the clean sensor data for all trials. Shape: (n_trials, n_sensors, n_times).
            - noise_var_all_trials : np.ndarray
                Noise variance for each trial. Shape: (n_trials,).
        """
        noise_rng = np.random.RandomState(global_seed + 123456)
        noise_seeds = noise_rng.randint(0, 2**32 - 1, size=n_trials)

        y_clean_all_trials, y_noisy_all_trials = [], []
        noise_all_trials, noise_var_all_trials = [], []
        
        for i in range(n_trials):
            x_trial = x_trials[i]  # Source time courses for this trial
            noise_seed = noise_seeds[i]

            self.logger.debug(f"[Trial {i+1}/{n_trials}] Adding noise with seed {noise_seed}")

            # Generate source time courses for this trial
            y_clean = self._project_sources_to_sensors(x=x_trial, L=L, orientation_type='fixed')

            # Add noise to the clean sensor data for this trial
            y_noisy, noise, noise_var = self._add_noise(y_clean, alpha_SNR, noise_seed)

            y_clean_all_trials.append(y_clean)
            y_noisy_all_trials.append(y_noisy)
            noise_all_trials.append(noise)
            noise_var_all_trials.append(noise_var)
            
        y_clean_all_trials = np.array(y_clean_all_trials) # (n_trials, n_channels, n_times)
        x_trials = np.array(x_trials) # (n_trials, n_sources, [n_orient,] n_times)
        y_noisy_all_trials = np.array(y_noisy_all_trials) # (n_trials, n_channels, n_times)
        noise_all_trials = np.array(noise_all_trials) # (n_trials, n_channels, n_times)
        noise_var_all_trials = np.array(noise_var_all_trials) # (n_trials,)

        self.logger.info(f"Noise addition complete.")
        self.logger.info(f"Shape of clean sensor data for all trials: {y_clean_all_trials.shape}")
        self.logger.info(f"Shape of noisy sensor data for all trials: {y_noisy_all_trials.shape}")
        self.logger.info(f"Shape of noise data for all trials: {noise_all_trials.shape}")
        self.logger.info(f"Shape of noise variance data for all trials: {noise_var_all_trials.shape}")

        # Reshape leadfield matrix for free orientation if needed by downstream estimators.
        if orientation_type == "free":
            self.logger.info("Reshaping free orientation leadfield from (sensors, sources, 3) to (sensors, sources*3)")
            L = L.reshape(L.shape[0], -1)
            
        return y_clean_all_trials, y_noisy_all_trials, noise_all_trials, noise_var_all_trials

