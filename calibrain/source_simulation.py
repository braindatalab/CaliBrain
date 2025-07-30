"""
source_simulation.py
Module for simulating synthetic brain activity data for source-level measurements.

Specifically simulating event-related potential (ERP)-like signals for use in 
neuroimaging research (e.g., MEG/EEG source simulation). It supports flexible 
configuration of ERP waveform properties, source orientation (fixed or free), 
and trial-based simulation with reproducible randomization.
"""

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

from calibrain import LeadfieldBuilder
from calibrain.utils import load_config


class SourceSimulator:
    """Simulates synthetic brain activity data for source-level measurements.
    """
    def __init__(
        self,
        ERP_config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the SourceSimulator with parameters for simulating dipole sources.
        
        Parameters
        ----------
        ERP_config : Optional[Dict[str, Any]]
            Configuration dictionary for the ERP simulation parameters. If None, default values are used.
            Default values include:
                - tmin: -0.5 (start time of the ERP segment in seconds)
                - tmax: 0.5 (end time of the ERP segment in seconds)
                - stim_onset: 0.0 (time of stimulus onset in seconds, relative to the start of the ERP segment)
                - sfreq: 250 (sampling frequency in Hz)
                - fmin: 1 (minimum frequency for the bandpass filter in Hz)
                - fmax: 5 (maximum frequency for the bandpass filter in Hz)
                - amplitude: 5.0 (amplitude of the ERP waveform)
                - random_erp_timing: True (if True, the exact start time and duration of the ERP waveform within the post-stimulus window are randomized)
                - erp_min_length : Optional[int] (minimum length of the ERP waveform in samples; if None, a default value is used)
                - units : str (units of the source activity, e.g., "nAm")
        logger : Optional[logging.Logger], optional
            Logger instance, by default None.
        """
        self.ERP_config = ERP_config if ERP_config else {
            "tmin": -0.5,
            "tmax": 0.5,
            "stim_onset": 0.0,
            "sfreq": 250,
            "fmin": 1,
            "fmax": 5,
            "amplitude": 5.0,
            "random_erp_timing": True,
            "erp_min_length": None,
            "units": "nAm",
        }
        
        self.logger = logger if logger else logging.getLogger(__name__)

    def _simulate_erp_waveform(
        self,
        source_seed: int = 512,
    ) -> np.ndarray:
        """
        Generate a smoothed ERP-like waveform for a single source.

        This method creates an ERP-like signal segment using bandpass-filtered white noise, applies a Hanning window, normalizes and scales by the specified amplitude, and places the segment at a randomized or fixed position after the stimulus onset within the time course.
        
        Parameters
        ----------
        source_seed : int
            Seed for the random number generator to ensure reproducibility of the ERP waveform generation. Default is 512.
        
        Returns
        -------
        np.ndarray
            The generated ERP signal of length n_times.
        
        Notes
        -----
            - The output signal is zero-padded before the `stim_onset` to ensure it starts with zeros, simulating the pre-stimulus baseline.
            - If `random_erp_timing` is True, the exact start time
            (offset from `stim_onset`) and duration of the ERP waveform within the
            post-`stim_onset` window are randomized. The ERP will still be contained
            entirely within the `stim_onset` to `n_times` interval.
            - If `random_erp_timing` is False, the ERP waveform
            spans the entire duration from `stim_onset` to `n_times`.
        """
        # Extract ERP configuration parameters
        tmin = self.ERP_config['tmin']
        tmax = self.ERP_config['tmax']
        stim_onset = self.ERP_config['stim_onset']
        sfreq = self.ERP_config['sfreq']
        fmin = self.ERP_config['fmin']
        fmax = self.ERP_config['fmax']
        amplitude = self.ERP_config['amplitude']
        random_erp_timing = self.ERP_config['random_erp_timing']
        erp_min_length = self.ERP_config['erp_min_length']
        units = self.ERP_config['units']

        # Ensure stim_onset is within [tmin, tmax]
        if stim_onset < tmin or stim_onset > tmax:
            raise ValueError(f"stim_onset ({stim_onset}) is outside the time range [{tmin}, {tmax}]")
        
        source_duration_rng = np.random.RandomState(source_seed)
        
        # For filter stability (filtfilt butter order 4) & meaningful Hanning window
        _DEFAULT_MIN_ERP_LEN = 82

        times = np.arange(tmin, tmax, 1.0 / sfreq)
        n_times = len(times)

        # Determine the index for stimulus onset
        stim_indices = np.where(times >= stim_onset)[0]
        if len(stim_indices) == 0:
            # Stimulus onset is at or after tmax, effectively no ERP in this epoch
            stim_onset_samples = n_times
        else:
            stim_onset_samples = stim_indices[0]

        waveform = np.zeros(n_times)
        current_min_erp_len = erp_min_length if erp_min_length is not None else _DEFAULT_MIN_ERP_LEN

        # Maximum available duration for ERP activity after stim_onset_samples
        max_available_post_stim_duration = n_times - stim_onset_samples

        if max_available_post_stim_duration < current_min_erp_len:
            # Not enough samples in the post-stimulus window for a meaningful ERP
            return waveform

        actual_placement_start_sample: int
        erp_duration_samples: int

        if random_erp_timing:
            # Randomize ERP duration: from current_min_erp_len up to max_available_post_stim_duration (inclusive)
            actual_erp_duration = source_duration_rng.randint(low=current_min_erp_len, high=max_available_post_stim_duration + 1)
            self.logger.debug(f"Randomized ERP duration: {actual_erp_duration} samples")
            
            # Randomize ERP start offset within the available post-stimulus window
            # Max possible start offset (from stim_onset_samples) for the chosen actual_erp_duration
            max_start_offset_from_onset = max_available_post_stim_duration - actual_erp_duration
            start_offset_from_onset = source_duration_rng.randint(0, max_start_offset_from_onset + 1)
            
            self.logger.debug(f"Randomized ERP start offset from onset: {start_offset_from_onset} samples")
                
            actual_placement_start_sample = stim_onset_samples + start_offset_from_onset
            erp_duration_samples = actual_erp_duration
        else:
            # ERP spans the entire available post-stimulus duration
            erp_duration_samples = max_available_post_stim_duration
            actual_placement_start_sample = stim_onset_samples

        # Safeguard, though preceding logic should ensure this
        if erp_duration_samples < current_min_erp_len:
            return waveform

        # Generate noise only for the determined duration of the ERP activity
        white_noise_for_erp = source_duration_rng.randn(erp_duration_samples)
        self.logger.debug(f"Generated white noise for ERP with {erp_duration_samples} samples.")
        
        # Design a Butterworth bandpass filter
        low = fmin / (sfreq / 2)
        high = fmax / (sfreq / 2)

        epsilon = 1e-9
        low = max(epsilon, low)
        high = min(1.0 - epsilon, high)
        if low >= high:
            return waveform # Invalid frequency band

        try:
            b, a = butter(4, [low, high], btype='band')
        except ValueError as e:
            return waveform # Filter design failed

        # Filter the noise segment
        erp_segment = filtfilt(b, a, white_noise_for_erp)
        
        # Apply Hanning window over the ERP segment
        erp_segment *= np.hanning(erp_duration_samples) 
        
        std_erp_segment = np.std(erp_segment)
        if std_erp_segment < 1e-9: # Check if standard deviation is effectively zero
            return waveform # Avoid division by zero; segment is flat
            
        erp_segment /= std_erp_segment # Normalize
        erp_segment *= amplitude      # Scale
        
        # Place the generated ERP segment into the output signal at the determined start
        end_sample_for_erp_segment = actual_placement_start_sample + len(erp_segment)
        
        # Ensure placement is within bounds (should be guaranteed by earlier logic)
        if actual_placement_start_sample < n_times and end_sample_for_erp_segment <= n_times:
            waveform[actual_placement_start_sample : end_sample_for_erp_segment] = erp_segment    

        self.logger.debug(f"ERP waveform generated with shape: {waveform.shape}")
        
        return waveform

    def _simulate_source_time_courses(
        self,
        orientation_type: str = "fixed",
        n_sources: int = 100,
        nnz: int = 5,
        trial_seed: int = 256,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate simulated source time courses for a single trial.

        This method creates ERP-like signals for a subset of active sources, determined by `nnz`. For each active source, an ERP waveform is generated using a bandpass-filtered noise segment, optionally randomized in onset and duration, and scaled by the specified amplitude. The ERP waveform is placed at the appropriate time index based on `stim_onset`.

        Parameters
        ----------
        orientation_type : str
            Orientation of the sources, either "fixed" or "free". Default is "fixed".
        n_sources : int
            Total number of sources to simulate. Default is 100.
        nnz : int
            Number of non-zero (active) sources in the trial. Must be less than or equal to `n_sources`. Default is 5.
        trial_seed : int
            Seed for the random number generator to ensure reproducibility of the source activity. Default is 256.

        Returns
        -------
        x : np.ndarray
            Simulated source activity array.
            - Shape (n_sources, n_times) for "fixed" orientation.
            - Shape (n_sources, n_orient, n_times) for "free" orientation.
        active_indices : np.ndarray
            Indices of the sources that were activated in this trial.
            
        Notes
        -------
        - For "fixed" orientation, each active source has a single time course.
        - For "free" orientation, each active source has three orientation components, with random orientation coefficients.
        """
        trial_rng = np.random.RandomState(trial_seed)
        
        tmin = self.ERP_config['tmin']
        tmax = self.ERP_config['tmax']
        sfreq = self.ERP_config['sfreq']

        times = np.arange(tmin, tmax, 1.0 / sfreq)
        n_times = len(times)
        
        if orientation_type == "fixed":
            # active_indices = np.sort(rng.choice(self.n_sources, size=self.nnz, replace=False))
            active_indices = trial_rng.choice(n_sources, size=nnz, replace=False)
            x = np.zeros((n_sources, n_times))
            for i, src_idx in enumerate(active_indices):
                # Generate ERP signal with specified onset
                source_seed = trial_rng.randint(low=0, high=2**32 -1) # Derive a new seed for this source
                
                self.logger.debug(f"Generating ERP for source index {src_idx} with seed {source_seed}")
                        
                erp_waveform = self._simulate_erp_waveform(source_seed=source_seed)
                x[src_idx, :] = erp_waveform # Assign the full waveform (includes leading zeros)
                
        elif orientation_type == "free":
            # TODO: +++ THIS IS A TEMPORARY FIX. A NEW APPROACH IS NEEDED TO HANDLE +++
            n_orient = 3 # TODO: Make this configurable
            # active_indices = np.sort(rng.choice(self.n_sources, size=self.nnz, replace=False))
            active_indices = trial_rng.choice(n_sources, size=nnz, replace=False)
            x = np.zeros((n_sources, n_orient, n_times))
            for i, src_idx in enumerate(active_indices):
                source_seed = trial_rng.randint(0, 2**32 -1)
                erp_waveform = self._simulate_erp_waveform(
                    source_seed,
                )
                orient_coeffs = trial_rng.randn(n_orient)
                norm_orient = np.linalg.norm(orient_coeffs)
                if norm_orient < 1e-9: # Avoid division by zero
                    orient_coeffs = np.array([1.0, 0.0, 0.0]) # Default orientation
                else:
                    orient_coeffs /= norm_orient

                for j_orient in range(n_orient):
                    x[src_idx, j_orient, :] = orient_coeffs[j_orient] * erp_waveform

                # Alternatively, if we want to assign the same waveform to all orientations
                # for j_orient in range(n_orientations_free):
                #     x[src_idx, j_orient, :] = erp_waveform
        else:
            raise ValueError("Invalid orientation_type. Choose 'fixed' or 'free'.")

        self.logger.debug(f"Simulated source time courses with shape: {x.shape}")
        self.logger.debug(f"Active source indices: {active_indices}")
        
        return x, active_indices

    def simulate(
        self,
        orientation_type: str = "fixed",
        n_sources: int = 100,
        nnz: int = 5,
        n_trials: int = 1,
        global_seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate multiple trials of source time courses.

        This function generates synthetic source activity for `n_trials` trials using 
        ERP-like signals. Each trial uses a unique random seed derived from the 
        provided `global_seed` for reproducibility.

        Parameters
        ----------
        orientation_type : str
            Orientation of the sources, either "fixed" or "free". Default is "fixed".
        n_sources : int
            Total number of sources to simulate. Default is 100.
        nnz : int
            Number of non-zero (active) sources in each trial. Must be less than or equal to `n_sources`. Default is 5.
        n_trials : int
            Number of trials to simulate. Default is 1.
        global_seed : int
            Seed for the random number generator to ensure reproducibility across trials. Default is 42.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - x_all_trials : np.ndarray
                Array of shape (n_trials, ...) containing simulated source time courses.
                Shape depends on source orientation: 
                    - fixed: (n_trials, n_sources, n_times)
                    - free:  (n_trials, n_sources, 3, n_times)
            - active_indices_all_trials : np.ndarray
                Array of shape (n_trials, nnz) containing indices of active sources per trial.
        """
        source_rng = np.random.RandomState(global_seed)
        source_seeds = source_rng.randint(0, 2**32 - 1, size=n_trials)

        x_all_trials = []
        active_indices_all_trials = []

        for i, seed in enumerate(source_seeds):
            self.logger.debug(f"Simulating trial {i + 1}/{n_trials} with seed {seed}")
            x, active_indices = self._simulate_source_time_courses(
                orientation_type=orientation_type,
                n_sources=n_sources,
                nnz=nnz,
                trial_seed=seed,
            )
            x_all_trials.append(x)
            active_indices_all_trials.append(active_indices)

        # Convert lists to numpy arrays
        x_all_trials = np.array(x_all_trials)
        active_indices_all_trials = np.array(active_indices_all_trials)
        
        # Log the shapes of the results
        self.logger.info(f"Completed simulating source time courses for {n_trials} trials.")
        self.logger.info(f"Shape of source time courses of all trials {n_trials} trials: {x_all_trials.shape}")
        self.logger.info(f"Shape of active indices for all {n_trials} trials: {active_indices_all_trials.shape}")

        # Print active indices for all trials, each trial on a new line
        self.logger.info("Active indices for all trials:")
        for i, indices in enumerate(active_indices_all_trials):
            self.logger.info(f"  Trial {i+1}: {indices}")

        return x_all_trials, active_indices_all_trials



def main():

    logging.basicConfig(
        level=logging.INFO,  # or DEBUG
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),  # Console
            logging.FileHandler("simulation.log")  # Optional: file output
        ]
    )
    logger = logging.getLogger("SourceSimulator")

    ERP_config = {
        "tmin": -0.5,
        "tmax": 0.5,
        "stim_onset": 0.0,
        "sfreq": 250,
        "fmin": 1,
        "fmax": 5,
        "amplitude": 5.0,
        "random_erp_timing": True,
        "erp_min_length": None,
        "units": "nAm",
    }

    n_trials=4
    orientation_type="fixed"
    n_sources=10
    nnz=5
    global_seed=42
    
    source_simulator = SourceSimulator(
        ERP_config=ERP_config,
        logger=logger
    )

    # waveform = source_sim._simulate_erp_waveform(**ERP_config, source_seed=512)
    
    # x, active_indices = source_sim._simulate_source_time_courses(
    #     orientation_type=orientation_type,
    #     n_sources=n_sources,
    #     nnz=nnz,
    #     trial_seed=256,
    #     ERP_config=ERP_config,
    # )

    # x_trials, active_indices_trials = source_sim.simulate(
    #     orientation_type=orientation_type,
    #     n_sources=n_sources,
    #     nnz=nnz,
    #     global_seed=global_seed,
    #     ERP_config=ERP_config,
    # )


    x_trials, active_indices_trials = source_simulator.simulate(
        orientation_type=orientation_type,
        n_sources=n_sources,
        nnz=nnz,
        n_trials=n_trials,
        global_seed=global_seed,
    )


    logger.info("Simulation complete.")

if __name__ == "__main__":
    main()