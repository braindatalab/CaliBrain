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
from mne.io.constants import FIFF

import matplotlib.pyplot as plt
import matplotlib.cm as cm # Import colormap functionality
from matplotlib.lines import Line2D # Import for custom legend
import matplotlib.gridspec as gridspec
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable # For better colorbar placement

from calibrain.utils import load_config


class SourceSimulator:
    """
    Simulate synthetic source coefficients for three settings:

    1) fixed orientation (MEG and EEG):
         s shape = (N, T)
         One scalar coefficient per source location.

    2) free orientation (EEG):
         x shape = (N, 3, T)
         General 3D coefficient process in the retained local 3D source basis.

    3) free orientation (MEG):
         a shape = (N, 2, T)
         General reduced 2D coefficient process in the MEG-sensitive local subspace.
    """
    def __init__(
        self,
        ERP_config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Parameters
        ----------
        ERP_config : dict, optional
            ERP simulation configuration. If None, defaults are used.

            Required/used keys (with defaults):
            - tmin : float, default -0.5
                Start time (s) of the epoch.
            - tmax : float, default  0.5
                End time (s) of the epoch.
            - stim_onset : float, default 0.0
                Stimulus onset time (s) within [tmin, tmax].
            - sfreq : float, default 250
                Sampling frequency (Hz).
            - fmin, fmax : float, default 1, 5
                Bandpass limits (Hz) for ERP-like waveform shaping.
            - amplitude_distribution : dict
                Log-normal amplitude distribution in nAm:
                * median : float, default 20.0
                * sigma  : float, default 0.2
                * clip   : (low, high), default (2.5, 50.0)
            - random_erp_timing : bool, default True
                If True, randomize ERP segment duration and start offset after stim_onset.
            - erp_min_length : int or None, default None
                Minimum ERP segment length in samples; if None, uses internal default (82).

        logger : logging.Logger, optional
            Logger instance. If None, uses module logger.
        """
        # --- Draft-default config (kept unchanged) ---
        self.ERP_config = ERP_config if ERP_config else {
            "tmin": -0.5,
            "tmax": 0.5,
            "stim_onset": 0.0,
            "sfreq": 250,
            "fmin": 1,
            "fmax": 5,
            "amplitude_distribution": {
                "median": 20.0, # peak dipole moment (nAm)
                "sigma": 0.2, # std of the underlying normal distribution
                "clip": (2.5, 50.0),  # (min, max) bounds in nAm for peak absolute amplitude clipping
            },
            "random_erp_timing": True,
            "erp_min_length": None,
        }

        self.logger = logger if logger else logging.getLogger(__name__)

        # Default units for ERP simulation
        self.kind: int = FIFF.FIFFV_DIPOLE_WAVE # Dipole time curve. Encoded into "1000"
        self.units: str = FIFF.FIFF_UNIT_AM # Amperes (Am)
        self.unitmult: int = FIFF.FIFF_UNITM_N # (Nano = 1e-9)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["logger"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)

    def _get_times(self) -> np.ndarray:
        tmin = self.ERP_config["tmin"]
        tmax = self.ERP_config["tmax"]
        sfreq = self.ERP_config["sfreq"]
        return np.arange(tmin, tmax, 1.0 / sfreq)
    
    # -------------------------
    # Amplitude sampling (nAm)
    # -------------------------
    def _sample_source_amplitude(self, rng: np.random.RandomState) -> float:
        """
        Sample a peak dipole moment (nAm) from a clipped log-normal distribution.

        Math:
        - Draw A ~ LogNormal(mu, sigma)
          with mu = log(median) so that median(A)=median parameter.
        - Clip A into [low, high] if clip bounds provided.
        """
        base_amplitude = 20.0
        dist_cfg = self.ERP_config.get("amplitude_distribution")
        if not dist_cfg:
            return float(max(base_amplitude, 0.0))

        clip_bounds = dist_cfg.get("clip")
        median = float(dist_cfg.get("median", base_amplitude))
        sigma = float(dist_cfg.get("sigma", 0.2))

        # Draft choice (kept): allow very small medians without log(0)
        safe_median = max(median, 1e-6)
        mu = np.log(safe_median)
        amplitude = rng.lognormal(mean=mu, sigma=sigma)

        if clip_bounds is not None:
            low, high = clip_bounds
            amplitude = float(np.clip(amplitude, low, high))

        return float(amplitude)

    # -------------------------
    # ERP waveform simulation
    # -------------------------
    def _simulate_erp_waveform(self, source_seed: int = 512) -> np.ndarray:
        """
        Generate one ERP-like waveform of length n_times.

        Steps (math/ops):
        1) Choose ERP segment length L (>= min length) and start index after stim_onset (optional randomization).
        2) Draw white noise w ~ N(0, I_L).
        3) Bandpass filter: y = filtfilt(BP, w).
        4) Apply Hann window: y <- y * hann(L).
        5) Normalize to unit peak: y <- y / max(|y|).
        6) Scale by sampled amplitude A (nAm): y <- A * y.
        7) Place y into the full epoch waveform x(t), with zeros elsewhere.
        """
        tmin = self.ERP_config["tmin"]
        tmax = self.ERP_config["tmax"]
        stim_onset = self.ERP_config["stim_onset"]
        sfreq = self.ERP_config["sfreq"]
        fmin = self.ERP_config["fmin"]
        fmax = self.ERP_config["fmax"]
        random_erp_timing = self.ERP_config["random_erp_timing"]
        erp_min_length = self.ERP_config["erp_min_length"]

        if stim_onset < tmin or stim_onset > tmax:
            raise ValueError(f"stim_onset ({stim_onset}) is outside [{tmin}, {tmax}]")

        # Draft RNG choice (kept): waveform RNG depends only on source_seed
        rng = np.random.RandomState(int(source_seed))
        
        # Draft default (kept): ensures filter stability + meaningful windowing
        _DEFAULT_MIN_ERP_LEN = 82

        times = self._get_times()
        n_times = len(times)

        # First sample index at/after stimulus onset
        stim_indices = np.where(times >= stim_onset)[0]
        stim_onset_samples = stim_indices[0] if len(stim_indices) > 0 else n_times

        waveform = np.zeros(n_times)
        current_min_erp_len = (
            erp_min_length if erp_min_length is not None else _DEFAULT_MIN_ERP_LEN
        )
        
        # Max available duration after stimulus onset
        max_post = n_times - stim_onset_samples
        if max_post < current_min_erp_len:
            return waveform  # Not enough post-stim samples to form ERP

        # Randomize ERP length and placement *within* post-stim window (draft logic kept)
        if random_erp_timing:
            erp_len = rng.randint(low=current_min_erp_len, high=max_post + 1)
            max_start = max_post - erp_len
            start_offset = rng.randint(0, max_start + 1)
            start_sample = stim_onset_samples + start_offset
        else:
            erp_len = max_post
            start_sample = stim_onset_samples

        if erp_len < current_min_erp_len:
            return waveform

        # White noise segment for the ERP portion
        white = rng.randn(erp_len)

        # Butterworth bandpass design
        low = fmin / (sfreq / 2.0)
        high = fmax / (sfreq / 2.0)

        eps = 1e-9
        low = max(eps, low)
        high = min(1.0 - eps, high)
        if low >= high:
            return waveform

        try:
            b, a = butter(4, [low, high], btype="band")
        except ValueError:
            return waveform

        # Zero-phase filtering + windowing
        seg = filtfilt(b, a, white)
        seg *= np.hanning(erp_len)

        # Normalize by peak so amplitude corresponds to physical peak dipole moment
        peak = float(np.max(np.abs(seg)))
        if peak < 1e-9:
            return waveform
        seg /= peak

        # Scale by sampled dipole-moment amplitude (nAm)
        seg *= self._sample_source_amplitude(rng)

        # Place ERP segment into the full waveform
        end_sample = start_sample + seg.size
        if start_sample < n_times and end_sample <= n_times:
            waveform[start_sample:end_sample] = seg

        return waveform
    
    def _simulate_fixed(
        self,
        n_sources: int,
        nnz: int,
        trial_seed: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if nnz > n_sources:
            raise ValueError(f"nnz ({nnz}) must be <= n_sources ({n_sources})")

        trial_rng = np.random.RandomState(int(trial_seed))
        seed_high = np.iinfo(np.int32).max
        n_times = len(self._get_times())

        active_indices = trial_rng.choice(n_sources, size=nnz, replace=False)
        x = np.zeros((n_sources, n_times))

        for src_idx in active_indices:
            source_seed = int(trial_rng.randint(0, seed_high))
            x[src_idx, :] = self._simulate_erp_waveform(source_seed=source_seed)

        return x, active_indices

    def _simulate_multicomponent(
        self,
        n_sources: int,
        nnz: int,
        n_comp: int,
        trial_seed: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if nnz > n_sources:
            raise ValueError(f"nnz ({nnz}) must be <= n_sources ({n_sources})")

        trial_rng = np.random.RandomState(int(trial_seed))
        seed_high = np.iinfo(np.int32).max
        n_times = len(self._get_times())

        active_indices = trial_rng.choice(n_sources, size=nnz, replace=False)
        x = np.zeros((n_sources, n_comp, n_times))

        # deterministic component offsets
        prime_offset = 104729

        for src_idx in active_indices:
            base_seed = int(trial_rng.randint(0, seed_high))
            for k in range(n_comp):
                comp_seed = int((base_seed + k * prime_offset) % seed_high)
                x[src_idx, k, :] = self._simulate_erp_waveform(source_seed=comp_seed)

        return x, active_indices

    # -----------------------------
    # Source time course simulation
    # -----------------------------
    
    def simulate(
        self,
        n_sources: int = 1284,
        nnz: int = 5,
        orientation_type: str = "fixed",
        coil_type: str = FIFF.FIFFV_COIL_EEG, # 1 (eeg) default coil type for simulation
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:

        if orientation_type == "fixed":
            return self._simulate_fixed(
                n_sources=n_sources,
                nnz=nnz,
                trial_seed=int(seed),
            )

        elif orientation_type == "free" and coil_type == FIFF.FIFFV_COIL_EEG: # 1 (eeg)
            # general 3D coefficient process in the retained local 3D basis
            return self._simulate_multicomponent(
                n_sources=n_sources,
                nnz=nnz,
                n_comp=3,
                trial_seed=int(seed),
            )

        elif orientation_type == "free" and coil_type in [FIFF.FIFFV_COIL_VV_MAG_T1, FIFF.FIFFV_COIL_VV_PLANAR_T1]: # 3022 (mag), 3012 (grad)
            # general reduced 2D coefficient process in the MEG-sensitive local subspace
            return self._simulate_multicomponent(
                n_sources=n_sources,
                nnz=nnz,
                n_comp=2,
                trial_seed=int(seed),
            )

        raise ValueError("orientation_type must be 'fixed' or 'free' with appropriate coil_type for MEG/EEG")
