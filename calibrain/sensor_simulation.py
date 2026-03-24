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


# =============================================================================
# Sensor simulator
# =============================================================================
class SensorSimulator:
    """
    Simulates synthetic sensor-level measurements from source-level activity.

    This class performs two steps:
      (1) Forward projection (clean signal): y_clean = L * x
      (2) Additive Gaussian white noise with controlled SNR level:
          y_noisy = y_clean + eta * eps, where eps ~ N(0, sigma^2 I)

    Supported orientation conventions
    -------------------------------
    Fixed orientation:
      - x: (n_sources, n_times)              [nAm]
      - L: (n_sensors, n_sources)            [sensor_unit / nAm]
      - y: (n_sensors, n_times)              [sensor_unit]

    Free orientation:
      - x: (n_sources, K, n_times)           [nAm]  (3 components of dipole moment)
      - L: (n_sensors, n_sources, K)         [sensor_unit / nAm]
      - y: (n_sensors, n_times)              [sensor_unit]
      with K = 2 for MEG (tangential components) and K = 3 for EEG (full 3D).

    Units
    -----
    By default, metadata assumes MEG magnetometers:
      - kind    = FIFFV_MEG_CH
      - units   = FIFF_UNIT_T
      - unitmult= FIFF_UNITM_F  (femto; 1e-15)
    These are metadata fields; numerical outputs depend on your L units.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Parameters
        ----------
        logger : logging.Logger, optional
            Logger instance. If None, uses module logger.
        """
        self.logger = logger if logger else logging.getLogger(__name__)

        # Default metadata (MEG magnetometers unless overwritten elsewhere)
        self.kind: int = FIFF.FIFFV_MEG_CH
        self.units: int = FIFF.FIFF_UNIT_T
        self.unitmult: int = FIFF.FIFF_UNITM_F
        self.coil_type: Optional[int] = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["logger"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)

    def set_sensor_metadata(
        self,
        *,
        kind: Optional[int] = None,
        units: Optional[int] = None,
        unitmult: Optional[int] = None,
        coil_type: Optional[int] = None,
    ) -> None:
        """Update sensor metadata (FIFF kind/unit/multiplier) atomically."""
        if kind is not None:
            self.kind = kind # 1 for FIFFV_MEG_CH, 2 for FIFFV_EEG_CH
        if units is not None:
            self.units = units
        if unitmult is not None:
            self.unitmult = unitmult
        if coil_type is not None:
            self.coil_type = coil_type
    
    def _project_sources_to_sensors(self, x: np.ndarray, L: np.ndarray) -> np.ndarray:
        """
        Project source activity into sensor space using the leadfield.

        Parameters
        ----------
        x : np.ndarray
            Source activity.
            - fixed: (n_sources, n_times)
            - free : (n_sources, K, n_times)
            with K = 2 for MEG (tangential) and K = 3 for EEG (full 3D).
            Typically measured in nAm for dipole moments.

        L : np.ndarray
            Leadfield mapping sources to sensors.
            - fixed: (n_sensors, n_sources)
            - free : (n_sensors, n_sources, K)
            with K = 2 for MEG (tangential) and K = 3 for EEG (full 3D).

            Units depend on modality and how L is constructed, e.g.
            - MEG: fT / nAm (or T / nAm with metadata scaling)
            - EEG: µV / nAm

        Returns
        -------
        y_clean : np.ndarray
            Clean sensor signal (noiseless), shape (n_sensors, n_times).

        Notes (math)
        -----------
        Fixed orientation:
            y(t) = L x(t)

        Free orientation:
            y_n(t) = sum_m sum_r L_{n,m,r} * x_{m,r}(t)
            Implemented as: einsum("nmr,mrt->nt", L, x)
        """
        # fixed
        if L.ndim == 2 and x.ndim == 2:
            if L.shape[1] != x.shape[0]:
                raise ValueError(
                    f"Dimension mismatch: L has n_sources={L.shape[1]}, x has n_sources={x.shape[0]}"
                )
            return L @ x

        # generic reduced/free (K=2 or K=3)
        if L.ndim == 3 and x.ndim == 3:
            if L.shape[1] != x.shape[0]:
                raise ValueError(
                    f"Dimension mismatch: L has n_sources={L.shape[1]}, x has n_sources={x.shape[0]}"
                )
            if L.shape[2] != x.shape[1]:
                raise ValueError(
                    f"Reduced dimension mismatch: L has K={L.shape[2]}, x has K={x.shape[1]}"
                )
            return np.einsum("mnk,nkt->mt", L, x)

        raise ValueError(f"Incompatible shapes for projection: L{L.shape}, x{x.shape}")

    def _add_noise(
        self,
        y_clean: np.ndarray,
        alpha_SNR: float = 0.5,
        sensor_white_noise_std: float = 1.0,
        noise_seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Add homoscedastic (same variance for all sensors) white Gaussian noise
        with a controlled SNR-like mixing parameter alpha_SNR.

        Noise model
        ----------
        eps ~ N(0, sigma^2 I) with sigma = sensor_white_noise_std
        y_noisy = y_clean + eta * eps

        Scaling eta is chosen using Frobenius norms:
            ||y_clean||_F and ||eps||_F

        Interpretation of alpha_SNR
        ---------------------------
        alpha_SNR in [0, 1]:
          - alpha_SNR = 1.0  => y_noisy = y_clean  (no noise)
          - alpha_SNR = 0.0  => y_noisy = eta*eps  (pure noise scaled to match signal norm)
          - otherwise:
              eta = ((1 - alpha_SNR) / alpha_SNR) * (||y_clean||_F / ||eps||_F)

        Parameters
        ----------
        y_clean : np.ndarray
            Clean sensor signal, shape (n_sensors, n_times).
        alpha_SNR : float
            Mixing parameter in [0, 1].
        sensor_white_noise_std : float
            Standard deviation sigma of base Gaussian noise eps.
        noise_seed : int
            Seed for reproducibility of eps.

        Returns
        -------
        y_noisy : np.ndarray
            Noisy sensor signal, shape (n_sensors, n_times).
        eps_scaled : np.ndarray
            Added noise term (eta * eps), shape (n_sensors, n_times).
        eta : float
            Noise scaling factor.
        """
        if not (0.0 <= alpha_SNR <= 1.0):
            raise ValueError("alpha_SNR must be in [0, 1].")

        noise_rng = np.random.RandomState(int(noise_seed))

        if alpha_SNR == 1.0:
            eps = np.zeros_like(y_clean)
            return y_clean.copy(), eps, 0.0

        # Base white Gaussian noise
        eps = noise_rng.normal(
            loc=0.0,
            scale=float(sensor_white_noise_std),
            size=y_clean.shape,
        )
        
        # Frobenius norms
        signal_norm = np.linalg.norm(y_clean, ord="fro")
        eps_norm = np.linalg.norm(eps, ord="fro")

        # Guard against degenerate eps (e.g., std=0)
        if eps_norm < 1e-12:
            return y_clean.copy(), np.zeros_like(y_clean), 0.0

        if alpha_SNR == 0.0:
            # Pure noise scaled to have the same Frobenius norm as the signal
            eta = signal_norm / eps_norm
            eps_scaled = eta * eps
            return eps_scaled.copy(), eps_scaled, float(eta)

        eta = ((1.0 - alpha_SNR) / alpha_SNR) * (signal_norm / eps_norm)
        eps_scaled = eta * eps
        y_noisy = y_clean + eps_scaled

        return y_noisy, eps_scaled, float(eta)


    def simulate(
        self,
        x: np.ndarray,
        L: np.ndarray,
        alpha_SNR: float = 0.5,
        sensor_white_noise_std: float = 1.0,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Simulate sensor data by (1) forward projection and (2) noise addition.

        Parameters
        ----------
        x : np.ndarray
            Source activity:
              - fixed: (n_sources, n_times)
              - free : (n_sources, K, n_times), with K = 2 for MEG (tangential) and K = 3 for EEG (full 3D).
              Typically in nAm for dipole moments.
        L : np.ndarray
            Leadfield:
              - fixed: (n_sensors, n_sources)
              - free : (n_sensors, n_sources, K), with K = 2 for MEG and K = 3 for EEG.
                Units depend on modality and construction, e.g. fT/nAm for MEG or µV/nAm for EEG.
        alpha_SNR : float
            Noise mixing parameter in [0, 1].
        sensor_white_noise_std : float
            Base noise standard deviation.
        seed : int
            Noise seed (used only for sensor noise).

        Returns
        -------
        y_clean : np.ndarray
            Noiseless sensor measurements, shape (n_sensors, n_times). Units depend on L and x, e.g. fT for MEG or µV for EEG.
        y_noisy : np.ndarray
            Noisy sensor measurements, shape (n_sensors, n_times). Units depend on L and x, e.g. fT for MEG or µV for EEG.
        noise : np.ndarray
            Added noise term (eta * eps), shape (n_sensors, n_times).
        noise_eta : float
            Noise scaling factor eta.
        """
        y_clean = self._project_sources_to_sensors(x=x, L=L)
        y_noisy, noise, noise_eta = self._add_noise(
            y_clean=y_clean,
            alpha_SNR=alpha_SNR,
            sensor_white_noise_std=sensor_white_noise_std,
            noise_seed=int(seed),
        )
        return y_clean, y_noisy, noise, noise_eta