
"""
@author: Ismail Huseynov
"""


import logging
from typing import Optional, Dict, Any, Tuple

import numpy as np
import mne
from scipy.signal import butter, filtfilt
from mne.io.constants import FIFF


# =============================================================================
# Source simulator
# =============================================================================
class SourceSimulator:
    """
    Simulate synthetic source coefficients for three settings:

    1) fixed:
         s shape = (N, T)
         One scalar coefficient per source location.

    2) free_eeg:
         x shape = (N, 3, T)
         General 3D coefficient process in the retained local 3D source basis.

    3) free_meg:
         a shape = (N, 2, T)
         General reduced 2D coefficient process in the MEG-sensitive local subspace.
    """

    def __init__(
        self,
        ERP_config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.ERP_config = ERP_config if ERP_config else {
            "tmin": -0.5,
            "tmax": 0.5,
            "stim_onset": 0.0,
            "sfreq": 250,
            "fmin": 1.0,
            "fmax": 5.0,
            "amplitude_distribution": {
                "median": 20.0,   # nAm
                "sigma": 0.2,
                "clip": (2.5, 50.0),
            },
            "random_erp_timing": True,
            "erp_min_length": None,
        }

        self.logger = logger if logger else logging.getLogger(__name__)

        # metadata only
        self.kind: int = FIFF.FIFFV_DIPOLE_WAVE
        self.units: int = FIFF.FIFF_UNIT_AM
        self.unitmult: int = FIFF.FIFF_UNITM_N

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

    def _sample_source_amplitude(self, rng: np.random.RandomState) -> float:
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

    def _simulate_erp_waveform(self, source_seed: int = 512) -> np.ndarray:
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

        white = rng.randn(erp_len)

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

    def simulate(
        self,
        setting: str = "fixed",   # "fixed", "free_eeg", "free_meg"
        n_sources: int = 100,
        nnz: int = 5,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        setting = setting.lower().strip()

        if setting == "fixed":
            return self._simulate_fixed(
                n_sources=n_sources,
                nnz=nnz,
                trial_seed=int(seed),
            )

        if setting == "free_eeg":
            # general 3D coefficient process in the retained local 3D basis
            return self._simulate_multicomponent(
                n_sources=n_sources,
                nnz=nnz,
                n_comp=3,
                trial_seed=int(seed),
            )

        if setting == "free_meg":
            # general reduced 2D coefficient process in the MEG-sensitive local subspace
            return self._simulate_multicomponent(
                n_sources=n_sources,
                nnz=nnz,
                n_comp=2,
                trial_seed=int(seed),
            )

        raise ValueError("setting must be 'fixed', 'free_eeg', or 'free_meg'")


# =============================================================================
# Sensor simulator
# =============================================================================
class SensorSimulator:
    """
    Simulate sensor data from source coefficients.

    Supported:
      fixed:
        x: (N, T)
        L: (M, N)

      reduced/free:
        x: (N, K, T)
        L: (M, N, K)
        with K = 2 or 3
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger if logger else logging.getLogger(__name__)

        # default metadata (MEG mags unless overwritten)
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

    def _project_sources_to_sensors(self, x: np.ndarray, L: np.ndarray) -> np.ndarray:
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
        y_clean = self._project_sources_to_sensors(x=x, L=L)
        y_noisy, noise, noise_eta = self._add_noise(
            y_clean=y_clean,
            alpha_SNR=alpha_SNR,
            sensor_white_noise_std=sensor_white_noise_std,
            noise_seed=int(seed),
        )
        return y_clean, y_noisy, noise, noise_eta


# =============================================================================
# Leadfield helpers
# =============================================================================
def reshape_leadfield_flat_to_block(L_flat: np.ndarray, k: int) -> np.ndarray:
    """
    Convert flattened leadfield (M, kN) -> block form (M, N, k),
    assuming per-source grouped columns.
    """
    if L_flat.ndim != 2:
        raise ValueError(f"Expected 2D leadfield. Got {L_flat.shape}")
    M, kN = L_flat.shape
    if kN % k != 0:
        raise ValueError(f"Second dimension {kN} is not divisible by k={k}")
    N = kN // k
    return L_flat.reshape(M, N, k)


def flatten_leadfield_block(L_block: np.ndarray) -> np.ndarray:
    """
    Convert block leadfield (M, N, k) -> flattened (M, kN),
    with per-source grouped columns.
    """
    if L_block.ndim != 3:
        raise ValueError(f"Expected 3D leadfield block. Got {L_block.shape}")
    M, N, k = L_block.shape
    return L_block.reshape(M, N * k)


def _canonicalize_basis_columns(Q: np.ndarray) -> np.ndarray:
    """
    Stabilize basis column signs so that the entry with largest absolute value
    in each column is nonnegative.

    This removes arbitrary SVD sign flips across runs/platforms while leaving
    the spanned subspace unchanged.
    """
    if Q.ndim != 2:
        raise ValueError(f"Expected 2D basis matrix. Got {Q.shape}")

    Q_out = np.array(Q, dtype=float, copy=True)
    for j in range(Q_out.shape[1]):
        idx = int(np.argmax(np.abs(Q_out[:, j])))
        if Q_out[idx, j] < 0:
            Q_out[:, j] *= -1.0
    return Q_out


def lift_reduced_sources_to_3d(a_red: np.ndarray, Q_basis: np.ndarray) -> np.ndarray:
    """
    Lift reduced coordinates back to the retained local 3D source basis.

    Parameters
    ----------
    a_red : (N, k, T)
        Reduced source coefficients, typically with k=2 for free_meg.
    Q_basis : (N, 3, k)
        Per-source basis mapping reduced coordinates into the local 3D basis.

    Returns
    -------
    x_3d : (N, 3, T)
        Lifted coefficients in the local 3D source basis.
    """
    if a_red.ndim != 3:
        raise ValueError(f"Expected a_red with shape (N, k, T). Got {a_red.shape}")
    if Q_basis.ndim != 3:
        raise ValueError(f"Expected Q_basis with shape (N, 3, k). Got {Q_basis.shape}")
    if a_red.shape[0] != Q_basis.shape[0]:
        raise ValueError("Mismatch in number of sources between a_red and Q_basis")
    if a_red.shape[1] != Q_basis.shape[2]:
        raise ValueError("Mismatch in reduced dimension k between a_red and Q_basis")
    return np.einsum("nok,nkt->not", Q_basis, a_red)


# =============================================================================
# Leadfield extractor
# =============================================================================
def extract_subset_leadfield(
    fwd_fif: str,
    n_per_hemi: int = 642,
    seed: int = 0,
    scale_to_nAm: bool = False,
    *,
    setting: str = "fixed",    # "fixed", "free_eeg", "free_meg"
    coil_name: Optional[str] = None,   # None / "mag" / "grad" / "eeg"
) -> Dict[str, Any]:
    """
    Extract a reproducible hemispheric subset for the chosen setting.

    Returns a dict with:
      setting
      subset_idx
      fwd
      n_sources
      n_sensors
      L_flat
      L_block
      Q_basis

    and for free_meg also:
      L3_flat
      L3_block

    Notes
    -----
    For free settings, we use `surf_ori=True`, so the 3-component source model
    is interpreted in a surface-based local source basis.

    - free_eeg keeps this full local 3D basis (k=3)
    - free_meg reduces each local 3D block to a 2D SVD basis (k=2)

    `Q_basis` is therefore:
    - identity for free_eeg, as an API-consistent placeholder in the retained 3D basis
    - the local 3x2 SVD basis for free_meg
    """
    rng = np.random.RandomState(int(seed))

    fwd = mne.read_forward_solution(fwd_fif, verbose="error")

    # channel picking
    if coil_name is not None:
        coil_name = coil_name.lower().strip()
        if coil_name == "mag":
            fwd = mne.pick_types_forward(fwd, meg="mag", eeg=False, ref_meg=False)
        elif coil_name == "grad":
            fwd = mne.pick_types_forward(fwd, meg="grad", eeg=False, ref_meg=False)
        elif coil_name == "eeg":
            fwd = mne.pick_types_forward(fwd, meg=False, eeg=True, ref_meg=False)
        else:
            raise ValueError("coil_name must be None, 'mag', 'grad', or 'eeg'")

    setting = setting.lower().strip()

    # -------------------------------------------------------------------------
    # fixed
    # -------------------------------------------------------------------------
    if setting == "fixed":
        fwd_out = mne.convert_forward_solution(
            fwd,
            force_fixed=True,
            surf_ori=True,
            verbose="error",
        )
        L_full = fwd_out["sol"]["data"].astype(float)   # (M, Nfull)

        src = fwd_out["src"]
        n_lh = int(src[0]["nuse"])
        n_rh = int(src[1]["nuse"])

        k_lh = min(int(n_per_hemi), n_lh)
        k_rh = min(int(n_per_hemi), n_rh)

        lh_sel = rng.choice(n_lh, size=k_lh, replace=False)
        rh_sel = rng.choice(n_rh, size=k_rh, replace=False)

        subset_idx = np.sort(np.concatenate([lh_sel, n_lh + rh_sel]).astype(int))
        L_sub = L_full[:, subset_idx]

        if scale_to_nAm:
            L_sub = L_sub * 1e-9

        return {
            "setting": "fixed",
            "subset_idx": subset_idx,
            "fwd": fwd_out,
            "n_sources": len(subset_idx),
            "n_sensors": L_sub.shape[0],
            "L_flat": L_sub,      # (M, N)
            "L_block": None,
            "Q_basis": None,
        }

    # -------------------------------------------------------------------------
    # free EEG or free MEG start from free 3D forward
    # -------------------------------------------------------------------------
    if setting in {"free_eeg", "free_meg"}:
        fwd_out = mne.convert_forward_solution(
            fwd,
            force_fixed=False,
            surf_ori=True,
            verbose="error",
        )
        L_full3 = fwd_out["sol"]["data"].astype(float)   # (M, 3Nfull)

        src = fwd_out["src"]
        n_lh = int(src[0]["nuse"])
        n_rh = int(src[1]["nuse"])

        k_lh = min(int(n_per_hemi), n_lh)
        k_rh = min(int(n_per_hemi), n_rh)

        lh_sel = rng.choice(n_lh, size=k_lh, replace=False)
        rh_sel = rng.choice(n_rh, size=k_rh, replace=False)

        subset_idx = np.sort(np.concatenate([lh_sel, n_lh + rh_sel]).astype(int))

        cols3 = (3 * subset_idx[:, None] + np.arange(3)[None, :]).reshape(-1)
        L_sub3_flat = L_full3[:, cols3]           # (M, 3Nsub)

        if scale_to_nAm:
            L_sub3_flat = L_sub3_flat * 1e-9

        L_sub3_block = reshape_leadfield_flat_to_block(L_sub3_flat, k=3)   # (M, Nsub, 3)
        M, Nsub, _ = L_sub3_block.shape

        # ---------------------------------------------------------------------
        # free EEG: keep full 3D
        # ---------------------------------------------------------------------
        if setting == "free_eeg":
            Q_basis = np.repeat(np.eye(3)[None, :, :], Nsub, axis=0)   # (Nsub, 3, 3)

            return {
                "setting": "free_eeg",
                "subset_idx": subset_idx,
                "fwd": fwd_out,
                "n_sources": Nsub,
                "n_sensors": M,
                "L_flat": L_sub3_flat,     # (M, 3Nsub)
                "L_block": L_sub3_block,   # (M, Nsub, 3)
                "Q_basis": Q_basis,        # identity bases in the retained local 3D basis
            }

        # ---------------------------------------------------------------------
        # free MEG: reduce each local block Mx3 -> Mx2 via local SVD
        # ---------------------------------------------------------------------
        Q_basis = np.zeros((Nsub, 3, 2))
        L_red_block = np.zeros((M, Nsub, 2))

        for i in range(Nsub):
            Li = L_sub3_block[:, i, :]          # (M, 3)
            _, _, Vt = np.linalg.svd(Li, full_matrices=False)
            Qi = _canonicalize_basis_columns(Vt[:2, :].T)   # (3, 2)
            Q_basis[i] = Qi
            L_red_block[:, i, :] = Li @ Qi      # (M, 2)

        L_red_flat = flatten_leadfield_block(L_red_block)   # (M, 2Nsub)

        return {
            "setting": "free_meg",
            "subset_idx": subset_idx,
            "fwd": fwd_out,
            "n_sources": Nsub,
            "n_sensors": M,
            "L_flat": L_red_flat,       # (M, 2Nsub)
            "L_block": L_red_block,     # (M, Nsub, 2)
            "Q_basis": Q_basis,         # (Nsub, 3, 2)
            "L3_flat": L_sub3_flat,     # original local 3D free leadfield subset
            "L3_block": L_sub3_block,   # (M, Nsub, 3)
        }

    raise ValueError("setting must be 'fixed', 'free_eeg', or 'free_meg'")


# =============================================================================
# Example usage
# =============================================================================

# path to forward
FWD_FIF = r"C:\Users\User\CaliBrain\calibrain\CC120264-fwd.fif"

src_sim = SourceSimulator()
sen_sim = SensorSimulator()

# ============================================================
# 1) Fixed MEG  --> k = 1, n_orient = 1
# ============================================================
lf_fixed_meg = extract_subset_leadfield(
    fwd_fif=FWD_FIF,
    setting="fixed",
    coil_name="mag",          # or "grad"
    scale_to_nAm=True,        # recommended for unit consistency
)

x_fixed_meg, active_fixed_meg = src_sim.simulate(
    setting="fixed",
    n_sources=lf_fixed_meg["n_sources"],
    nnz=10,
    seed=43,
)

y_clean_fixed_meg, y_noisy_fixed_meg, noise_fixed_meg, eta_fixed_meg = sen_sim.simulate(
    x=x_fixed_meg,
    L=lf_fixed_meg["L_flat"],
    alpha_SNR=0.5,
    seed=43,
)

# ============================================================
# 2) Fixed EEG  --> k = 1, n_orient = 1
# ============================================================
# Use this only if the forward file actually contains EEG channels.
# lf_fixed_eeg = extract_subset_leadfield(
#     fwd_fif=FWD_FIF,
#     setting="fixed",
#     coil_name="eeg",
#     scale_to_nAm=True,
# )
#
# x_fixed_eeg, active_fixed_eeg = src_sim.simulate(
#     setting="fixed",
#     n_sources=lf_fixed_eeg["n_sources"],
#     nnz=10,
#     seed=43,
# )
#
# y_clean_fixed_eeg, y_noisy_fixed_eeg, noise_fixed_eeg, eta_fixed_eeg = sen_sim.simulate(
#     x=x_fixed_eeg,
#     L=lf_fixed_eeg["L_flat"],
#     alpha_SNR=0.5,
#     seed=43,
# )

# ============================================================
# 3) Free EEG  --> k = 3, n_orient = 3
# ============================================================
# lf_free_eeg = extract_subset_leadfield(
#     fwd_fif=FWD_FIF,
#     setting="free_eeg",
#     coil_name="eeg",
#     scale_to_nAm=True,
# )
#
# x_free_eeg, active_free_eeg = src_sim.simulate(
#     setting="free_eeg",
#     n_sources=lf_free_eeg["n_sources"],
#     nnz=10,
#     seed=43,
# )
#
# y_clean_free_eeg, y_noisy_free_eeg, noise_free_eeg, eta_free_eeg = sen_sim.simulate(
#     x=x_free_eeg,
#     L=lf_free_eeg["L_block"],
#     alpha_SNR=0.5,
#     seed=43,
# )

# ============================================================
# 4) Reduced free MEG  --> k = 2, n_orient = 2
# ============================================================
lf_free_meg = extract_subset_leadfield(
    fwd_fif=FWD_FIF,
    setting="free_meg",
    coil_name="mag",
    scale_to_nAm=True,
)

a_free_meg, active_free_meg = src_sim.simulate(
    setting="free_meg",
    n_sources=lf_free_meg["n_sources"],
    nnz=10,
    seed=43,
)

y_clean_free_meg, y_noisy_free_meg, noise_free_meg, eta_free_meg = sen_sim.simulate(
    x=a_free_meg,
    L=lf_free_meg["L_block"],
    alpha_SNR=0.5,
    seed=43,
)

# Optional: lift reduced MEG coordinates back to the retained local 3D basis
x_free_meg_3d = lift_reduced_sources_to_3d(a_free_meg, lf_free_meg["Q_basis"])



