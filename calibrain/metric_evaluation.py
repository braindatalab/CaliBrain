import logging
from typing import Optional, Dict, Any, Iterable, Tuple

import numpy as np
from calibrain import UncertaintyEstimator

try:
    from scipy.spatial.distance import cdist
except Exception:
    cdist = None

try:
    from ot import emd2
except Exception:
    emd2 = None


# =============================================================================
# Helpers
# =============================================================================
def lift_reduced_sources_to_3d(a_red: np.ndarray, Q_basis: np.ndarray) -> np.ndarray:
    """
    Lift reduced coordinates back to the retained local 3D source basis.

    Parameters
    ----------
    a_red : (N, k, T)
    Q_basis : (N, 3, k)

    Returns
    -------
    x_3d : (N, 3, T)
    """
    a_red = np.asarray(a_red, dtype=float)
    Q_basis = np.asarray(Q_basis, dtype=float)

    if a_red.ndim != 3:
        raise ValueError(f"Expected a_red with shape (N, k, T). Got {a_red.shape}")
    if Q_basis.ndim != 3:
        raise ValueError(f"Expected Q_basis with shape (N, 3, k). Got {Q_basis.shape}")
    if a_red.shape[0] != Q_basis.shape[0]:
        raise ValueError("Mismatch in number of sources between a_red and Q_basis")
    if a_red.shape[1] != Q_basis.shape[2]:
        raise ValueError("Mismatch in reduced dimension k between a_red and Q_basis")

    return np.einsum("nok,nkt->not", Q_basis, a_red)


def _reshape_free_mean(arr: np.ndarray, n_sources: int, n_components: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 3:
        if arr.shape[0] != n_sources or arr.shape[1] != n_components:
            raise ValueError(
                f"Expected shape ({n_sources},{n_components},T); got {arr.shape}"
            )
        return arr
    if arr.ndim != 2:
        raise ValueError("Posterior means must be 2D or 3D arrays.")
    n_times = arr.shape[1]
    expected = n_sources * n_components
    if arr.shape[0] != expected:
        raise ValueError(
            f"First dimension must equal {expected}; got {arr.shape[0]}"
        )
    return arr.reshape(n_components, n_sources, n_times).transpose(1, 0, 2)


def get_subset_source_rr(
    lf_dict: Dict[str, Any],
    *,
    to_mm: bool = False,
) -> np.ndarray:
    """
    Get subset-aligned source coordinates for EMD and other source-space metrics.

    Parameters
    ----------
    lf_dict : dict
        Output of extract_subset_leadfield(...), containing:
          - "fwd"
          - "subset_idx"

    to_mm : bool
        If True, return coordinates in millimeters.
        Otherwise return coordinates in meters (MNE default).

    Returns
    -------
    coords_subset : ndarray, shape (N, 3)
        Source coordinates aligned exactly with lf_dict["subset_idx"] and therefore
        with the source ordering used in L_flat and L_block.

    Notes
    -----
    - This helper uses the used-source ordering in the forward solution:
          [left hemisphere used sources, right hemisphere used sources]
    - This matches how subset_idx is constructed inside extract_subset_leadfield(...).
    - The same helper works for fixed, free_eeg, and free_meg, because subset_idx
      is source-level, not coefficient-level.
    """
    if "fwd" not in lf_dict:
        raise ValueError('lf_dict must contain key "fwd".')
    if "subset_idx" not in lf_dict:
        raise ValueError('lf_dict must contain key "subset_idx".')

    fwd = lf_dict["fwd"]
    subset_idx = np.asarray(lf_dict["subset_idx"], dtype=int)

    src = fwd["src"]
    if len(src) != 2:
        raise ValueError("Expected a two-hemisphere source space in fwd['src'].")

    rr_lh = np.asarray(src[0]["rr"][src[0]["vertno"]], dtype=float)
    rr_rh = np.asarray(src[1]["rr"][src[1]["vertno"]], dtype=float)
    rr_used = np.vstack([rr_lh, rr_rh])

    if np.any(subset_idx < 0) or np.any(subset_idx >= rr_used.shape[0]):
        raise ValueError(
            f"subset_idx contains out-of-range entries for used-source coordinates. "
            f"Valid range is [0, {rr_used.shape[0] - 1}]."
        )

    coords_subset = rr_used[subset_idx]

    if to_mm:
        coords_subset = 1000.0 * coords_subset

    return coords_subset


# =============================================================================
# MetricEvaluator
# =============================================================================
DEFAULT_CALIBRATION_METRICS = (
    "mean_signed_deviation",
    "mean_absolute_deviation",
    "max_underconfidence_deviation",
    "max_overconfidence_deviation",
)


class MetricEvaluator:
    """
    Evaluator aligned with the updated UncertaintyEstimator.

    Supported settings
    ------------------
    - fixed
    - eeg_free
    - meg_free

    Supported modes
    ---------------
    - pointwise
    - aggregated

    Conventions
    -----------
    1) fixed:
       x_true, x_hat have shape (N,T)

    2) eeg_free:
       x_true, x_hat have shape (N,3,T)
       error metrics use amplitude representation ||x_i(t)||_2

    3) meg_free:
       x_true is 3D truth in the retained local 3D basis, shape (N,3,T)
       x_hat is reduced 2D posterior mean, shape (N,2,T) or flat (2N,T)
       posterior covariance is reduced 2D, shape (2N,2N)
       error metrics use reduced-coordinate amplitude representation after
       projecting truth to 2D via the same V_tan basis used by the inverse

    Notes
    -----
    - Aggregated mode always means time-average.
    - EMD expects coords already aligned with the source subset being evaluated.
    - Calibration metrics are:
        * max_underconfidence_deviation
        * max_overconfidence_deviation
        * mean_absolute_deviation
        * mean_signed_deviation
    - For MEG, pass V_tan = lf_free_meg["Q_basis"] whenever possible to avoid
      basis mismatches.
    """

    def __init__(
        self,
        ue: UncertaintyEstimator,
        *,
        nominal_coverages: Optional[Iterable[float]] = None,
        evaluation_metrics: Optional[Iterable[str]] = None,
        calibration_metrics: Optional[Iterable[str]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.ue = ue
        self.logger = logger or logging.getLogger(__name__)
        if nominal_coverages is not None:
            cov = np.asarray(list(nominal_coverages), dtype=float)
            if cov.ndim != 1 or cov.size == 0:
                raise ValueError("nominal_coverages must be a 1-D array with at least one entry.")
            self.ue.nominal_coverages = cov
        self.nominal_coverages = np.asarray(self.ue.nominal_coverages, dtype=float)
        self.evaluation_metrics = (
            tuple(evaluation_metrics) if evaluation_metrics is not None else tuple()
        )
        self.calibration_metrics = (
            tuple(calibration_metrics) if calibration_metrics is not None else DEFAULT_CALIBRATION_METRICS
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _check_setting(setting: str) -> str:
        setting = (setting or "").lower().strip()
        if setting not in {"fixed", "eeg_free", "meg_free"}:
            raise ValueError("setting must be one of: 'fixed', 'eeg_free', 'meg_free'.")
        return setting

    @staticmethod
    def _check_mode(mode: str) -> str:
        mode = (mode or "").lower().strip()
        if mode not in {"pointwise", "aggregated"}:
            raise ValueError("mode must be one of: 'pointwise', 'aggregated'.")
        return mode

    def __getstate__(self):
        state = self.__dict__.copy()
        state['logger'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
        if not hasattr(self, "evaluation_metrics"):
            legacy = state.get("metrics")
            if legacy is None:
                legacy = ()
            self.evaluation_metrics = tuple(legacy)
        if not hasattr(self, "calibration_metrics") or self.calibration_metrics is None:
            self.calibration_metrics = DEFAULT_CALIBRATION_METRICS

    @staticmethod
    def _reshape_meg_mean_if_needed(x_hat_meg: np.ndarray, N: int, T: int) -> np.ndarray:
        x_hat_meg = np.asarray(x_hat_meg, dtype=float)
        if x_hat_meg.ndim == 3:
            if x_hat_meg.shape != (N, 2, T):
                raise ValueError(f"x_hat for meg_free must be (N,2,T); got {x_hat_meg.shape}")
            return x_hat_meg
        if x_hat_meg.ndim == 2:
            if x_hat_meg.shape != (2 * N, T):
                raise ValueError(f"x_hat for meg_free flat form must be (2N,T); got {x_hat_meg.shape}")
            return x_hat_meg.reshape(N, 2, T)
        raise ValueError("x_hat for meg_free must have shape (N,2,T) or (2N,T).")

    @staticmethod
    def _cov_blocks_from_full(posterior_uncert: np.ndarray, block_dim: int) -> np.ndarray:
        posterior_uncert = np.asarray(posterior_uncert, dtype=float)

        if posterior_uncert.ndim == 3:
            if posterior_uncert.shape[1:] != (block_dim, block_dim):
                raise ValueError(
                    f"Block covariance form must be (N,{block_dim},{block_dim}); "
                    f"got {posterior_uncert.shape}"
                )
            return posterior_uncert

        if posterior_uncert.ndim == 2:
            M, K = posterior_uncert.shape
            if M != K or M % block_dim != 0:
                raise ValueError(
                    f"Full covariance must be square with size multiple of {block_dim}; "
                    f"got {posterior_uncert.shape}"
                )
            N = M // block_dim
            blocks = np.zeros((N, block_dim, block_dim), dtype=float)
            for i in range(N):
                blocks[i] = posterior_uncert[
                    block_dim * i:block_dim * i + block_dim,
                    block_dim * i:block_dim * i + block_dim,
                ]
            return blocks

        raise ValueError("posterior_uncert must be either full covariance or block covariance.")

    @staticmethod
    def _fixed_variance_from_uncert(posterior_uncert: np.ndarray) -> np.ndarray:
        posterior_uncert = np.asarray(posterior_uncert, dtype=float)
        if posterior_uncert.ndim == 1:
            return np.maximum(posterior_uncert, 0.0)
        if posterior_uncert.ndim == 2:
            if posterior_uncert.shape[0] != posterior_uncert.shape[1]:
                raise ValueError("Fixed posterior covariance must be square.")
            return np.maximum(np.diag(posterior_uncert), 0.0)
        raise ValueError("For fixed setting, posterior_uncert must be (N,) or (N,N).")

    def _get_meg_truth_2d(
        self,
        x_true_meg_3d: np.ndarray,
        *,
        V_tan: Optional[np.ndarray] = None,
        L_free_Mx3N: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        x_true_meg_3d = np.asarray(x_true_meg_3d, dtype=float)
        if x_true_meg_3d.ndim != 3 or x_true_meg_3d.shape[1] != 3:
            raise ValueError(f"x_true for meg_free must be (N,3,T); got {x_true_meg_3d.shape}")

        N = x_true_meg_3d.shape[0]

        if V_tan is None:
            if L_free_Mx3N is None:
                raise ValueError("Provide either V_tan or L_free_Mx3N for meg_free.")
            V_tan = self.ue.precompute_meg_tangent_bases_svd(L_free_Mx3N)

        V_tan = np.asarray(V_tan, dtype=float)
        if V_tan.shape != (N, 3, 2):
            raise ValueError(f"V_tan must be (N,3,2); got {V_tan.shape}")

        return self.ue.project_meg_3d_to_2d(x_true_meg_3d, V_tan)

    @staticmethod
    def calibration_metrics_4(nominal: np.ndarray, empirical: np.ndarray) -> Dict[str, float]:
        nominal = np.asarray(nominal, dtype=float)
        empirical = np.asarray(empirical, dtype=float)

        dev = empirical - nominal
        under = np.maximum(nominal - empirical, 0.0)
        over = np.maximum(empirical - nominal, 0.0)

        return {
            "max_underconfidence_deviation": float(np.max(under)),
            "max_overconfidence_deviation": float(np.max(over)),
            "mean_absolute_deviation": float(np.mean(np.abs(dev))),
            "mean_signed_deviation": float(np.mean(dev)),
        }

    # ------------------------------------------------------------------
    # Signal extraction for error metrics
    # ------------------------------------------------------------------
    def _signals_for_error_metrics(
        self,
        *,
        x_true: np.ndarray,
        x_hat: np.ndarray,
        setting: str,
        mode: str,
        V_tan: Optional[np.ndarray] = None,
        L_free_Mx3N: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        setting = self._check_setting(setting)
        mode = self._check_mode(mode)

        if setting == "fixed":
            x_true = np.asarray(x_true, dtype=float)
            x_hat = np.asarray(x_hat, dtype=float)

            if x_true.ndim != 2 or x_hat.ndim != 2 or x_true.shape != x_hat.shape:
                raise ValueError("For fixed, x_true and x_hat must both be (N,T).")

            if mode == "pointwise":
                return {"truth_signal": x_true, "est_signal": x_hat}

            x_true_agg = np.mean(x_true, axis=1)
            x_hat_agg = np.mean(x_hat, axis=1)
            return {"truth_signal": x_true_agg, "est_signal": x_hat_agg}

        if setting == "eeg_free":
            x_true = np.asarray(x_true, dtype=float)
            x_hat = np.asarray(x_hat, dtype=float)

            if x_true.ndim != 3 or x_true.shape[1] != 3:
                raise ValueError(f"For eeg_free, x_true must be (N,3,T); got {x_true.shape}")
            if x_hat.shape != x_true.shape:
                raise ValueError("For eeg_free, x_hat must match x_true shape (N,3,T).")

            if mode == "pointwise":
                amp_true = np.linalg.norm(x_true, axis=1)
                amp_hat = np.linalg.norm(x_hat, axis=1)
                return {"truth_signal": amp_true, "est_signal": amp_hat}

            x_true_agg = np.mean(x_true, axis=2)
            x_hat_agg = np.mean(x_hat, axis=2)
            amp_true_agg = np.linalg.norm(x_true_agg, axis=1)
            amp_hat_agg = np.linalg.norm(x_hat_agg, axis=1)
            return {"truth_signal": amp_true_agg, "est_signal": amp_hat_agg}

        # meg_free
        x_true = np.asarray(x_true, dtype=float)
        if x_true.ndim != 3 or x_true.shape[1] != 3:
            raise ValueError(f"For meg_free, x_true must be (N,3,T); got {x_true.shape}")

        N, _, T = x_true.shape
        x_hat_2d = self._reshape_meg_mean_if_needed(x_hat, N, T)
        x_true_2d = self._get_meg_truth_2d(
            x_true,
            V_tan=V_tan,
            L_free_Mx3N=L_free_Mx3N,
        )

        if mode == "pointwise":
            amp_true = np.linalg.norm(x_true_2d, axis=1)
            amp_hat = np.linalg.norm(x_hat_2d, axis=1)
            return {"truth_signal": amp_true, "est_signal": amp_hat}

        x_true_agg = np.mean(x_true_2d, axis=2)
        x_hat_agg = np.mean(x_hat_2d, axis=2)
        amp_true_agg = np.linalg.norm(x_true_agg, axis=1)
        amp_hat_agg = np.linalg.norm(x_hat_agg, axis=1)
        return {"truth_signal": amp_true_agg, "est_signal": amp_hat_agg}

    # ------------------------------------------------------------------
    # Error metrics
    # ------------------------------------------------------------------
    def mse(
        self,
        *,
        x_true: np.ndarray,
        x_hat: np.ndarray,
        setting: str,
        mode: str = "pointwise",
        V_tan: Optional[np.ndarray] = None,
        L_free_Mx3N: Optional[np.ndarray] = None,
    ) -> float:
        sig = self._signals_for_error_metrics(
            x_true=x_true,
            x_hat=x_hat,
            setting=setting,
            mode=mode,
            V_tan=V_tan,
            L_free_Mx3N=L_free_Mx3N,
        )
        d = sig["truth_signal"] - sig["est_signal"]
        return float(np.mean(d * d))

    def mae(
        self,
        *,
        x_true: np.ndarray,
        x_hat: np.ndarray,
        setting: str,
        mode: str = "pointwise",
        V_tan: Optional[np.ndarray] = None,
        L_free_Mx3N: Optional[np.ndarray] = None,
    ) -> float:
        sig = self._signals_for_error_metrics(
            x_true=x_true,
            x_hat=x_hat,
            setting=setting,
            mode=mode,
            V_tan=V_tan,
            L_free_Mx3N=L_free_Mx3N,
        )
        return float(np.mean(np.abs(sig["truth_signal"] - sig["est_signal"])))

    def rmse(
        self,
        *,
        x_true: np.ndarray,
        x_hat: np.ndarray,
        setting: str,
        mode: str = "pointwise",
        V_tan: Optional[np.ndarray] = None,
        L_free_Mx3N: Optional[np.ndarray] = None,
    ) -> float:
        return float(np.sqrt(self.mse(
            x_true=x_true,
            x_hat=x_hat,
            setting=setting,
            mode=mode,
            V_tan=V_tan,
            L_free_Mx3N=L_free_Mx3N,
        )))

    def rmae(
        self,
        *,
        x_true: np.ndarray,
        x_hat: np.ndarray,
        setting: str,
        mode: str = "pointwise",
        V_tan: Optional[np.ndarray] = None,
        L_free_Mx3N: Optional[np.ndarray] = None,
    ) -> float:
        return float(np.sqrt(self.mae(
            x_true=x_true,
            x_hat=x_hat,
            setting=setting,
            mode=mode,
            V_tan=V_tan,
            L_free_Mx3N=L_free_Mx3N,
        )))

    # ------------------------------------------------------------------
    # Posterior uncertainty summary
    # ------------------------------------------------------------------
    def mean_posterior_std(
        self,
        *,
        posterior_uncert: np.ndarray,
        setting: str,
        mode: str = "pointwise",
        n_times: Optional[int] = None,
    ) -> float:
        setting = self._check_setting(setting)
        mode = self._check_mode(mode)

        if setting == "fixed":
            var = self._fixed_variance_from_uncert(posterior_uncert)
            std = np.sqrt(np.maximum(var, 0.0))
            if mode == "aggregated":
                if n_times is None:
                    raise ValueError("For aggregated mode, n_times must be provided.")
                std = std / np.sqrt(float(n_times))
            return float(np.mean(std))

        if setting == "eeg_free":
            blocks = self._cov_blocks_from_full(posterior_uncert, block_dim=3)
            source_std = np.sqrt(np.maximum(np.trace(blocks, axis1=1, axis2=2) / 3.0, 0.0))
            if mode == "aggregated":
                if n_times is None:
                    raise ValueError("For aggregated mode, n_times must be provided.")
                source_std = source_std / np.sqrt(float(n_times))
            return float(np.mean(source_std))

        # meg_free
        blocks = self._cov_blocks_from_full(posterior_uncert, block_dim=2)
        source_std = np.sqrt(np.maximum(np.trace(blocks, axis1=1, axis2=2) / 2.0, 0.0))
        if mode == "aggregated":
            if n_times is None:
                raise ValueError("For aggregated mode, n_times must be provided.")
            source_std = source_std / np.sqrt(float(n_times))
        return float(np.mean(source_std))

    # ------------------------------------------------------------------
    # EMD
    # ------------------------------------------------------------------
    def _source_mass_for_emd(
        self,
        *,
        x: np.ndarray,
        setting: str,
        mode: str,
        V_tan: Optional[np.ndarray] = None,
        L_free_Mx3N: Optional[np.ndarray] = None,
        is_truth: bool = True,
    ) -> np.ndarray:
        setting = self._check_setting(setting)
        mode = self._check_mode(mode)

        if setting == "fixed":
            x = np.asarray(x, dtype=float)
            if x.ndim != 2:
                raise ValueError("For fixed, x must be (N,T).")
            if mode == "pointwise":
                return np.linalg.norm(x, axis=1)
            return np.abs(np.mean(x, axis=1))

        if setting == "eeg_free":
            x = np.asarray(x, dtype=float)
            if x.ndim != 3 or x.shape[1] != 3:
                raise ValueError("For eeg_free, x must be (N,3,T).")
            if mode == "pointwise":
                amp = np.linalg.norm(x, axis=1)
                return np.linalg.norm(amp, axis=1)
            x_agg = np.mean(x, axis=2)
            return np.linalg.norm(x_agg, axis=1)

        # meg_free
        if is_truth:
            x_true_2d = self._get_meg_truth_2d(
                x,
                V_tan=V_tan,
                L_free_Mx3N=L_free_Mx3N,
            )
            if mode == "pointwise":
                amp = np.linalg.norm(x_true_2d, axis=1)
                return np.linalg.norm(amp, axis=1)
            x_agg = np.mean(x_true_2d, axis=2)
            return np.linalg.norm(x_agg, axis=1)

        x = np.asarray(x, dtype=float)
        if x.ndim != 3 or x.shape[1] != 2:
            raise ValueError("For meg_free estimate, x must be (N,2,T).")
        if mode == "pointwise":
            amp = np.linalg.norm(x, axis=1)
            return np.linalg.norm(amp, axis=1)
        x_agg = np.mean(x, axis=2)
        return np.linalg.norm(x_agg, axis=1)

    def emd(
        self,
        *,
        x_true: np.ndarray,
        x_hat: np.ndarray,
        coords: np.ndarray,
        setting: str,
        mode: str = "pointwise",
        V_tan: Optional[np.ndarray] = None,
        L_free_Mx3N: Optional[np.ndarray] = None,
        eps: float = 1e-12,
    ) -> float:
        if cdist is None or emd2 is None:
            raise ImportError("EMD requires scipy.spatial.distance.cdist and POT (ot.emd2).")

        setting = self._check_setting(setting)
        mode = self._check_mode(mode)

        coords = np.asarray(coords, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords must be aligned subset coordinates with shape (N,3).")

        if setting == "meg_free":
            x_true = np.asarray(x_true, dtype=float)
            if x_true.ndim != 3 or x_true.shape[1] != 3:
                raise ValueError("For meg_free, x_true must be (N,3,T).")
            N, _, T = x_true.shape
            x_hat_2d = self._reshape_meg_mean_if_needed(x_hat, N, T)
            a = self._source_mass_for_emd(
                x=x_true,
                setting="meg_free",
                mode=mode,
                V_tan=V_tan,
                L_free_Mx3N=L_free_Mx3N,
                is_truth=True,
            )
            b = self._source_mass_for_emd(
                x=x_hat_2d,
                setting="meg_free",
                mode=mode,
                is_truth=False,
            )
        else:
            a = self._source_mass_for_emd(
                x=x_true,
                setting=setting,
                mode=mode,
                V_tan=V_tan,
                L_free_Mx3N=L_free_Mx3N,
                is_truth=True,
            )
            b = self._source_mass_for_emd(
                x=x_hat,
                setting=setting,
                mode=mode,
                V_tan=V_tan,
                L_free_Mx3N=L_free_Mx3N,
                is_truth=False,
            )

        if coords.shape[0] != a.shape[0] or coords.shape[0] != b.shape[0]:
            raise ValueError(
                f"coords must align with evaluated subset. "
                f"Got coords.shape[0]={coords.shape[0]}, masses {a.shape[0]} and {b.shape[0]}"
            )

        a_mask = a > eps
        b_mask = b > eps
        if not np.any(a_mask) or not np.any(b_mask):
            self.logger.warning("EMD: empty active set in true or estimate -> returning inf.")
            return float(np.inf)

        a_w = a[a_mask]
        b_w = b[b_mask]
        rr_a = coords[a_mask]
        rr_b = coords[b_mask]

        M = cdist(rr_a, rr_b, metric="euclidean")
        a_norm = a_w / np.sum(a_w)
        b_norm = b_w / np.sum(b_w)

        return float(emd2(a_norm, b_norm, M))


def _prepare_meg_sources_for_emd_dataset(eval_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_sources = int(eval_data.get("n_sources") or 0)
    if n_sources <= 0:
        raise ValueError("MEG dataset missing n_sources metadata required for EMD.")

    q_basis = eval_data.get("Q_basis")
    if q_basis is None:
        raise ValueError("MEG dataset missing Q_basis required for EMD.")
    q_basis = np.asarray(q_basis, dtype=float)

    x_hat = _reshape_free_mean(np.asarray(eval_data["x_hat"], dtype=float), n_sources, 2)

    x_true_raw = np.asarray(eval_data["x_true"], dtype=float)
    if x_true_raw.ndim == 3 and x_true_raw.shape[1] == 2:
        x_true_2d = x_true_raw
        x_true_3d = lift_reduced_sources_to_3d(x_true_2d, q_basis)
    elif x_true_raw.ndim == 3 and x_true_raw.shape[1] == 3:
        x_true_3d = x_true_raw
        basis_T = np.transpose(q_basis, (0, 2, 1))
        x_true_2d = np.einsum("nki,nit->nkt", basis_T, x_true_3d)
    else:
        x_true_2d = _reshape_free_mean(x_true_raw, n_sources, 2)
        x_true_3d = lift_reduced_sources_to_3d(x_true_2d, q_basis)

    return x_true_2d, x_true_3d, x_hat, q_basis


def _prepare_eeg_sources_for_emd_dataset(eval_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    n_sources = int(eval_data.get("n_sources") or 0)
    if n_sources <= 0:
        raise ValueError("EEG dataset missing n_sources metadata required for EMD.")

    x_true = np.asarray(eval_data["x_true"], dtype=float)
    if not (x_true.ndim == 3 and x_true.shape[1] == 3):
        x_true = _reshape_free_mean(x_true, n_sources, 3)

    x_hat = np.asarray(eval_data["x_hat"], dtype=float)
    if not (x_hat.ndim == 3 and x_hat.shape[1] == 3):
        x_hat = _reshape_free_mean(x_hat, n_sources, 3)

    return x_true, x_hat


def compute_dataset_emd(
    *,
    metric_evaluator: MetricEvaluator,
    eval_data: Dict[str, Any],
    coords: Optional[np.ndarray],
    setting: Optional[str],
    emd_mode: str = "reduced",
) -> Optional[float]:
    if coords is None or setting is None:
        return None

    logger = getattr(metric_evaluator, "logger", None) or logging.getLogger(__name__)
    mode = (emd_mode or "reduced").lower()
    if mode not in {"reduced", "lifted"}:
        raise ValueError("emd_mode must be 'reduced' or 'lifted'.")

    try:
        if setting == "meg_free":
            x_true_2d, x_true_3d, x_hat_2d, q_basis = _prepare_meg_sources_for_emd_dataset(eval_data)
            if mode == "lifted":
                x_hat_3d = lift_reduced_sources_to_3d(x_hat_2d, q_basis)
                return metric_evaluator.emd(
                    x_true=x_true_3d,
                    x_hat=x_hat_3d,
                    coords=coords,
                    setting="eeg_free",
                    mode="aggregated",
                )
            return metric_evaluator.emd(
                x_true=x_true_2d,
                x_hat=x_hat_2d,
                coords=coords,
                setting="meg_free",
                mode="aggregated",
                V_tan=q_basis,
            )

        if setting == "eeg_free":
            x_true_3d, x_hat_3d = _prepare_eeg_sources_for_emd_dataset(eval_data)
            return metric_evaluator.emd(
                x_true=x_true_3d,
                x_hat=x_hat_3d,
                coords=coords,
                setting="eeg_free",
                mode="aggregated",
            )

        return metric_evaluator.emd(
            x_true=eval_data["x_true"],
            x_hat=eval_data["x_hat"],
            coords=coords,
            setting=setting,
            mode="aggregated",
            V_tan=eval_data.get("Q_basis"),
        )
    except Exception as exc:
        logger.warning("EMD computation failed (%s mode): %s", setting, exc)
        return None

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def calibration_curve(
        self,
        *,
        x_true: np.ndarray,
        x_hat: np.ndarray,
        posterior_uncert: np.ndarray,
        setting: str,
        mode: str = "aggregated",
        V_tan: Optional[np.ndarray] = None,
        L_free_Mx3N: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        setting = self._check_setting(setting)
        mode = self._check_mode(mode)

        if setting == "fixed":
            posterior_var = self._fixed_variance_from_uncert(posterior_uncert)
            if mode == "pointwise":
                curve = self.ue.calibration_curve_intervals_pointwise(
                    x_true=x_true,
                    x_hat=x_hat,
                    posterior_var=posterior_var,
                )
            else:
                curve = self.ue.calibration_curve_intervals_aggregated(
                    x_true=x_true,
                    x_hat=x_hat,
                    posterior_var=posterior_var,
                )

        elif setting == "eeg_free":
            if mode == "pointwise":
                curve = self.ue.calibration_curve_ellipsoid_eeg_free_pointwise(
                    x_true=x_true,
                    x_hat=x_hat,
                    posterior_cov=posterior_uncert,
                )
            else:
                curve = self.ue.calibration_curve_ellipsoid_eeg_free_aggregated(
                    x_true=x_true,
                    x_hat=x_hat,
                    posterior_cov=posterior_uncert,
                )

        else:  # meg_free
            x_true = np.asarray(x_true, dtype=float)
            if x_true.ndim != 3 or x_true.shape[1] != 3:
                raise ValueError(f"For meg_free, x_true must be (N,3,T); got {x_true.shape}")
            N, _, T = x_true.shape
            x_hat_2d = self._reshape_meg_mean_if_needed(x_hat, N, T)

            if mode == "pointwise":
                curve = self.ue.calibration_curve_ellipse_meg_free_pointwise(
                    x_true_3d=x_true,
                    x_hat_2d=x_hat_2d,
                    posterior_cov_2d=posterior_uncert,
                    V_tan=V_tan,
                    L_free_Mx3N=L_free_Mx3N,
                )
            else:
                curve = self.ue.calibration_curve_ellipse_meg_free_aggregated(
                    x_true_3d=x_true,
                    x_hat_2d=x_hat_2d,
                    posterior_cov_2d=posterior_uncert,
                    V_tan=V_tan,
                    L_free_Mx3N=L_free_Mx3N,
                )

        nominal = np.asarray(curve["nominal_coverages"], dtype=float)
        empirical = np.asarray(curve["empirical_coverages"], dtype=float)

        return {
            "nominal": nominal,
            "empirical": empirical,
            "metrics_4": self.calibration_metrics_4(nominal, empirical),
        }

    # ------------------------------------------------------------------
    # All metrics together
    # ------------------------------------------------------------------
    def evaluate_all(
        self,
        *,
        x_true: np.ndarray,
        x_hat: np.ndarray,
        posterior_uncert: np.ndarray,
        setting: str,
        mode: str = "aggregated",
        coords: Optional[np.ndarray] = None,
        V_tan: Optional[np.ndarray] = None,
        L_free_Mx3N: Optional[np.ndarray] = None,
        compute_emd: bool = False,
    ) -> Dict[str, Any]:
        setting = self._check_setting(setting)
        mode = self._check_mode(mode)

        if mode == "aggregated":
            if setting == "fixed":
                n_times = np.asarray(x_true).shape[1]
            else:
                n_times = np.asarray(x_true).shape[2]
        else:
            n_times = None

        out = {
            "mse": self.mse(
                x_true=x_true,
                x_hat=x_hat,
                setting=setting,
                mode=mode,
                V_tan=V_tan,
                L_free_Mx3N=L_free_Mx3N,
            ),
            "mae": self.mae(
                x_true=x_true,
                x_hat=x_hat,
                setting=setting,
                mode=mode,
                V_tan=V_tan,
                L_free_Mx3N=L_free_Mx3N,
            ),
            "rmse": self.rmse(
                x_true=x_true,
                x_hat=x_hat,
                setting=setting,
                mode=mode,
                V_tan=V_tan,
                L_free_Mx3N=L_free_Mx3N,
            ),
            "rmae": self.rmae(
                x_true=x_true,
                x_hat=x_hat,
                setting=setting,
                mode=mode,
                V_tan=V_tan,
                L_free_Mx3N=L_free_Mx3N,
            ),
            "mean_posterior_std": self.mean_posterior_std(
                posterior_uncert=posterior_uncert,
                setting=setting,
                mode=mode,
                n_times=n_times,
            ),
            "calibration": self.calibration_curve(
                x_true=x_true,
                x_hat=x_hat,
                posterior_uncert=posterior_uncert,
                setting=setting,
                mode=mode,
                V_tan=V_tan,
                L_free_Mx3N=L_free_Mx3N,
            ),
        }

        if compute_emd:
            if coords is None:
                raise ValueError("compute_emd=True requires subset-aligned coords with shape (N,3).")
            out["emd"] = self.emd(
                x_true=x_true,
                x_hat=x_hat,
                coords=coords,
                setting=setting,
                mode=mode,
                V_tan=V_tan,
                L_free_Mx3N=L_free_Mx3N,
            )

        return out


# =============================================================================
# Example helpers
# =============================================================================
