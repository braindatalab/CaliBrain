from typing import Optional

from sklearn.isotonic import IsotonicRegression
import numpy as np
from mne.io.constants import FIFF
from calibrain import UncertaintyEstimator
from calibrain import MetricEvaluator
from calibrain.uncertainty_estimation import lift_reduced_sources_to_3d

EEG_COIL_TYPES = {FIFF.FIFFV_COIL_EEG}
MEG_COIL_TYPES = {
    FIFF.FIFFV_COIL_VV_MAG_T1,
    FIFF.FIFFV_COIL_VV_PLANAR_T1,
}
ALLOWED_COIL_TYPES = EEG_COIL_TYPES | MEG_COIL_TYPES


def _normalize_orientation(orientation_type: str | None) -> str:
    if orientation_type is None:
        return "fixed"
    normalized = orientation_type.lower()
    if normalized not in {"fixed", "free"}:
        raise ValueError(f"Unsupported orientation_type '{orientation_type}'")
    return normalized

def _validate_coil_type(coil_type: int | None) -> None:
    if coil_type is None:
        return
    if coil_type not in ALLOWED_COIL_TYPES:
        raise ValueError(
            f"Unsupported coil_type '{coil_type}'. Expected one of "
            f"{sorted(ALLOWED_COIL_TYPES)} or None."
        )

def is_fixed_orientation(orientation_type: str | None) -> bool:
    return _normalize_orientation(orientation_type) == "fixed"

def is_free_eeg_orientation(
    orientation_type: str | None,
    coil_type: int | None,
    n_components: int | None = None,
) -> bool:
    if _normalize_orientation(orientation_type) != "free":
        return False
    if coil_type is not None:
        _validate_coil_type(coil_type)
        return coil_type in EEG_COIL_TYPES
    return n_components == 3

def is_free_meg_orientation(
    orientation_type: str | None,
    coil_type: int | None,
    n_components: int | None = None,
) -> bool:
    if _normalize_orientation(orientation_type) != "free":
        return False
    if coil_type is not None:
        _validate_coil_type(coil_type)
        return coil_type in MEG_COIL_TYPES
    return n_components == 2


# ============================================================================
# UNCERTAINTY CALIBRATOR (NOMINAL RE-CALIBRATION)
# ============================================================================

class UncertaintyCalibrator:
    """
    Uncertainty Calibrator
    =====================

    This class connects UncertaintyEstimator and MetricEvaluator with an
    isotonic-regression based re-calibration of nominal coverages.

    It uses:
      - UncertaintyEstimator's aggregated membership/curve utilities to evaluate
        empirical coverage for any nominal coverage level c.
      - MetricEvaluator.calibration_metrics_4 (derived from the demos in
        ``metric_evaluation.py``) to summarize AAD, ASD, etc. using the default
        calibration metrics.
      - Experiment-level train/test splits provided by the caller to avoid
        overfitting entire sources.

    Conceptual model
    ----------------
    For each nominal coverage level c (e.g. 0.5, 0.9):

        g(c) = empirical coverage when building CIs with nominal coverage c.

    On the training experiments we learn g(c) via isotonic regression, then we
    numerically invert this curve to obtain a re-calibrated nominal coverage
    c_cal(c) ≈ g^{-1}(c). On a held-out evaluation split we build intervals
    using c_cal(c) and measure empirical coverage again.

    The final goal is that after recalibration the empirical coverage is
    closer to the original nominal coverage grid defined in your
    UncertaintyEstimator (self.nominal_coverages).
    """

    def __init__(self,
                 uncertainty_estimator: UncertaintyEstimator,
                 metric_evaluator: MetricEvaluator):
        """
        Parameters
        ----------
        uncertainty_estimator : UncertaintyEstimator
            Your existing UncertaintyEstimator instance.
        metric_evaluator : MetricEvaluator
            Your existing MetricEvaluator instance.
        """
        self.uncertainty_estimator = uncertainty_estimator
        self.metric_evaluator = metric_evaluator

        # Nominal coverage grid c (what we want in the end)
        self.nominal_coverages = uncertainty_estimator.nominal_coverages

        # Will store final mapping nominal -> recalibrated nominal
        self.calibration_model = None
        self.is_calibrated = False
        if getattr(self.metric_evaluator, "nominal_coverages", None) is None:
            self.metric_evaluator.nominal_coverages = self.nominal_coverages

    def _invert_isotonic(self,
                          iso_model: IsotonicRegression,
                          targets: np.ndarray,
                          grid_size: int = 2001) -> np.ndarray:
        """
        Numerically invert a fitted isotonic regression model.

        iso_model approximates the forward curve:

            empirical = g(self.nominal_coveragesinal)

        This function returns for each target t (typically t = self.nominal_coveragesinal) a
        nominal coverage c_cal such that g(c_cal) ≈ t.

        Parameters
        ----------
        iso_model : fitted IsotonicRegression
        targets : array-like
            Desired coverage values in [0, 1].
        grid_size : int
            Resolution of the inversion grid.

        Returns
        -------
        c_calibrated : ndarray, same shape as targets
            Recalibrated nominal coverages.
        """
        grid_c = np.linspace(0.0, 1.0, grid_size)
        grid_emp = iso_model.predict(grid_c)  # g(c) over a dense grid
        c_cal = np.interp(targets, grid_emp, grid_c)
        return np.clip(c_cal, 0.0, 1.0)

    def _orientation_flags(self, dataset: dict) -> tuple:
        orientation_type = dataset.get("orientation_type")
        coil_type = dataset.get("coil_type")
        n_components_full: Optional[int] = None
        n_components_reduced: Optional[int] = None
        x_true = dataset.get("x_true")
        if x_true is not None:
            x_arr = np.asarray(x_true)
            if x_arr.ndim >= 3:
                n_components_full = x_arr.shape[1]
            else:
                n_components_full = 1
            n_components_reduced = n_components_full
        is_fixed = is_fixed_orientation(orientation_type)
        is_free_eeg = is_free_eeg_orientation(
            orientation_type, coil_type, n_components_full
        )
        is_free_meg = is_free_meg_orientation(
            orientation_type, coil_type, n_components_reduced
        )
        return orientation_type, coil_type, is_fixed, is_free_eeg, is_free_meg

    @staticmethod
    def _reshape_free_mean(x_hat: np.ndarray, n_sources: int, n_components: int) -> np.ndarray:
        arr = np.asarray(x_hat, dtype=float)
        if arr.ndim == 3:
            if arr.shape[0] != n_sources or arr.shape[1] != n_components:
                raise ValueError(
                    f"Expected mean shape ({n_sources},{n_components},T); got {arr.shape}"
                )
            return arr
        if arr.ndim != 2:
            raise ValueError("Posterior means must be 2D or 3D arrays.")
        n_times = arr.shape[1]
        expected = n_sources * n_components
        if arr.shape[0] != expected:
            raise ValueError(
                f"Posterior mean first dimension must equal {expected}; got {arr.shape[0]}"
            )
        return arr.reshape(n_components, n_sources, n_times).transpose(1, 0, 2)

    @staticmethod
    def _extract_meg_tangent_basis(q_basis: np.ndarray, n_sources: int) -> np.ndarray:
        basis = np.asarray(q_basis, dtype=float)
        if basis.ndim != 3 or basis.shape[0] != n_sources or basis.shape[1] != 3:
            raise ValueError(
                f"Q_basis must have shape ({n_sources},3,K); got {basis.shape}"
            )
        if basis.shape[2] >= 2:
            return basis[:, :, :2]
        raise ValueError("Q_basis must provide at least two tangent vectors.")

    def calibrate(
        self,
        x_true: Optional[np.ndarray] = None,
        x_hat: Optional[np.ndarray] = None,
        posterior_std: Optional[np.ndarray] = None,
        *,
        train_data: Optional[dict] = None,
        test_data: Optional[dict] = None,
        verbose: bool = False,
        fit: bool = True,
        free_interval_type: str = "full_cov",
    ) -> dict:
        """
        Run the full calibration procedure on the provided posterior statistics.

        Parameters
        ----------
        x_true : np.ndarray, optional
            Ground-truth sources used both for training and evaluation when
            ``train_data`` is not supplied. Shape (n_sources, n_times) or
            (n_sources,).
        x_hat : np.ndarray, optional
            Posterior mean estimates paired with ``x_true``.
        posterior_std : np.ndarray, optional
            Posterior standard deviation paired with ``x_true``.
        train_data : dict, optional
            Dictionary containing ``x_true``, ``x_hat``, and ``posterior_std``
            arrays for the training experiments. When provided, the positional
            arguments are ignored for training.
        test_data : dict, optional
            Dictionary containing ``x_true``, ``x_hat``, and ``posterior_std``
            arrays for the evaluation experiments. If omitted, the training
            data is also used for evaluation.
        verbose : bool, optional
            If True, print progress information.
        fit : bool, optional
            If False, skip isotonic regression fitting and report only the raw
            (pre-calibration) coverage curve on the evaluation data. The
            returned ``post_calibration`` block mirrors the pre-calibration
            results with an identity mapping.

        Returns
        -------
        dict
            Dictionary containing:
                - ``pre_calibration`` and ``post_calibration`` metadata
                  (nominal coverages, empirical coverages, metrics, CI bounds).
                - ``train_empirical_coverages`` for diagnosing how the
                  calibration map behaves on the training split.
                - ``calibration_metric_names`` listing the metrics evaluated.
        """
        if verbose:
            print("=== UNCERTAINTY CALIBRATION (NOMINAL RECALIBRATION) ===")

        if train_data is None and fit:
            if x_true is None or x_hat is None or posterior_std is None:
                raise ValueError(
                    "Provide either 'train_data' or raw arrays x_true, x_hat, posterior_std."
                )
            if x_true.shape != x_hat.shape:
                raise ValueError("x_true and x_hat must have identical shapes.")
            if posterior_std.shape[0] != x_true.shape[0]:
                raise ValueError(
                    "posterior_std must provide one value per source for fixed orientation."
                )
            train_data = {
                "orientation_type": "fixed",
                "coil_type": None,
                "x_true": x_true,
                "x_hat": x_hat,
                "posterior_std": posterior_std,
                "n_sources": x_true.shape[0],
                "n_times": x_true.shape[-1],
            }

        if test_data is not None:
            if test_data.get("x_true") is None:
                raise ValueError("test_data must include 'x_true'.")
            eval_data = test_data
        elif train_data is not None:
            if train_data.get("x_true") is None:
                raise ValueError("train_data must include 'x_true'.")
            eval_data = train_data
        else:
            if x_true is None or x_hat is None or posterior_std is None:
                raise ValueError(
                    "Provide 'test_data' (or raw arrays) when fit=False and no train_data is supplied."
                )
            if x_true.shape != x_hat.shape:
                raise ValueError("x_true and x_hat must have identical shapes.")
            if posterior_std.shape[0] != x_true.shape[0]:
                raise ValueError(
                    "posterior_std must provide one value per source for fixed orientation."
                )
            eval_data = {
                "orientation_type": "fixed",
                "coil_type": None,
                "x_true": x_true,
                "x_hat": x_hat,
                "posterior_std": posterior_std,
                "n_sources": x_true.shape[0],
                "n_times": x_true.shape[-1],
            }

        if verbose:
            if train_data is not None:
                print(
                    f"  train sources = {train_data['n_sources']}, train times = {train_data['n_times']}"
                )
            else:
                print("  train sources = (skipped; fit=False)")
            print(
                f"  eval sources  = {eval_data['n_sources']}, eval times  = {eval_data['n_times']}\n"
            )

        train_data_for_flags = train_data if train_data is not None else eval_data

        (
            train_orientation,
            train_coil,
            train_is_fixed,
            train_is_eeg,
            train_is_meg,
        ) = self._orientation_flags(train_data_for_flags)
        (
            eval_orientation,
            eval_coil,
            eval_is_fixed,
            eval_is_eeg,
            eval_is_meg,
        ) = self._orientation_flags(eval_data)

        if free_interval_type not in {"full_cov", "marginal"}:
            raise ValueError(
                "free_interval_type must be 'full_cov' or 'marginal'. "
                f"Got {free_interval_type!r}."
            )

        if fit:
            if (
                train_is_fixed,
                train_is_eeg,
                train_is_meg,
            ) != (
                eval_is_fixed,
                eval_is_eeg,
                eval_is_meg,
            ):
                raise ValueError(
                    "Train and evaluation datasets must share the same orientation configuration."
                )

        setting_label: str

        if train_is_fixed:
            setting_label = "fixed"
            if "posterior_var" in eval_data:
                eval_var = np.asarray(eval_data["posterior_var"], dtype=float).reshape(-1)
            elif "posterior_std" in eval_data:
                eval_std = np.asarray(eval_data["posterior_std"], dtype=float)
                eval_var = np.square(eval_std.reshape(-1))
            else:
                eval_cov = eval_data.get("posterior_cov")
                if eval_cov is None:
                    raise ValueError(
                        "Fixed-orientation datasets must include one of: posterior_var, posterior_std, posterior_cov."
                    )
                eval_var = self.uncertainty_estimator.posterior_variance_from_cov(eval_cov)

            pre_curve_eval = self.uncertainty_estimator.calibration_curve_intervals_aggregated(
                x_true=eval_data["x_true"],
                x_hat=eval_data["x_hat"],
                posterior_var=eval_var,
            )

            train_curve = None
            if fit:
                if train_data is None:
                    raise ValueError("train_data is required when fit=True.")
                if "posterior_var" in train_data:
                    train_var = np.asarray(train_data["posterior_var"], dtype=float).reshape(-1)
                elif "posterior_std" in train_data:
                    train_std = np.asarray(train_data["posterior_std"], dtype=float)
                    train_var = np.square(train_std.reshape(-1))
                else:
                    train_cov = train_data.get("posterior_cov")
                    if train_cov is None:
                        raise ValueError(
                            "Fixed-orientation training datasets must include one of: posterior_var, posterior_std, posterior_cov."
                        )
                    train_var = self.uncertainty_estimator.posterior_variance_from_cov(train_cov)
                train_curve = self.uncertainty_estimator.calibration_curve_intervals_aggregated(
                    x_true=train_data["x_true"],
                    x_hat=train_data["x_hat"],
                    posterior_var=train_var,
                )

            def eval_empirical(levels):
                return [
                    self.uncertainty_estimator.aggregated_interval_membership(
                        x_true=eval_data["x_true"],
                        x_hat=eval_data["x_hat"],
                        posterior_var=eval_var,
                        nominal_coverage=float(c),
                    )["empirical_coverage"]
                    for c in levels
                ]

        elif train_is_eeg:
            setting_label = "eeg_free"
            n_eval = int(eval_data.get("n_sources") or eval_data["x_true"].shape[0])
            eval_true = self._reshape_free_mean(eval_data["x_true"], n_eval, 3)
            eval_hat = self._reshape_free_mean(eval_data["x_hat"], n_eval, 3)
            if "posterior_cov_blocks" in eval_data:
                eval_cov = np.asarray(eval_data["posterior_cov_blocks"], dtype=float)
            else:
                eval_cov = np.asarray(eval_data["posterior_cov"], dtype=float)

            if free_interval_type == "marginal":
                pre_curve_eval = self.uncertainty_estimator.calibration_curve_componentwise_eeg_free_aggregated(
                    x_true=eval_true,
                    x_hat=eval_hat,
                    posterior_uncert=eval_cov,
                )
            else:
                pre_curve_eval = self.uncertainty_estimator.calibration_curve_ellipsoid_eeg_free_aggregated(
                    x_true=eval_true,
                    x_hat=eval_hat,
                    posterior_cov=eval_cov,
                )

            train_curve = None
            if fit:
                if train_data is None:
                    raise ValueError("train_data is required when fit=True.")
                n_train = int(train_data.get("n_sources") or train_data["x_true"].shape[0])
                train_true = self._reshape_free_mean(train_data["x_true"], n_train, 3)
                train_hat = self._reshape_free_mean(train_data["x_hat"], n_train, 3)
                if "posterior_cov_blocks" in train_data:
                    train_cov = np.asarray(train_data["posterior_cov_blocks"], dtype=float)
                else:
                    train_cov = np.asarray(train_data["posterior_cov"], dtype=float)

                if free_interval_type == "marginal":
                    train_curve = self.uncertainty_estimator.calibration_curve_componentwise_eeg_free_aggregated(
                        x_true=train_true,
                        x_hat=train_hat,
                        posterior_uncert=train_cov,
                    )
                else:
                    train_curve = self.uncertainty_estimator.calibration_curve_ellipsoid_eeg_free_aggregated(
                        x_true=train_true,
                        x_hat=train_hat,
                        posterior_cov=train_cov,
                    )

            def eval_empirical(levels):
                if free_interval_type == "marginal":
                    return [
                        self.uncertainty_estimator.aggregated_componentwise_interval_membership_free(
                            x_true=eval_true,
                            x_hat=eval_hat,
                            posterior_uncert=eval_cov,
                            nominal_coverage=float(c),
                            n_orient=3,
                        )["empirical_coverage"]
                        for c in levels
                    ]

                return [
                    self.uncertainty_estimator.aggregated_ellipsoid_membership_eeg_free(
                        x_true=eval_true,
                        x_hat=eval_hat,
                        posterior_cov=eval_cov,
                        nominal_coverage=float(c),
                    )["empirical_coverage"]
                    for c in levels
                ]


        elif train_is_meg:
            setting_label = "meg_free"
            eval_basis = eval_data.get("Q_basis")
            if eval_basis is None:
                raise ValueError("Free-orientation MEG datasets must include Q_basis.")
            n_eval = int(eval_data.get("n_sources") or eval_data["x_true"].shape[0])

            eval_V_tan = self._extract_meg_tangent_basis(eval_basis, n_eval)

            eval_hat_2d = self._reshape_free_mean(eval_data["x_hat"], n_eval, 2)
            eval_true_2d = self._reshape_free_mean(eval_data["x_true"], n_eval, 2)

            if "posterior_cov_blocks" in eval_data:
                eval_cov = np.asarray(eval_data["posterior_cov_blocks"], dtype=float)
            else:
                eval_cov = np.asarray(eval_data["posterior_cov"], dtype=float)

            if free_interval_type == "marginal":
                pre_curve_eval = self.uncertainty_estimator.calibration_curve_componentwise_meg_free_aggregated(
                    x_true_2d=eval_true_2d,
                    x_hat_2d=eval_hat_2d,
                    posterior_uncert_2d=eval_cov,
                )
            else:
                eval_true_3d = lift_reduced_sources_to_3d(eval_true_2d, eval_V_tan)
                pre_curve_eval = self.uncertainty_estimator.calibration_curve_ellipse_meg_free_aggregated(
                    x_true_3d=eval_true_3d,
                    x_hat_2d=eval_hat_2d,
                    posterior_cov_2d=eval_cov,
                    V_tan=eval_V_tan,
                )

            train_curve = None
            if fit:
                if train_data is None:
                    raise ValueError("train_data is required when fit=True.")
                train_basis = train_data.get("Q_basis")
                if train_basis is None:
                    raise ValueError("Free-orientation MEG training datasets must include Q_basis.")
                n_train = int(train_data.get("n_sources") or train_data["x_true"].shape[0])
                train_V_tan = self._extract_meg_tangent_basis(train_basis, n_train)
                train_hat_2d = self._reshape_free_mean(train_data["x_hat"], n_train, 2)
                train_true_2d = self._reshape_free_mean(train_data["x_true"], n_train, 2)
                if "posterior_cov_blocks" in train_data:
                    train_cov = np.asarray(train_data["posterior_cov_blocks"], dtype=float)
                else:
                    train_cov = np.asarray(train_data["posterior_cov"], dtype=float)

                if free_interval_type == "marginal":
                    train_curve = self.uncertainty_estimator.calibration_curve_componentwise_meg_free_aggregated(
                        x_true_2d=train_true_2d,
                        x_hat_2d=train_hat_2d,
                        posterior_uncert_2d=train_cov,
                    )
                else:
                    train_true_3d = lift_reduced_sources_to_3d(train_true_2d, train_V_tan)
                    train_curve = self.uncertainty_estimator.calibration_curve_ellipse_meg_free_aggregated(
                        x_true_3d=train_true_3d,
                        x_hat_2d=train_hat_2d,
                        posterior_cov_2d=train_cov,
                        V_tan=train_V_tan,
                    )

            def eval_empirical(levels):
                if free_interval_type == "marginal":
                    return [
                        self.uncertainty_estimator.aggregated_componentwise_interval_membership_free(
                            x_true=eval_true_2d,
                            x_hat=eval_hat_2d,
                            posterior_uncert=eval_cov,
                            nominal_coverage=float(c),
                            n_orient=2,
                        )["empirical_coverage"]
                        for c in levels
                    ]

                return [
                    self.uncertainty_estimator.aggregated_ellipse_membership_meg_free(
                        x_true_3d=eval_true_3d,
                        x_hat_2d=eval_hat_2d,
                        posterior_cov_2d=eval_cov,
                        nominal_coverage=float(c),
                        V_tan=eval_V_tan,
                    )["empirical_coverage"]
                    for c in levels
                ]


        else:
            raise ValueError(
                f"Unsupported orientation/coil combination: {train_orientation}, {train_coil}"
            )

        empirical_before_eval = pre_curve_eval["empirical_coverages"]
        calibration_metrics_before = self.metric_evaluator.calibration_metrics_4(
            np.asarray(pre_curve_eval["nominal_coverages"], dtype=float),
            np.asarray(empirical_before_eval, dtype=float),
        )

        train_meta = train_data if train_data is not None else eval_data
        split_metadata = {
            'train_n_sources': train_meta.get('n_sources'),
            'train_n_times': train_meta.get('n_times'),
            'eval_n_sources': eval_data.get('n_sources'),
            'eval_n_times': eval_data.get('n_times'),
            'uses_separate_eval_split': test_data is not None,
            'orientation_type': train_meta.get('orientation_type'),
            'coil_type': train_meta.get('coil_type'),
            'setting': setting_label,
            'interval_type': pre_curve_eval.get('interval_type'),
            'fit': bool(fit),
        }

        pre_results = {
            'nominal_coverages': pre_curve_eval['nominal_coverages'],
            'empirical_coverages': empirical_before_eval,
            'calibration_metrics': calibration_metrics_before,
            'ci_lowers': pre_curve_eval.get('ci_lowers'),
            'ci_uppers': pre_curve_eval.get('ci_uppers'),
            'ci_counts': pre_curve_eval.get('ci_counts'),
            'interval_type': pre_curve_eval.get('interval_type'),
            'split_metadata': split_metadata,
        }

        if not fit:
            identity_levels = np.asarray(pre_curve_eval["nominal_coverages"], dtype=float)
            post_results = {
                'nominal_coverages': identity_levels,
                'empirical_coverages': np.asarray(empirical_before_eval, dtype=float),
                'calibration_metrics': calibration_metrics_before,
                'recalibrated_nominal_coverages': identity_levels,
                'interval_type': pre_curve_eval.get('interval_type'),
            'split_metadata': split_metadata,
        }
            return {
                'pre_calibration': pre_results,
                'post_calibration': post_results,
                'train_empirical_coverages': None,
                'calibration_metric_names': tuple(
                    getattr(self.metric_evaluator, "calibration_metrics", tuple())
                ),
            }

        if train_curve is None:
            raise RuntimeError("Internal error: train_curve missing despite fit=True.")
        empirical_train = train_curve["empirical_coverages"]

        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(self.nominal_coverages, empirical_train)
        c_recalibrated = self._invert_isotonic(iso, self.nominal_coverages)

        # ------------------------------------------------------------------
        # 3) Evaluate after calibration on evaluation split
        # ------------------------------------------------------------------
        empirical_after_eval = np.asarray(eval_empirical(c_recalibrated), dtype=float)

        calibration_metrics_after = self.metric_evaluator.calibration_metrics_4(
            self.nominal_coverages,
            empirical_after_eval,
        )

        # ------------------------------------------------------------------
        # 4) Final mapping nominal -> recalibrated nominal
        # ------------------------------------------------------------------
        self.calibration_model = IsotonicRegression(
            increasing=True,
            out_of_bounds="clip",
        )
        self.calibration_model.fit(self.nominal_coverages, c_recalibrated)
        self.is_calibrated = True

        post_results = {
            'nominal_coverages': self.nominal_coverages,
            'empirical_coverages': empirical_after_eval,
            'calibration_metrics': calibration_metrics_after,
            'recalibrated_nominal_coverages': c_recalibrated,
            'split_metadata': split_metadata,
        }

        results = {
            'pre_calibration': pre_results,
            'post_calibration': post_results,
            'train_empirical_coverages': empirical_train,
            'calibration_metric_names': tuple(
                getattr(self.metric_evaluator, "calibration_metrics", tuple())
            ),
        }

        return results

    def fit_mapping(
        self,
        *,
        train_data: dict,
        free_interval_type: str = "full_cov",
    ) -> dict:
        """
        Fit an isotonic nominal-recalibration mapping from `train_data` only.

        This is intended for experiments like "post_fixed", where you fit once
        (e.g., at the default setting) and reuse the same mapping over many
        evaluation datasets (e.g., an SNR/NNZ sweep).

        Returns
        -------
        dict with:
          - train_curve: pre-calibration curve on the training split
          - recalibrated_nominal_coverages: c_recalibrated on the nominal grid
        """
        if train_data.get("x_true") is None:
            raise ValueError("train_data must include 'x_true'.")
        if free_interval_type not in {"full_cov", "marginal"}:
            raise ValueError(
                "free_interval_type must be 'full_cov' or 'marginal'. "
                f"Got {free_interval_type!r}."
            )

        (
            train_orientation,
            train_coil,
            train_is_fixed,
            train_is_eeg,
            train_is_meg,
        ) = self._orientation_flags(train_data)

        if train_is_fixed:
            if "posterior_var" in train_data:
                train_var = np.asarray(train_data["posterior_var"], dtype=float).reshape(-1)
            elif "posterior_std" in train_data:
                train_std = np.asarray(train_data["posterior_std"], dtype=float)
                train_var = np.square(train_std.reshape(-1))
            else:
                train_cov = train_data.get("posterior_cov")
                if train_cov is None:
                    raise ValueError(
                        "Fixed-orientation training datasets must include one of: posterior_var, posterior_std, posterior_cov."
                    )
                train_var = self.uncertainty_estimator.posterior_variance_from_cov(train_cov)

            train_curve = self.uncertainty_estimator.calibration_curve_intervals_aggregated(
                x_true=train_data["x_true"],
                x_hat=train_data["x_hat"],
                posterior_var=train_var,
            )

        elif train_is_eeg:
            n_train = int(train_data.get("n_sources") or train_data["x_true"].shape[0])
            train_true = self._reshape_free_mean(train_data["x_true"], n_train, 3)
            train_hat = self._reshape_free_mean(train_data["x_hat"], n_train, 3)
            if "posterior_cov_blocks" in train_data:
                train_cov = np.asarray(train_data["posterior_cov_blocks"], dtype=float)
            else:
                train_cov = np.asarray(train_data["posterior_cov"], dtype=float)

            if free_interval_type == "marginal":
                train_curve = self.uncertainty_estimator.calibration_curve_componentwise_eeg_free_aggregated(
                    x_true=train_true,
                    x_hat=train_hat,
                    posterior_uncert=train_cov,
                )
            else:
                train_curve = self.uncertainty_estimator.calibration_curve_ellipsoid_eeg_free_aggregated(
                    x_true=train_true,
                    x_hat=train_hat,
                    posterior_cov=train_cov,
                )

        elif train_is_meg:
            train_basis = train_data.get("Q_basis")
            if train_basis is None:
                raise ValueError("Free-orientation MEG training datasets must include Q_basis.")
            n_train = int(train_data.get("n_sources") or train_data["x_true"].shape[0])
            train_V_tan = self._extract_meg_tangent_basis(train_basis, n_train)
            train_hat_2d = self._reshape_free_mean(train_data["x_hat"], n_train, 2)
            train_true_2d = self._reshape_free_mean(train_data["x_true"], n_train, 2)
            if "posterior_cov_blocks" in train_data:
                train_cov = np.asarray(train_data["posterior_cov_blocks"], dtype=float)
            else:
                train_cov = np.asarray(train_data["posterior_cov"], dtype=float)

            if free_interval_type == "marginal":
                train_curve = self.uncertainty_estimator.calibration_curve_componentwise_meg_free_aggregated(
                    x_true_2d=train_true_2d,
                    x_hat_2d=train_hat_2d,
                    posterior_uncert_2d=train_cov,
                )
            else:
                train_true_3d = lift_reduced_sources_to_3d(train_true_2d, train_V_tan)
                train_curve = self.uncertainty_estimator.calibration_curve_ellipse_meg_free_aggregated(
                    x_true_3d=train_true_3d,
                    x_hat_2d=train_hat_2d,
                    posterior_cov_2d=train_cov,
                    V_tan=train_V_tan,
                )

        else:
            raise ValueError(
                f"Unsupported orientation/coil combination: {train_orientation}, {train_coil}"
            )

        empirical_train = np.asarray(train_curve["empirical_coverages"], dtype=float)
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(self.nominal_coverages, empirical_train)
        c_recalibrated = self._invert_isotonic(iso, self.nominal_coverages)

        self.calibration_model = IsotonicRegression(
            increasing=True,
            out_of_bounds="clip",
        )
        self.calibration_model.fit(self.nominal_coverages, c_recalibrated)
        self.is_calibrated = True

        return {
            "train_curve": train_curve,
            "recalibrated_nominal_coverages": c_recalibrated,
        }

    def evaluate_with_mapping(
        self,
        *,
        test_data: dict,
        free_interval_type: str = "full_cov",
    ) -> dict:
        """
        Evaluate pre- and post-calibration curves on `test_data` using an
        already-fitted mapping (from `fit_mapping`).
        """
        if not self.is_calibrated:
            raise ValueError("Call fit_mapping(...) before evaluate_with_mapping(...).")
        if test_data.get("x_true") is None:
            raise ValueError("test_data must include 'x_true'.")
        if free_interval_type not in {"full_cov", "marginal"}:
            raise ValueError(
                "free_interval_type must be 'full_cov' or 'marginal'. "
                f"Got {free_interval_type!r}."
            )

        (
            orientation,
            coil_type,
            is_fixed,
            is_eeg,
            is_meg,
        ) = self._orientation_flags(test_data)

        setting_label: str
        if is_fixed:
            setting_label = "fixed"
            if "posterior_var" in test_data:
                eval_var = np.asarray(test_data["posterior_var"], dtype=float).reshape(-1)
            elif "posterior_std" in test_data:
                eval_std = np.asarray(test_data["posterior_std"], dtype=float)
                eval_var = np.square(eval_std.reshape(-1))
            else:
                eval_cov = test_data.get("posterior_cov")
                if eval_cov is None:
                    raise ValueError(
                        "Fixed-orientation datasets must include one of: posterior_var, posterior_std, posterior_cov."
                    )
                eval_var = self.uncertainty_estimator.posterior_variance_from_cov(eval_cov)

            pre_curve = self.uncertainty_estimator.calibration_curve_intervals_aggregated(
                x_true=test_data["x_true"],
                x_hat=test_data["x_hat"],
                posterior_var=eval_var,
            )

            def empirical(levels):
                return [
                    self.uncertainty_estimator.aggregated_interval_membership(
                        x_true=test_data["x_true"],
                        x_hat=test_data["x_hat"],
                        posterior_var=eval_var,
                        nominal_coverage=float(c),
                    )["empirical_coverage"]
                    for c in levels
                ]

        elif is_eeg:
            setting_label = "eeg_free"
            n_eval = int(test_data.get("n_sources") or test_data["x_true"].shape[0])
            eval_true = self._reshape_free_mean(test_data["x_true"], n_eval, 3)
            eval_hat = self._reshape_free_mean(test_data["x_hat"], n_eval, 3)
            if "posterior_cov_blocks" in test_data:
                eval_cov = np.asarray(test_data["posterior_cov_blocks"], dtype=float)
            else:
                eval_cov = np.asarray(test_data["posterior_cov"], dtype=float)

            if free_interval_type == "marginal":
                pre_curve = self.uncertainty_estimator.calibration_curve_componentwise_eeg_free_aggregated(
                    x_true=eval_true,
                    x_hat=eval_hat,
                    posterior_uncert=eval_cov,
                )

                def empirical(levels):
                    return [
                        self.uncertainty_estimator.aggregated_componentwise_interval_membership_free(
                            x_true=eval_true,
                            x_hat=eval_hat,
                            posterior_uncert=eval_cov,
                            nominal_coverage=float(c),
                            n_orient=3,
                        )["empirical_coverage"]
                        for c in levels
                    ]
            else:
                pre_curve = self.uncertainty_estimator.calibration_curve_ellipsoid_eeg_free_aggregated(
                    x_true=eval_true,
                    x_hat=eval_hat,
                    posterior_cov=eval_cov,
                )

                def empirical(levels):
                    return [
                        self.uncertainty_estimator.aggregated_ellipsoid_membership_eeg_free(
                            x_true=eval_true,
                            x_hat=eval_hat,
                            posterior_cov=eval_cov,
                            nominal_coverage=float(c),
                        )["empirical_coverage"]
                        for c in levels
                    ]

        elif is_meg:
            setting_label = "meg_free"
            eval_basis = test_data.get("Q_basis")
            if eval_basis is None:
                raise ValueError("Free-orientation MEG datasets must include Q_basis.")
            n_eval = int(test_data.get("n_sources") or test_data["x_true"].shape[0])
            eval_V_tan = self._extract_meg_tangent_basis(eval_basis, n_eval)
            eval_hat_2d = self._reshape_free_mean(test_data["x_hat"], n_eval, 2)
            eval_true_2d = self._reshape_free_mean(test_data["x_true"], n_eval, 2)
            if "posterior_cov_blocks" in test_data:
                eval_cov = np.asarray(test_data["posterior_cov_blocks"], dtype=float)
            else:
                eval_cov = np.asarray(test_data["posterior_cov"], dtype=float)

            if free_interval_type == "marginal":
                pre_curve = self.uncertainty_estimator.calibration_curve_componentwise_meg_free_aggregated(
                    x_true_2d=eval_true_2d,
                    x_hat_2d=eval_hat_2d,
                    posterior_uncert_2d=eval_cov,
                )

                def empirical(levels):
                    return [
                        self.uncertainty_estimator.aggregated_componentwise_interval_membership_free(
                            x_true=eval_true_2d,
                            x_hat=eval_hat_2d,
                            posterior_uncert=eval_cov,
                            nominal_coverage=float(c),
                            n_orient=2,
                        )["empirical_coverage"]
                        for c in levels
                    ]
            else:
                eval_true_3d = lift_reduced_sources_to_3d(eval_true_2d, eval_V_tan)
                pre_curve = self.uncertainty_estimator.calibration_curve_ellipse_meg_free_aggregated(
                    x_true_3d=eval_true_3d,
                    x_hat_2d=eval_hat_2d,
                    posterior_cov_2d=eval_cov,
                    V_tan=eval_V_tan,
                )

                def empirical(levels):
                    return [
                        self.uncertainty_estimator.aggregated_ellipse_membership_meg_free(
                            x_true_3d=eval_true_3d,
                            x_hat_2d=eval_hat_2d,
                            posterior_cov_2d=eval_cov,
                            nominal_coverage=float(c),
                            V_tan=eval_V_tan,
                        )["empirical_coverage"]
                        for c in levels
                    ]

        else:
            raise ValueError(f"Unsupported orientation/coil combination: {orientation}, {coil_type}")

        empirical_before = np.asarray(pre_curve["empirical_coverages"], dtype=float)
        metrics_before = self.metric_evaluator.calibration_metrics_4(
            np.asarray(pre_curve["nominal_coverages"], dtype=float),
            empirical_before,
        )

        c_recal = self.apply_calibration(self.nominal_coverages)
        empirical_after = np.asarray(empirical(c_recal), dtype=float)
        metrics_after = self.metric_evaluator.calibration_metrics_4(
            self.nominal_coverages,
            empirical_after,
        )

        split_metadata = {
            "train_n_sources": None,
            "train_n_times": None,
            "eval_n_sources": test_data.get("n_sources"),
            "eval_n_times": test_data.get("n_times"),
            "uses_separate_eval_split": True,
            "orientation_type": test_data.get("orientation_type"),
            "coil_type": test_data.get("coil_type"),
            "setting": setting_label,
            "fit": True,
            "interval_type": pre_curve.get("interval_type"),
        }

        pre_results = {
            "nominal_coverages": pre_curve["nominal_coverages"],
            "empirical_coverages": empirical_before,
            "calibration_metrics": metrics_before,
            "ci_lowers": pre_curve.get("ci_lowers"),
            "ci_uppers": pre_curve.get("ci_uppers"),
            "ci_counts": pre_curve.get("ci_counts"),
            "interval_type": pre_curve.get("interval_type"),
            "split_metadata": split_metadata,
        }
        post_results = {
            "nominal_coverages": self.nominal_coverages,
            "empirical_coverages": empirical_after,
            "calibration_metrics": metrics_after,
            "recalibrated_nominal_coverages": c_recal,
            "interval_type": pre_curve.get("interval_type"),
            "split_metadata": split_metadata,
        }

        return {
            "pre_calibration": pre_results,
            "post_calibration": post_results,
            "train_empirical_coverages": None,
            "calibration_metric_names": tuple(
                getattr(self.metric_evaluator, "calibration_metrics", tuple())
            ),
        }

    def apply_calibration(self, nominal_levels: np.ndarray) -> np.ndarray:
        """
        Apply learned calibration to new nominal coverage levels.

        Parameters
        ----------
        nominal_levels : array-like
            Original nominal coverage levels.

        Returns
        -------
        nominal_levels_recalibrated : ndarray
            Recalibrated nominal coverage levels to use when constructing
            credible intervals (i.e. in place of the original nominal levels).
        """
        if not self.is_calibrated:
            raise ValueError("Calibrator must be fitted first using calibrate().")
        nominal_levels = np.asarray(nominal_levels, dtype=float)
        return self.calibration_model.predict(nominal_levels)
