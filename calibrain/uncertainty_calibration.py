from typing import Optional

from sklearn.isotonic import IsotonicRegression
import numpy as np
from scipy.stats import norm
from calibrain import UncertaintyEstimator
from calibrain import MetricEvaluator

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
      - UncertaintyEstimator.compute_empirical_coverage() to evaluate
        empirical coverage for any nominal coverage level c.
      - MetricEvaluator.evaluate_metrics(which="calibration") to summarize
        AAD, ASD, etc. using the configured calibration metrics.
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

    # ----------------------------------------------------------------------
    # Auxiliary functions
    # ----------------------------------------------------------------------
    def _ensure_z_score(self, c: float):
        """
        Ensure that the UncertaintyEstimator has a z-score entry for coverage c.

        Your UncertaintyEstimator precomputes z-scores only for the original
        nominal grid. After calibration we will generate new coverage levels
        c_cal(c) which may not lie exactly on this grid, so we extend
        `z_scores` when needed.
        """
        if c in self.uncertainty_estimator.z_scores:
            return

        if c <= 0.0:
            self.uncertainty_estimator.z_scores[c] = 0.0
        elif c >= 1.0:
            self.uncertainty_estimator.z_scores[c] = np.inf
        else:
            self.uncertainty_estimator.z_scores[c] = norm.ppf((1.0 + c) / 2.0)

    def _empirical_curve_for_levels(
        self,
        x_true: np.ndarray,
        x_hat: np.ndarray,
        posterior_var: np.ndarray,
        coverages: np.ndarray,
    ) -> np.ndarray:
        """
        Compute empirical coverage for each coverage level in `coverages`
        using the UncertaintyEstimator.

        Parameters
        ----------
        x_true, x_hat, posterior_var : arrays, shape (n_sources, n_times) or (n_sources,)
        coverages : array-like
            Coverage levels c in [0, 1].

        Returns
        -------
        empirical_coverages : ndarray, shape (len(coverages),)
            Empirical coverage for each c.
        """
        empirical = []
        for c in coverages:
            c = float(c)
            self._ensure_z_score(c)
            _, _, _, cov = self.uncertainty_estimator.compute_empirical_coverage(
                x_true=x_true,
                x_hat=x_hat,
                posterior_var=posterior_var,
                nominal_coverage=c,
            )
            empirical.append(cov)
        return np.array(empirical)

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

    # ----------------------------------------------------------------------
    # Main calibration routine
    # ----------------------------------------------------------------------
    def _prepare_dataset(self,
                         *,
                         x_true: np.ndarray,
                         x_hat: np.ndarray,
                         posterior_std: np.ndarray) -> dict:
        """Ensure arrays are numpy ndarrays with consistent 2D shapes."""
        if x_true is None or x_hat is None or posterior_std is None:
            raise ValueError("x_true, x_hat and posterior_std must be provided.")

        x_true = np.asarray(x_true)
        x_hat = np.asarray(x_hat)
        posterior_std = np.asarray(posterior_std)

        if x_true.ndim == 1:
            x_true = x_true[:, np.newaxis]
        if x_hat.ndim == 1:
            x_hat = x_hat[:, np.newaxis]
        if posterior_std.ndim == 1:
            posterior_std = posterior_std[:, np.newaxis]

        posterior_var = np.square(posterior_std)

        if x_true.shape != x_hat.shape or x_true.shape != posterior_std.shape:
            raise ValueError(
                "x_true, x_hat, and posterior_std must have the same shape after reshaping."
            )

        return {
            "x_true": x_true,
            "x_hat": x_hat,
            "posterior_std": posterior_std,
            "posterior_var": posterior_var,
            "n_sources": x_true.shape[0],
            "n_times": x_true.shape[-1],
        }

    def calibrate(
        self,
        x_true: Optional[np.ndarray] = None,
        x_hat: Optional[np.ndarray] = None,
        posterior_std: Optional[np.ndarray] = None,
        *,
        train_data: Optional[dict] = None,
        test_data: Optional[dict] = None,
        verbose: bool = False,
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

        if train_data is None:
            train_data = {
                "x_true": x_true,
                "x_hat": x_hat,
                "posterior_std": posterior_std,
            }

        train = self._prepare_dataset(**train_data)
        eval_payload = test_data or train_data
        eval_data = self._prepare_dataset(**eval_payload)

        if verbose:
            print(f"  train sources = {train['n_sources']}, train times = {train['n_times']}")
            print(f"  eval sources  = {eval_data['n_sources']}, eval times  = {eval_data['n_times']}\n")

        # ------------------------------------------------------------------
        # 1) Pre-calibration baseline on evaluation split
        # ------------------------------------------------------------------
        pre_curve_eval = self.uncertainty_estimator.get_calibration_curve(
            x_true=eval_data['x_true'],
            x_hat=eval_data['x_hat'],
            posterior_var=eval_data['posterior_var'],
        )
        empirical_before_eval = pre_curve_eval['empirical_coverages']
        calibration_metrics_before = self.metric_evaluator.evaluate_metrics(
            which="calibration",
            empirical_coverages=empirical_before_eval,
        )

        # ------------------------------------------------------------------
        # 2) Learn isotonic mapping on training experiments
        # ------------------------------------------------------------------
        empirical_train = self._empirical_curve_for_levels(
            x_true=train['x_true'],
            x_hat=train['x_hat'],
            posterior_var=train['posterior_var'],
            coverages=self.nominal_coverages,
        )

        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(self.nominal_coverages, empirical_train)
        c_recalibrated = self._invert_isotonic(iso, self.nominal_coverages)

        # ------------------------------------------------------------------
        # 3) Evaluate after calibration on evaluation split
        # ------------------------------------------------------------------
        empirical_after_eval = self._empirical_curve_for_levels(
            x_true=eval_data['x_true'],
            x_hat=eval_data['x_hat'],
            posterior_var=eval_data['posterior_var'],
            coverages=c_recalibrated,
        )

        calibration_metrics_after = self.metric_evaluator.evaluate_metrics(
            which="calibration",
            empirical_coverages=empirical_after_eval,
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

        split_metadata = {
            'train_n_sources': train['n_sources'],
            'train_n_times': train['n_times'],
            'eval_n_sources': eval_data['n_sources'],
            'eval_n_times': eval_data['n_times'],
            'uses_separate_eval_split': test_data is not None,
        }

        pre_results = {
            'nominal_coverages': pre_curve_eval['nominal_coverages'],
            'empirical_coverages': empirical_before_eval,
            'calibration_metrics': calibration_metrics_before,
            'ci_lowers': pre_curve_eval.get('ci_lowers'),
            'ci_uppers': pre_curve_eval.get('ci_uppers'),
            'ci_counts': pre_curve_eval.get('ci_counts'),
            'split_metadata': split_metadata,
        }
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

    # ----------------------------------------------------------------------
    # Public API: apply + summary
    # ----------------------------------------------------------------------
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
