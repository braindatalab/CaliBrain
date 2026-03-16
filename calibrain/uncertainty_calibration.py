from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
import numpy as np
from scipy.stats import norm
from calibrain import UncertaintyEstimator
from calibrain import MetricEvaluator

# ============================================================================
# UNCERTAINTY CALIBRATOR (NOMINAL RE-CALIBRATION + CV)
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
      - Source-wise K-fold cross-validation to avoid overfitting.

    Conceptual model
    ----------------
    For each nominal coverage level c (e.g. 0.5, 0.9):

        g(c) = empirical coverage when building CIs with nominal coverage c.

    On the training sources in each fold, we learn g(c) via isotonic regression,
    then we numerically invert this curve to obtain a re-calibrated nominal
    coverage c_cal(c) ≈ g^{-1}(c). On the test sources we then build intervals
    using c_cal(c) and measure empirical coverage again.

    The final goal is that after recalibration the empirical coverage is
    closer to the original nominal coverage grid defined in your
    UncertaintyEstimator (self.nominal_coverages).
    """

    def __init__(self,
                 uncertainty_estimator: UncertaintyEstimator,
                 metric_evaluator: MetricEvaluator,
                 n_folds: int = 5,
                 random_state: int = 42):
        """
        Parameters
        ----------
        uncertainty_estimator : UncertaintyEstimator
            Your existing UncertaintyEstimator instance.
        metric_evaluator : MetricEvaluator
            Your existing MetricEvaluator instance.
        n_folds : int
            Number of cross-validation folds (source-wise).
        random_state : int
            Random seed for reproducible KFold splits.
        """
        self.uncertainty_estimator = uncertainty_estimator
        self.metric_evaluator = metric_evaluator
        self.n_folds = n_folds
        self.random_state = random_state

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
    def calibrate(
        self,
        x_true: np.ndarray,
        x_hat: np.ndarray,
        posterior_std: np.ndarray,
        verbose: bool = False,
    ) -> dict:
        """
        Run the full calibration procedure on the provided posterior statistics.

        Parameters
        ----------
        x_true : np.ndarray
            Ground-truth sources with shape (n_sources, n_times) or (n_sources,).
        x_hat : np.ndarray
            Posterior mean estimates with the same shape as ``x_true``.
        posterior_std : np.ndarray
            Posterior standard deviation (same shape as ``x_true``) used to
            build nominal credible intervals.
        verbose : bool, optional
            If True, print progress information.

        Returns
        -------
        dict
            Dictionary containing:
                - ``pre_calibration`` and ``post_calibration`` metadata
                  (nominal coverages, empirical coverages, metrics, CI bounds).
                - ``fold_results`` with per-fold diagnostics.
                - ``calibration_metric_names`` listing the metrics evaluated.
        """
        if verbose:
            print("=== UNCERTAINTY CALIBRATION (NOMINAL RECALIBRATION + CV) ===")

        # Make sure we always have 2D (n_sources, n_times)
        if x_true.ndim == 1:
            x_true = x_true[:, np.newaxis]
        if x_hat.ndim == 1:
            x_hat = x_hat[:, np.newaxis]
        if posterior_std.ndim == 1:
            posterior_std = posterior_std[:, np.newaxis]

        posterior_var = np.square(posterior_std)
        n_sources, n_times = x_hat.shape[0], x_hat.shape[-1]
        K_levels = len(self.nominal_coverages)
        if verbose:
            print(f"  n_sources = {n_sources}, n_times = {n_times}")
            print(f"  n_levels  = {K_levels}, n_folds = {self.n_folds}\n")

        # ------------------------------------------------------------------
        # 1) Global pre-calibration baseline (using all sources & times)
        # ------------------------------------------------------------------
        pre_curve = self.uncertainty_estimator.get_calibration_curve(
            x_true=x_true,
            x_hat=x_hat,
            posterior_var=posterior_var,
        )
        empirical_before_global = pre_curve['empirical_coverages']
        calibration_metrics_before = self.metric_evaluator.evaluate_metrics(
            which="calibration",
            empirical_coverages=empirical_before_global,
        )

        # ------------------------------------------------------------------
        # 2) Source-wise K-fold CV: learn nominal recalibration
        # ------------------------------------------------------------------
        kf = KFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state,
        )
        idx = np.arange(n_sources)

        empirical_before_folds = np.zeros((self.n_folds, K_levels))
        empirical_after_folds = np.zeros((self.n_folds, K_levels))
        c_recal_folds = np.zeros((self.n_folds, K_levels))

        fold_results = []

        if verbose:
            print("  Running K-fold source-wise calibration:")
        for fold, (train_idx, test_idx) in enumerate(kf.split(idx)):
            if verbose:
                print(f"    Fold {fold+1}: {len(train_idx)} train, {len(test_idx)} test")

            x_true_train = x_true[train_idx]
            x_hat_train = x_hat[train_idx]
            var_train = posterior_var[train_idx]

            x_true_test = x_true[test_idx]
            x_hat_test = x_hat[test_idx]
            var_test = posterior_var[test_idx]

            # 2.1 Forward calibration curve on training sources: g(c) = empirical(c)
            empirical_train = self._empirical_curve_for_levels(
                x_true=x_true_train,
                x_hat=x_hat_train,
                posterior_var=var_train,
                coverages=self.nominal_coverages,
            )

            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            iso.fit(self.nominal_coverages, empirical_train)  # g: nominal -> empirical

            # 2.2 Invert g to obtain recalibrated nominal coverages
            c_recal = self._invert_isotonic(iso, self.nominal_coverages)

            # 2.3 Test empirical coverage BEFORE calibration on test sources
            empirical_test_before = self._empirical_curve_for_levels(
                x_true=x_true_test,
                x_hat=x_hat_test,
                posterior_var=var_test,
                coverages=self.nominal_coverages,  # original nominal
            )

            # 2.4 Test empirical coverage AFTER calibration (using c_recal)
            empirical_test_after = self._empirical_curve_for_levels(
                x_true=x_true_test,
                x_hat=x_hat_test,
                posterior_var=var_test,
                coverages=c_recal,  # recalibrated nominal used internally
            )

            empirical_before_folds[fold] = empirical_test_before
            empirical_after_folds[fold] = empirical_test_after
            c_recal_folds[fold] = c_recal

            fold_results.append({
                'fold': fold,
                'train_empirical': empirical_train,
                'c_recalibrated': c_recal,
                'test_empirical_before': empirical_test_before,
                'test_empirical_after': empirical_test_after,
                'n_train': len(train_idx),
                'n_test': len(test_idx),
            })

        # ------------------------------------------------------------------
        # 3) Aggregate across folds
        # ------------------------------------------------------------------
        empirical_after = empirical_after_folds.mean(axis=0)
        c_recal_final = c_recal_folds.mean(axis=0)

        # Metrics after calibration (relative to original nominal c)
        calibration_metrics_after = self.metric_evaluator.evaluate_metrics(
            which="calibration",
            empirical_coverages=empirical_after,
        )

        # ------------------------------------------------------------------
        # 4) Final global mapping nominal -> recalibrated nominal
        # ------------------------------------------------------------------
        self.calibration_model = IsotonicRegression(
            increasing=True, out_of_bounds="clip"
        )
        self.calibration_model.fit(self.nominal_coverages, c_recal_final)
        self.is_calibrated = True

        pre_results = {
            'nominal_coverages': pre_curve['nominal_coverages'],
            'empirical_coverages': empirical_before_global,
            'calibration_metrics': calibration_metrics_before,
            'ci_lowers': pre_curve.get('ci_lowers'),
            'ci_uppers': pre_curve.get('ci_uppers'),
            'ci_counts': pre_curve.get('ci_counts'),
        }
        post_results = {
            'nominal_coverages': self.nominal_coverages,
            'empirical_coverages': empirical_after,
            'calibration_metrics': calibration_metrics_after,
            'recalibrated_nominal_coverages': c_recal_final,
        }

        results = {
            'pre_calibration': pre_results,
            'post_calibration': post_results,
            'fold_results': fold_results,
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
