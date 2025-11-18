
import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2, norm
from itertools import combinations, zip_longest
import mne
import logging

from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold


class UncertaintyEstimator:
    def __init__(self, nominal_coverages: np.ndarray = None, 
                 logger: logging.Logger = None):
        """
        Initialize the uncertainty estimator with nominal coverages (theoretical or claimed confidence levels).
        
        Parameters:
        -----------
        nominal_coverages : np.ndarray
            Nominal confidence levels (c) - what we expect theoretically
        """
        if nominal_coverages is None:
            # Include 0 and 1 for proper calibration curve endpoints
            nominal_coverages = np.array([0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 
                                        0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])
            
        # Validate that all nominal coverages are between 0 and 1
        if np.any(nominal_coverages < 0) or np.any(nominal_coverages > 1):
            raise ValueError("Nominal coverages must be between 0 and 1")
            
        self.nominal_coverages = nominal_coverages
        self.logger = logger or logging.getLogger(__name__)
        
        # Precompute z-scores for common confidence levels
        self.z_scores = {}
        for c in nominal_coverages:
            if c == 0.0:
                self.z_scores[c] = 0.0  # Zero width interval
            elif c == 1.0:
                self.z_scores[c] = np.inf  # Infinite interval
            else:
                self.z_scores[c] = norm.ppf((1 + c) / 2)

    def get_posterior_variance(
        self,
        posterior_cov: np.ndarray,
        orientation_type: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract and regularize the posterior variance from the covariance matrix.
        Parameters:
        ----------
        posterior_cov : np.ndarray
            Posterior covariance matrix of shape (n_sources, n_sources) scaled by n_times.
        orientation_type : str
            Orientation type, either 'free' or 'fixed'. 
        Returns:
        -------
        np.ndarray
            Posterior variance vector of shape (n_sources, n_times).
        """    
        # Ensure covariance matrix is positive semi-definite for variance calculation
        # Note: _make_psd might modify cov in place if not careful, consider passing a copy if needed elsewhere. However, we only need the diagonal here, so modifying cov might be acceptable if not used later.

        # Extract diagonal variances
        var = np.diag(posterior_cov).copy()
        
        # Check for negative variances
        negative_mask = var < 0
        n_negative = np.sum(negative_mask)

        need_full_cov = True
        
        if n_negative > 0:
            self.logger.warning(f"Found {n_negative} non-positive variances (<= 0). Regularizing...")

            # Option 1: Simple diagonal regularization (faster)
            min_var = 1e-12  # Small positive value based on your data scale
            var[negative_mask] = min_var
            self.logger.info(f"Clipped {n_negative} variances to {min_var}")
            
            # Option 2: Full PSD regularization (more principled but slower)
            # Only do this if we need the full covariance matrix elsewhere
            if need_full_cov:
                cov_psd = self._make_psd(posterior_cov.copy())
                var = np.diag(cov_psd)
                
                # Recheck after PSD adjustment
                remaining_negative = np.sum(var <= 0)
                if remaining_negative > 0:
                    self.logger.warning(f"Still {remaining_negative} non-positive variances after PSD adjustment")
                    var[var <= 0] = min_var
        
        # Validate final variances
        assert np.all(var >= 0), "All variances must be positive for std deviation calculation"
        
        return var

    def construct_full_covariance(
        self,
        x : np.ndarray = None,
        x_hat_active_indices : np.ndarray = None,
        posterior_cov : np.ndarray = None,
        orientation_type : str = 'fixed',
    ) -> np.ndarray:
        """
        Create a full covariance matrix corresponding to all source components,
        embedding the posterior covariance of the active set.

        Parameters:
        ----------
        x : np.ndarray
            Ground truth source activity values for active components.
            Shape: (n_sources, n_times) for fixed orientation or (n_sources, 3, n_times) for free orientation.
        x_hat_active_indices : np.ndarray
            Indices of the active reconstructed sources in the original x_hat.
        posterior_cov : np.ndarray
            Full Posterior covariance matrix.
            Shape: (n_active_sources, n_active_sources) for fixed orientation or
                   (n_active_sources*3, n_active_sources*3) for free orientation.
        orientation_type : str
            Orientation type of the sources, either 'fixed' or 'free'.
        
        Returns:
        -------
        full_posterior_cov : np.ndarray
            Full covariance matrix of shape (n_total_components, n_total_components),
            where n_total_components is the total number of source components.
        """
        # For dense inverse methods like eLORETA, the posterior covariance matrix is already in the full form (number of reconstructed sources matches the simulated sources).
        if posterior_cov.shape[0] == x.shape[0]:
            return posterior_cov
        # Determine the total number of source components based on orientation type and x shape
        n_total_components = 0
        if orientation_type == 'fixed':
            if x.ndim != 2:
                raise ValueError(f"For fixed orientation, expected x to be 2D (n_sources, n_times), but got shape {x.shape}")
            n_total_components = x.shape[0] # n_sources
        elif orientation_type == 'free':
            # Expects original shape (n_sources, 3, n_times) or reshaped (n_sources*3, n_times)
            if x.ndim == 3: # Original shape
                 if x.shape[1] != 3:
                     raise ValueError(f"For free orientation with 3D x, expected shape (n_sources, 3, n_times), but got {x.shape}")
                 n_total_components = x.shape[0] * x.shape[1] # n_sources * 3

    
        # Initialize the full covariance matrix
        # full_posterior_cov = np.zeros((n_total_components, n_total_components), dtype=posterior_cov.dtype)
        full_posterior_cov = np.eye(n_total_components, dtype=posterior_cov.dtype) * 1e-12 # small positive diagonal

        self.logger.debug(f"Initialized full_posterior_cov with shape {full_posterior_cov.shape}")
    
        try:
            for i, idx_i in enumerate(x_hat_active_indices):
                for j, idx_j in enumerate(x_hat_active_indices):
                    full_posterior_cov[idx_i, idx_j] = posterior_cov[i, j]
            self.logger.debug("Successfully embedded posterior_cov into full_posterior_cov.")
        except IndexError as e:
            self.logger.error(f"IndexError during covariance embedding: {e}. "
                              f"i={i}, j={j}, idx_i={idx_i}, idx_j={idx_j}, "
                              f"posterior_cov shape={posterior_cov.shape}, x_hat_active_indices size={x_hat_active_indices.size}")
            raise IndexError(f"Error accessing elements during covariance embedding. Check x_hat_active_indices indices ({idx_i}, {idx_j}) against posterior_cov shape {posterior_cov.shape} using indices ({i}, {j}).") from e

        try:
            # This assertion holds true if posterior_cov is dense and square for the active set
            assert posterior_cov.size == x_hat_active_indices.size ** 2, \
                f"Size of posterior_cov ({posterior_cov.size}) should be square of x_hat_active_indices size ({x_hat_active_indices.size}^2 = {x_hat_active_indices.size**2})."
        except AssertionError as e:
            self.logger.error(f"Assertion failed: {e}")
            raise AssertionError(f"Validation failed: {e}") from e
    
        self.logger.debug(f"Constructed full covariance matrix of shape {full_posterior_cov.shape}")
        return full_posterior_cov

    def _make_psd(self, cov: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
        """
        Proper PSD regularization using eigenvalue clipping.
        """
        # Ensure symmetric
        cov = (cov + cov.T) / 2.0
        
        # Eigen decomposition
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        
        # Clip negative eigenvalues
        eigenvals_clipped = np.maximum(eigenvals, epsilon)
        
        # Reconstruct
        cov_psd = eigenvecs @ np.diag(eigenvals_clipped) @ eigenvecs.T
        
        # Ensure symmetry
        cov_psd = (cov_psd + cov_psd.T) / 2.0
        
        n_negative = np.sum(eigenvals < 0)
        if n_negative > 0:
            self.logger.warning(f"Clipped {n_negative} negative eigenvalues")
            
        return cov_psd


    # ========== CREDIBLE INTERVALS AND COVERAGE =========
    def _compute_credible_intervals(
        self, 
        mean: np.ndarray, 
        variance: np.ndarray, 
        nominal_coverage: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute credible intervals with proper handling of extremes.
        
        For c=0: Zero-width intervals (lower = upper = mean)
        For c=1: Infinite intervals ([-∞, +∞])
        For others: Normal intervals
        """
        if nominal_coverage == 0.0:
            # Zero-width intervals
            return mean.copy(), mean.copy()
        elif nominal_coverage == 1.0:
            # Infinite intervals
            return np.full_like(mean, -np.inf), np.full_like(mean, np.inf)
        else:
            # Normal intervals
            z = self.z_scores[nominal_coverage]
            std_dev = np.sqrt(variance)
            ci_lower = mean - z * std_dev
            ci_upper = mean + z * std_dev
            return ci_lower, ci_upper

    def compute_empirical_coverage(
        self,
        x_true: np.ndarray,
        x_hat: np.ndarray, 
        posterior_var: np.ndarray,
        nominal_coverage: float
    ) -> float:
        """
        Compute empirical coverage for given nominal coverage level.
        
        ĉ = (1/N) * Σ_{i=1}^N I[x_i^{true} ∈ CI_i^{(c)}]
        
        Parameters:
        -----------
        x_true : np.ndarray
            Ground truth source activity, shape (n_sources, n_times) or (n_sources,)
        x_hat : np.ndarray
            Posterior mean estimates, shape (n_sources, n_times) or (n_sources,)
        posterior_var : np.ndarray
            Posterior variance, shape (n_sources, n_times) or (n_sources,)
        nominal_coverage : float
            Nominal coverage level c (what we expect)
            
            
        Returns:
        --------
        ci_lower : np.ndarray
            Lower bounds of credible intervals
        ci_upper : np.ndarray
            Upper bounds of credible intervals
        ci_count : int
            Number of true values within credible intervals
        empirical_coverage : float
            Empirical coverage ĉ (what we actually get)
        """
        # Compute credible intervals
        ci_lower, ci_upper = self._compute_credible_intervals(
            x_hat, posterior_var, nominal_coverage
        )
        
        # Count how many true values fall within intervals
        within_ci = (x_true >= ci_lower) & (x_true <= ci_upper)
        empirical_coverage = np.mean(within_ci)
        
        return ci_lower, ci_upper, np.sum(within_ci), empirical_coverage

    def get_calibration_curve(
        self,
        x_true: np.ndarray,
        x_hat: np.ndarray,
        posterior_var: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute calibration curve showing empirical vs nominal coverage.
        
        Parameters:
        -----------
        x_true : np.ndarray
            Ground truth source activity, shape (n_sources, n_times) or (n_sources,)
        x_hat : np.ndarray
            Posterior mean estimates, shape (n_sources, n_times) or (n_sources,)
        posterior_var : np.ndarray
            Posterior variance, shape (n_sources, n_times) or (n_sources,)
            
        Returns:
        --------
        dict
            Dictionary containing nominal coverages, empirical coverages, and credible interval bounds. Each key is:
            - "nominal_coverages": np.ndarray of nominal coverage levels
            - "empirical_coverages": np.ndarray of empirical coverage levels
            - "ci_lowers": np.ndarray of lower bounds of credible intervals
            - "ci_uppers": np.ndarray of upper bounds of credible intervals
            - "ci_counts": np.ndarray of counts of true values within credible intervals
        """
        # Flatten arrays to treat all sources and time points as independent observations
        x_true_flat = x_true.flatten()
        x_hat_flat = x_hat.flatten()
        posterior_var_flat = posterior_var.flatten()
        
        n_observations = x_true_flat.shape[0]
        self.logger.info(f"Computing calibration curve for {n_observations} observations")
        
        # Compute empirical coverage for each nominal coverage level
        empirical_coverages = []
        ci_lowers = []
        ci_uppers = []
        ci_counts = []
        
        for nominal_coverage in self.nominal_coverages:
            ci_lower, ci_upper, ci_count, coverage = self.compute_empirical_coverage(
                x_true_flat, x_hat_flat, posterior_var_flat, nominal_coverage
            )
            # Store
            ci_uppers.append(ci_upper)
            ci_counts.append(ci_count)
            ci_lowers.append(ci_lower)
            empirical_coverages.append(coverage)
            
            self.logger.debug(f"c = {nominal_coverage:.3f}: ĉ = {coverage:.3f}")
        

        return {
            "nominal_coverages": self.nominal_coverages,
            "empirical_coverages": np.array(empirical_coverages),
            "ci_lowers": np.array(ci_lowers),
            "ci_uppers": np.array(ci_uppers),
            "ci_counts": np.array(ci_counts)
        }

    #  ========= CALIBRATION METHODS =========
    def calibration_CV(self, x, x_hat, posterior_var, n_folds=5, random_state=42):
        """
        Calibrate credible intervals using isotonic regression (IR) with source-wise cross-validation avoiding overfitting
        
        Parameters:
        -----------
        Computes credible intervals and counts of true values within those intervals.

        Parameters:
        ----------
        x : np.ndarray
            Ground truth source activity.
            Shape (n_sources, n_times) averaged across time. For averaged time, n_times=1.
        x_hat : np.ndarray
            Estimated source activity (posterior mean) averaged across time.
            Shape (n_sources, n_times). For averaged time, n_times=1.
        posterior_var : np.ndarray
            Posterior variance vector of shape (n_sources,).
        orientation_type : str
            Orientation type, either 'free' or 'fixed'.
        n_folds : int
            Number of cross-validation folds
        random_state : int
            Random seed for reproducible splits
        
        Returns:
        --------
        cal_coverages : ndarray
            Calibrated coverage probabilities averaged across folds
        fold_results : list
            Detailed results for each fold
        
        Methodology:
        ------------
        1. Split sources, use cross-validation to train/test IR model to avoid overfitting
        2. For each fold:
        - Train: Compute empirical coverage on training sources
        - Train: Fit isotonic regression (empirical → nominal coverage)
        - Test: Apply calibration to test sources
        3. Average results across folds
        
        Theory
        ------------
        Learns the mapping from observed empirical coverage to desired nominal coverage
        using monotonic regression to preserve probability ordering
        """
        n_sources = x_hat.shape[0]
        
        # Initialize cross-validation with shuffling and fixed seed
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        source_idx = np.arange(n_sources)
        
        calibrated_coverages = np.zeros(len(self.nominal_coverages))
        fold_results = []
        
        print(f"Running {n_folds}-fold source-wise calibration:")
        print(f"Splitting {n_sources} sources into training/test sets")
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(source_idx)):
            # Split sources (not time points!)
            x_train = x[train_idx]
            x_hat_train = x_hat[train_idx]
            posterior_var_train = posterior_var[train_idx]
            
            x_test = x[test_idx]
            x_hat_test = x_hat[test_idx]
            posterior_var_test = posterior_var[test_idx]
            
            # Compute empirical coverages for train set
            train_coverage_dict = self.get_calibration_curve(
                x_true=x_train,
                x_hat=x_hat_train,
                posterior_var=posterior_var_train,
                )
            
            # Compute empirical coverages for test set
            test_coverage_dict = self.get_calibration_curve(
                    x_true=x_test,
                    x_hat=x_hat_test,
                    posterior_var=posterior_var_test,
                )
            
            # TRAIN: Fit isotonic regression (empirical -> nominal coverage)
            # Preserves monotonicity: higher confidence -> higher coverage
            ir_model = IsotonicRegression(out_of_bounds='clip')
            ir_model.fit(train_coverage_dict['empirical_coverages'], self.nominal_coverages)
            
            # TEST: Apply calibration to test sources
            test_calibrated = ir_model.transform(test_coverage_dict['empirical_coverages'])
            
            # Store fold results
            fold_results.append({
                'train_empirical': train_coverage_dict['empirical_coverages'],
                'test_empirical': test_coverage_dict['empirical_coverages'],
                'calibrated': test_calibrated,
                'n_train': len(train_idx),
                'n_test': len(test_idx),
                'fold': fold
            })
            
            # Accumulate calibrated coverages
            calibrated_coverages += test_calibrated
            
            print(f"  Fold {fold+1}: {len(train_idx)} train, {len(test_idx)} test sources")
        
        # Average across all folds
        calibrated_coverages /= n_folds
        
        print("Calibration completed successfully!")
        return calibrated_coverages, fold_results
    #  

    # ========= DEBUGGING FUNCTIONS =========
    def debug_covariance(self, posterior_cov, full_posterior_cov, step_name):
        """Debug covariance matrix properties"""
        print(f"\n--- {step_name} ---")
        
        # Original covariance
        orig_eigenvals = np.linalg.eigvals(posterior_cov)
        print(f"Original posterior_cov:")
        print(f"  Shape: {posterior_cov.shape}")
        print(f"  Eigenvalues range: [{np.min(orig_eigenvals):.2e}, {np.max(orig_eigenvals):.2e}]")
        print(f"  Rank: {np.linalg.matrix_rank(posterior_cov)}")
        print(f"  Condition number: {np.linalg.cond(posterior_cov):.2e}")
        
        # Full covariance
        full_eigenvals = np.linalg.eigvals(full_posterior_cov)
        full_diag = np.diag(full_posterior_cov)
        print(f"Full posterior_cov:")
        print(f"  Shape: {full_posterior_cov.shape}")
        print(f"  Eigenvalues range: [{np.min(full_eigenvals):.2e}, {np.max(full_eigenvals):.2e}]")
        print(f"  Diagonal range: [{np.min(full_diag):.2e}, {np.max(full_diag):.2e}]")
        print(f"  Negative diagonal elements: {np.sum(full_diag < 0)}")
        print(f"  Zero diagonal elements: {np.sum(full_diag == 0)}")
        # print(f"  Rank: {np.linalg.matrix_rank(full_posterior_cov)}")

    def debug_posterior_properties(self, x_hat: np.ndarray, posterior_cov: np.ndarray):
        """
        Debug function to check posterior properties.
        """
        print("\n=== POSTERIOR PROPERTIES ===")
        print(f"x_hat shape: {x_hat.shape}")
        print(f"posterior_cov shape: {posterior_cov.shape}")
        
        # Check if covariance is positive definite
        eigenvals = np.linalg.eigvals(posterior_cov)
        min_eigenval = np.min(eigenvals)
        max_eigenval = np.max(eigenvals)
        condition_number = np.linalg.cond(posterior_cov)
        
        print(f"Eigenvalue range: [{min_eigenval:.2e}, {max_eigenval:.2e}]")
        print(f"Condition number: {condition_number:.2e}")
        print(f"Negative eigenvalues: {np.sum(eigenvals < 0)}")
        
        # Check variances
        variances = np.diag(posterior_cov)
        print(f"Variance range: [{np.min(variances):.2e}, {np.max(variances):.2e}]")
        print(f"Zero variances: {np.sum(variances == 0)}")

    # def _compute_top_covariance_pairs(self, cov, top_k=None):
    #     """
    #     Compute and optionally sort the magnitudes of covariances for all pairs of dimensions.

    #     Parameters:
    #         cov (array-like): Covariance matrix of shape (n, n).
    #         top_k (int, optional): Number of top pairs to return. If None, return all pairs.

    #     Returns:
    #         list: A sorted list of tuples. Each tuple contains:
    #             - A pair of indices (i, j).
    #             - The absolute magnitude of their covariance.
    #     """
    #     # Ensure covariance matrix is a NumPy array
    #     cov = np.asarray(cov)

    #     # Get all unique pairs of indices
    #     n = cov.shape[0]
    #     pairs = list(combinations(range(n), 2))

    #     # Compute magnitudes of covariances for each pair
    #     pair_cov_magnitudes = [(pair, np.abs(cov[pair[0], pair[1]])) for pair in pairs]

    #     # Sort by covariance magnitude in descending order
    #     sorted_pairs = sorted(pair_cov_magnitudes, key=lambda x: x[1], reverse=True)

    #     # Return top-k pairs if specified
    #     if top_k is not None:
    #         return sorted_pairs[:top_k]
    #     return sorted_pairs
    
    # def _compute_credible_ellipse(self, mean, cov, confidence_level=0.95):
    #     """
    #     Compute the parameters of a confidence ellipse for a given mean and covariance matrix.
    #     """
    #     # Validate covariance matrix
    #     condition_number = np.linalg.cond(cov)
    #     if condition_number > 1e10:
    #         print("Covariance matrix is ill-conditioned")
        
    #     # Regularize covariance matrix if not positive definite by adding gradually increasing epsilon to the diagonal.
    #     if not np.all(np.linalg.eigvals(cov) > 0):
    #         cov_psd = self._make_psd(cov.copy(), epsilon=1e-6)
        
    #     chi2_val = chi2.ppf(confidence_level, df=2)

    #     eigenvals, eigenvecs = np.linalg.eigh(cov_psd)
        
    #     if np.all(eigenvals > 0):
    #         print("Covariance matrix is now positive definite.")
    #     else:
    #         print("Covariance matrix is still not positive definite.")

    #     order = np.argsort(eigenvals)[::-1]
    #     eigenvals = eigenvals[order]
    #     eigenvecs = eigenvecs[:, order]

    #     width, height = 2 * np.sqrt(eigenvals * chi2_val)
    #     angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        
    #     return width, height, angle
