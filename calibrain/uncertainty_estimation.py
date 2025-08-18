
import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm
from itertools import combinations, zip_longest
from matplotlib.patches import Ellipse

import mne
import logging


class UncertaintyEstimator:
    def __init__(self, confidence_levels : np.ndarray = None, logger : logging.Logger = None):
        """
        Initialize the uncertainty estimator.
        
        Parameters:
        -----------
        confidence_levels : list, optional
            List of confidence levels to compute confidence ellipses. Default is np.arange(0.0, 1.1, 0.1).
        logger : logging.Logger, optional
            Logger instance for logging messages.
        """
        self.confidence_levels = confidence_levels
        self.logger = logger

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

    def _compute_top_covariance_pairs(self, cov, top_k=None):
        """
        Compute and optionally sort the magnitudes of covariances for all pairs of dimensions.

        Parameters:
            cov (array-like): Covariance matrix of shape (n, n).
            top_k (int, optional): Number of top pairs to return. If None, return all pairs.

        Returns:
            list: A sorted list of tuples. Each tuple contains:
                - A pair of indices (i, j).
                - The absolute magnitude of their covariance.
        """
        # Ensure covariance matrix is a NumPy array
        cov = np.asarray(cov)

        # Get all unique pairs of indices
        n = cov.shape[0]
        pairs = list(combinations(range(n), 2))

        # Compute magnitudes of covariances for each pair
        pair_cov_magnitudes = [(pair, np.abs(cov[pair[0], pair[1]])) for pair in pairs]

        # Sort by covariance magnitude in descending order
        sorted_pairs = sorted(pair_cov_magnitudes, key=lambda x: x[1], reverse=True)

        # Return top-k pairs if specified
        if top_k is not None:
            return sorted_pairs[:top_k]
        return sorted_pairs

    def _make_psd(self, cov: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
        """
        Ensure that the covariance matrix is positive semi-definite by adding epsilon to the diagonal.
        """
        self.logger.info("Computing eigenvalues...")
        
        max_iterations = 100
        iterations = 0
        # compute eigenvalues
        eigenvalues = np.linalg.eigvals(cov)
        while not np.all(eigenvalues >= 0):
            self.logger.info(f"Iteration {iterations + 1}: Covariance matrix is not PSD.")
            cov += np.eye(cov.shape[0]) * epsilon
            epsilon *= 10
            iterations += 1
            if iterations > max_iterations:
                self.logger.warning("Covariance matrix could not be made positive semi-definite.")
                break
        return cov
    
    def _compute_confidence_ellipse(self, mean, cov, confidence_level=0.95):
        """
        Compute the parameters of a confidence ellipse for a given mean and covariance matrix.
        """
        # Validate covariance matrix
        condition_number = np.linalg.cond(cov)
        if condition_number > 1e10:
            print("Covariance matrix is ill-conditioned")
        
        # Regularize covariance matrix if not positive definite by adding gradually increasing epsilon to the diagonal.
        if not np.all(np.linalg.eigvals(cov) > 0):
            cov_psd = self._make_psd(cov.copy(), epsilon=1e-6)
        
        chi2_val = chi2.ppf(confidence_level, df=2)

        eigenvals, eigenvecs = np.linalg.eigh(cov_psd)
        
        if np.all(eigenvals > 0):
            print("Covariance matrix is now positive definite.")
        else:
            print("Covariance matrix is still not positive definite.")

        order = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[order]
        eigenvecs = eigenvecs[:, order]

        width, height = 2 * np.sqrt(eigenvals * chi2_val)
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        
        return width, height, angle

    def _compute_confidence_intervals(self, mean : np.ndarray, std_dev : np.ndarray, confidence_level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """Compute confidence intervals based on the diagonal of the covariance matrix.
        Assumes inputs correspond only to the active components.

        Parameters
        ----------
        mean : np.ndarray
            Mean array for active components, shape (n_sources, n_times).
        std_dev : np.ndarray
            Standard deviation array for active components, shape (n_sources, n_times).
        confidence_level : float, optional
            Confidence level for the intervals (e.g., 0.95 for 95%), by default 0.95

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
        ci_lower : np.ndarray
            Lower bounds of confidence intervals, shape (n_sources, n_times).
        ci_upper : np.ndarray
            Upper bounds of confidence intervals, shape (n_sources, n_times).
        """
        # Calculate the Z-score corresponding to the confidence level for a normal distribution
        # Example: 0.95 -> z = 1.96
        alpha = 1.0 - confidence_level
        # z = np.abs(np.percentile(np.random.normal(0, 1, 1000000), [alpha / 2 * 100, (1 - alpha / 2) * 100]))[1]
        z = norm.ppf(1 - alpha / 2)  # Two-tailed critical value
        self.logger.debug(f"Z-score for confidence level {confidence_level}: {z:.4f}")

        # Calculate confidence intervals: mean +/- z * std_dev
        ci_lower = mean - z * std_dev
        ci_upper = mean + z * std_dev

        self.logger.debug(f"Computed CI shapes: lower={ci_lower.shape}, upper={ci_upper.shape}")

        # The distinction between 'fixed' and 'free' is not needed here,
        # as the inputs `mean` and `cov` are already specific to the active components.
        # The interpretation of these components (fixed source vs. free orientation component)
        # happens in other functions like _count_values_within_ci.

        return ci_lower, ci_upper

    def _count_values_within_ci(self, x : np.ndarray, ci_lower : np.ndarray, ci_upper : np.ndarray, orientation_type : str) -> np.ndarray:
        """Count the number of ground truth values that lie within the confidence intervals for each time point.
        Assumes input arrays correspond only to the active components.

        Parameters
        ----------
        x : np.ndarray
            Ground truth source activity for active components, shape (n_sources, n_times).
        ci_lower : np.ndarray
            Lower bounds of confidence intervals for active components, shape (n_sources, n_times).
        ci_upper : np.ndarray
            Upper bounds of confidence intervals for active components, shape (n_sources, n_times).
        orientation_type : str
            Orientation type of the sources, either 'fixed' or 'free'.

        Returns
        -------
        np.ndarray
            Count of values within confidence intervals.
            - For "fixed" orientation: 1D array (n_times,) with counts per time point.
            - For "free" orientation: 2D array (3, n_times) with counts per orientation (X, Y, Z) per time point.
        
        Notes
        -----
        - active_indices is used only for "free" orientation to determine the orientation of each active component.            
        """
        n_times = x.shape[1]

        if orientation_type == "fixed":
            # Sum over all active sources for each time point
            count_within_ci = np.sum((x >= ci_lower) & (x <= ci_upper), axis=0)
            self.logger.debug(f"Fixed orientation counts shape: {count_within_ci.shape}")

        elif orientation_type == "free":
            # Initialize counts for each orientation (X, Y, Z) and each time point
            count_within_ci = np.zeros((3, n_times))

            # Determine the orientation for each row in the input arrays based on the original active_indices
            # orientations = active_indices % 3 # Shape: (n_active_components,)
            # TODO: hardcoded synthetic orientations for free orientation as active_indices are not used in this function
            orientations = np.array([0, 1, 2] * (x.shape[0] // 3))
            
            for i in range(3): # Iterate through X, Y, Z
                # Create a mask for rows corresponding to the current orientation
                orient_mask = (orientations == i)

                # Select the rows for the current orientation from the input arrays
                x_orient = x[orient_mask, :]
                ci_lower_orient = ci_lower[orient_mask, :]
                ci_upper_orient = ci_upper[orient_mask, :]

                # Count values within confidence intervals for the current orientation, summing over sources
                if x_orient.size > 0: # Ensure there are sources for this orientation
                     count_within_ci[i, :] = np.sum((x_orient >= ci_lower_orient) & (x_orient <= ci_upper_orient), axis=0)
                # else: counts remain zero, which is correct

            self.logger.debug(f"Free orientation counts shape: {count_within_ci.shape}")

    
        return count_within_ci

    def get_confidence_intervals_data(
        self,
        x : np.ndarray,
        x_hat: np.ndarray,
        posterior_cov: np.ndarray,
        orientation_type: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes confidence intervals and counts of true values within those intervals.

        Parameters:
        ----------
        x : np.ndarray
            Ground truth source activity.
            Shape (n_sources, n_times).
        x_hat : np.ndarray
            Estimated source activity (posterior mean).
            Shape (n_sources, n_times).
        posterior_cov : np.ndarray
            Posterior covariance matrix of shape (n_sources, n_sources).
        orientation_type : str
            Orientation type, either 'free' or 'fixed'.

        Returns:
        --------
        tuple
            - ci_lower_stacked (np.ndarray): Lower bounds of confidence intervals for each confidence level.
                Shape (n_confidence_levels, n_active_components, n_times).
            - ci_upper_stacked (np.ndarray): Upper bounds of confidence intervals for each confidence level.
                Shape (n_confidence_levels, n_active_components, n_times).  
            - counts_array (np.ndarray): Counts of true values within confidence intervals for each confidence level.
                Shape (n_confidence_levels, 3, n_times) for free orientation,
                or (n_confidence_levels, 1, n_times) for fixed orientation.
        """    
        # Ensure covariance matrix is positive semi-definite for variance calculation
        # Note: _make_psd might modify cov in place if not careful, consider passing a copy if needed elsewhere. However, we only need the diagonal here, so modifying cov might be acceptable if not used later.

        # Extract diagonal variances
        var = np.diag(posterior_cov).copy()
        
        # Check for negative variances
        negative_mask = var <= 0
        n_negative = np.sum(negative_mask)

        need_full_cov = True
        
        if n_negative > 0:
            self.logger.warning(f"Found {n_negative} non-positive variances. Regularizing...")
            
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
        assert np.all(var > 0), "All variances must be positive for std deviation calculation"
        
        # Calculate standard deviation
        std_dev = np.sqrt(var).reshape(-1, 1)
        
        # Add debugging info
        self.logger.info(f"Variance range: [{np.min(var):.2e}, {np.max(var):.2e}]")
        self.logger.info(f"Std dev range: [{np.min(std_dev):.2e}, {np.max(std_dev):.2e}]")

        all_ci_lower_list = []
        all_ci_upper_list = []
        collected_counts_within_ci_list = []

        self.logger.info("Computing confidence intervals and hit counts for each confidence level.")
        for cl_idx, confidence_level_val in enumerate(self.confidence_levels):
            self.logger.info(f"Processing confidence level {cl_idx + 1}/{len(self.confidence_levels)}: {confidence_level_val:.2f}")

            ci_lower, ci_upper = self._compute_confidence_intervals(
                mean=x_hat, 
                std_dev=std_dev, 
                confidence_level=confidence_level_val
            )

            count_within_ci = self._count_values_within_ci(
                x=x,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                orientation_type=orientation_type
            )
            
            all_ci_lower_list.append(ci_lower)
            all_ci_upper_list.append(ci_upper)
            collected_counts_within_ci_list.append(count_within_ci)

        counts_array = np.array(collected_counts_within_ci_list)
        
        # Ensure counts_array is 3D for consistent handling later,
        # especially if it's (L, T) for fixed, make it (L, 1, T)
        if orientation_type == 'fixed' and counts_array.ndim == 2: 
            counts_array = counts_array[:, np.newaxis, :] 
        
        ci_lower_stacked = np.stack(all_ci_lower_list, axis=0)
        ci_upper_stacked = np.stack(all_ci_upper_list, axis=0)

        self.logger.debug(f"Shapes returned: counts_array={counts_array.shape}, ci_lower_stacked={ci_lower_stacked.shape}, ci_upper_stacked={ci_upper_stacked.shape}")
        
        time_idx = 0
        total_matched_sources = x.shape[0]
        empirical_coverage = (counts_array / total_matched_sources)[:, 0, time_idx]

        return ci_lower_stacked, ci_upper_stacked, counts_array, empirical_coverage

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
