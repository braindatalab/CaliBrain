
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from itertools import combinations
from matplotlib.patches import Ellipse
import mne
import logging


class UncertaintyEstimator:
    def __init__(self, orientation_type, x, x_hat, active_set, posterior_cov, experiment_dir=None, logger=None):
        """
        Initialize the uncertainty estimator.
        
        Parameters:
        - orientation_type (str): Orientation type ('fixed' or 'free').
        - x (np.ndarray): Ground truth source activity.
        - x_hat (np.ndarray): Estimated source activity.
        - active_set (np.ndarray): Indices of active sources.
        - posterior_cov (np.ndarray, optional): Posterior covariance matrix.
        - experiment_dir (str, optional): Directory for experiment results.
        - logger (logging.Logger, optional): Logger instance for logging messages.
        """
        self.orientation_type = orientation_type
        self.x = x
        self.x_hat = x_hat
        self.active_set = active_set
        self.posterior_cov = posterior_cov
        self.experiment_dir = experiment_dir
        self.logger = logger

    def reshape_source_data(self, n_times):
        """
        Reshape data based on the orientation type.

        Parameters:
        - n_times (int): Number of time points.

        Returns:
        - x (np.ndarray): Reshaped ground truth source activity.
        - x_hat (np.ndarray): Reshaped estimated source activity.
        """
        if self.orientation_type == "free":
            self.x = self.x.reshape(-1, n_times)
            self.x_hat = self.x_hat.reshape(-1, n_times)
        return self.x, self.x_hat


    def construct_full_covariance(self):
        """
        Create a full covariance matrix corresponding to all source components,
        embedding the posterior covariance of the active set.
    
        Handles both 'fixed' and 'free' orientation types by inspecting the
        shape of self.x and the self.orientation_type attribute.
    
        Returns:
        - full_posterior_cov (np.ndarray): Full posterior covariance matrix.
                                            Shape: (n_total_components, n_total_components).
    
        Raises:
        - ValueError: If input shapes, orientation_type, or active_set indices are inconsistent.
        - AttributeError: If required attributes (x, active_set, posterior_cov, orientation_type) are missing.
        """
    
        # Determine the total number of source components based on orientation type and x shape
        n_total_components = 0
        if self.orientation_type == 'fixed':
            if self.x.ndim != 2:
                raise ValueError(f"For fixed orientation, expected self.x to be 2D (n_sources, n_times), but got shape {self.x.shape}")
            n_total_components = self.x.shape[0] # n_sources
        elif self.orientation_type == 'free':
            # Expects original shape (n_sources, 3, n_times) or reshaped (n_sources*3, n_times)
            if self.x.ndim == 3: # Original shape
                 if self.x.shape[1] != 3:
                     raise ValueError(f"For free orientation with 3D self.x, expected shape (n_sources, 3, n_times), but got {self.x.shape}")
                 n_total_components = self.x.shape[0] * self.x.shape[1] # n_sources * 3

    
        # Initialize the full covariance matrix
        full_posterior_cov = np.zeros((n_total_components, n_total_components), dtype=self.posterior_cov.dtype)
        self.logger.debug(f"Initialized full_posterior_cov with shape {full_posterior_cov.shape}")
    
        # Embed the active set covariance using nested loops (safe for unsorted active_set)
        #    Alternative: If performance critical and active_set is sorted:
        #    idx = np.ix_(self.active_set, self.active_set)
        #    full_posterior_cov[idx] = self.posterior_cov
        try:
            for i, idx_i in enumerate(self.active_set):
                for j, idx_j in enumerate(self.active_set):
                    full_posterior_cov[idx_i, idx_j] = self.posterior_cov[i, j]
            self.logger.debug("Successfully embedded posterior_cov into full_posterior_cov.")
        except IndexError as e:
             self.logger.error(f"IndexError during covariance embedding: {e}. "
                          f"i={i}, j={j}, idx_i={idx_i}, idx_j={idx_j}, "
                          f"posterior_cov shape={self.posterior_cov.shape}, active_set size={self.active_set.size}")
             raise IndexError(f"Error accessing elements during covariance embedding. Check active_set indices ({idx_i}, {idx_j}) against posterior_cov shape {self.posterior_cov.shape} using indices ({i}, {j}).") from e
      
        try:
            # This assertion holds true if posterior_cov is dense and square for the active set
            assert self.posterior_cov.size == self.active_set.size ** 2, \
                f"Size of posterior_cov ({self.posterior_cov.size}) should be square of active_set size ({self.active_set.size}^2 = {self.active_set.size**2})."
        except AssertionError as e:
            self.logger.error(f"Assertion failed: {e}")
            raise AssertionError(f"Validation failed: {e}") from e
    
        self.logger.debug(f"Constructed full covariance matrix of shape {full_posterior_cov.shape}")
        return full_posterior_cov
# ------------------------------

# ------------------------------
    def plot_sorted_posterior_variances(self, top_k=None):
        """
        Plot the sorted variances from the covariance matrix, highlighting the top-k variances.
        """
        variances = np.diag(self.posterior_cov)
        sorted_indices = np.argsort(variances)[::-1]
        sorted_variances = variances[sorted_indices]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(sorted_variances)), sorted_variances, color='skyblue', edgecolor='blue')

        if top_k is not None:
            for bar in bars[:top_k]:
                bar.set_color('orange')

        plt.xlabel("Source Index")
        plt.ylabel("Variance")
        plt.title(f"Sorted Posterior Variances (Top-{top_k if top_k else len(variances)} Highlighted)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        plt.savefig(os.path.join(self.experiment_dir, 'sorted_variances.png'))
        plt.close()
# ------------------------------
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
    
    def visualize_sorted_covariances(self, top_k=None):
        """
        Visualize the sorted magnitudes of covariances for all pairs of dimensions.
        """
        sorted_pairs = self._compute_top_covariance_pairs(self.posterior_cov, top_k=top_k)
        pairs = [f"({i},{j})" for (i, j), _ in sorted_pairs]
        magnitudes = [magnitude for _, magnitude in sorted_pairs]

        plt.figure(figsize=(10, 6))
        plt.bar(pairs, magnitudes, color='skyblue')
        plt.xlabel('Pairs of Dimensions')
        plt.ylabel('Covariance Magnitude')
        plt.title(f"Top-{top_k if top_k else len(magnitudes)} Sorted Covariance Magnitudes")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        plt.savefig(os.path.join(self.experiment_dir, 'sorted_covariances.png'))
        plt.close()
# ------------------------------
    def _make_psd(self, cov, epsilon=1e-6):
        """
        Ensure that the covariance matrix is positive semi-definite by adding epsilon to the diagonal.
        """
        # print("Regularizing covariance matrix...")
        max_iterations = 100
        iterations = 0
        while not np.all(np.linalg.eigvals(cov) >= 0):
            cov += np.eye(cov.shape[0]) * epsilon
            epsilon *= 10
            iterations += 1
            if iterations > max_iterations:
                self.logger.warning("Regularizing covariance matrix...")
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
            cov = self._make_psd(cov, epsilon=1e-6)
        
        chi2_val = chi2.ppf(confidence_level, df=2)

        eigenvals, eigenvecs = np.linalg.eigh(cov)
        
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

    def _plot_confidence_ellipse(self, mean, width, height, angle, ax=None, **kwargs):
        """
        Plot a confidence ellipse for given parameters.

        Parameters:
        - mean: array-like, shape (2,)
            The mean of the data in the two dimensions being plotted.
        - width: float
            The width of the ellipse (related to variance along the major axis).
        - height: float
            The height of the ellipse (related to variance along the minor axis).
        - angle: float
            The rotation angle of the ellipse in degrees.
        - ax: matplotlib.axes.Axes, optional
            The axis on which to plot the ellipse. If None, creates a new figure.
        - **kwargs: additional keyword arguments for matplotlib.patches.Ellipse.
        """
        if ax is None:
            fig, ax = plt.subplots()

        # Add ellipse patch
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
        ax.add_patch(ellipse)
        ax.scatter(*mean, color='blue', label='Mean')
        
        # Set axis labels
        ax.set_xlabel("Principal Component 1 (Variance in Dim 1)")
        ax.set_ylabel("Principal Component 2 (Variance in Dim 2)")

        # Set title
        ax.set_title("Confidence Ellipse (Width and Height Indicate Variance)")
        ax.grid()
        ax.legend()
    
    def plot_top_relevant_CE_pairs(self, top_k=5, confidence_level=0.95):
        """
        Identify the top-k relevant pairs of dimensions (based on covariance magnitude)
        and plot their confidence ellipses.
        """
        mean = self.x_hat[self.active_set]
        cov = self.posterior_cov

        n = len(mean)
        pairs = list(combinations(range(n), 2))
        pair_cov_magnitudes = [(pair, np.abs(cov[pair[0], pair[1]])) for pair in pairs]
        sorted_pairs = sorted(pair_cov_magnitudes, key=lambda x: x[1], reverse=True)
        top_pairs = [pair for pair, _ in sorted_pairs[:top_k]]

        n_cols = min(3, top_k)
        n_rows = (top_k + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
        axes = axes.flatten()

        for idx, (i, j) in enumerate(top_pairs):
            mean_ij = mean[[i, j]]
            cov_ij = cov[np.ix_([i, j], [i, j])]

            width, height, angle = self._compute_confidence_ellipse(mean_ij, cov_ij, confidence_level)
            self._plot_confidence_ellipse(mean_ij, width, height, angle, ax=axes[idx], edgecolor='blue', alpha=0.5)

            axes[idx].set_title(f"Dimensions {i} & {j}")

        for ax in axes[len(top_pairs):]:
            fig.delaxes(ax)

        fig.suptitle("Top Relevant Dimensional Pairs with Confidence Ellipses", fontsize=16)
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        plt.savefig(os.path.join(self.experiment_dir, 'top_relevant_CE_pairs.png'))
        plt.close()
        
# ------------------------------

    def plot_active_sources_single_time_step(self, time_step=0):
        """
        Plot the active sources for a single time step, comparing ground truth and estimated sources.
        Handles both 3D and 2D input shapes for x and x_hat.
        """
        if self.orientation_type == 'free':
            if self.x_hat.ndim == 3:
                n_sources, n_orient, n_times = self.x_hat.shape
                x_hat_is_3d = True
            elif self.x_hat.ndim == 2:
                n_components, n_times = self.x_hat.shape
                if n_components % 3 != 0:
                        self.logger.warning(f"Free orientation: self.x_hat is 2D, but first dim ({n_components}) not divisible by 3.")
                n_sources = n_components // 3
                x_hat_is_3d = False
            else:
                raise ValueError(f"Unexpected number of dimensions for self.x_hat: {self.x_hat.ndim}")

            # --- Handle self.x shape similarly ---
            if self.x.ndim == 3:
                x_is_3d = True
            elif self.x.ndim == 2:
                    x_is_3d = False

            fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True) # Share x-axis
            orientations = ['X', 'Y', 'Z']

            for i, ax in enumerate(axes): # i is the orientation index (0, 1, 2)
                # --- Ground Truth ---
                if x_is_3d:
                    # Find non-zero elements for this orientation at this time step
                    gt_source_indices_orient = np.where(self.x[:, i, time_step] != 0)[0]
                    gt_amplitudes = self.x[gt_source_indices_orient, i, time_step]
                else: # x is 2D (n_sources*3, n_times)
                    gt_indices_all = np.where(self.x[:, time_step] != 0)[0]
                    gt_indices_orient_flat = gt_indices_all[gt_indices_all % 3 == i] # Flat indices for this orientation
                    gt_source_indices_orient = gt_indices_orient_flat // 3 # Source indices
                    gt_amplitudes = self.x[gt_indices_orient_flat, time_step]

                # --- Estimated ---
                # Get the flat indices from active_set corresponding to this orientation
                active_indices_orient_flat = self.active_set[self.active_set % 3 == i]

                # Derive source indices from the flat indices
                est_source_indices_orient = active_indices_orient_flat // 3

                # Get amplitudes using appropriate indexing based on x_hat shape
                if x_hat_is_3d:
                    # Check bounds before indexing 3D array
                    valid_source_indices = est_source_indices_orient[est_source_indices_orient < n_sources]
                    if len(valid_source_indices) < len(est_source_indices_orient):
                        self.logger.warning(f"Orientation {i}: Some derived source indices from active_set were out of bounds for 3D x_hat. Filtering.")
                    est_amplitudes = self.x_hat[valid_source_indices, i, time_step]
                    # Use the valid source indices for plotting
                    plot_est_source_indices = valid_source_indices
                else: # x_hat is 2D (n_sources*3, n_times)
                    # Check bounds before indexing 2D array
                    max_idx_x_hat = self.x_hat.shape[0] - 1
                    valid_flat_indices = active_indices_orient_flat[active_indices_orient_flat <= max_idx_x_hat]
                    if len(valid_flat_indices) < len(active_indices_orient_flat):
                            self.logger.warning(f"Orientation {i}: Some flat indices from active_set were out of bounds for 2D x_hat. Filtering.")
                    est_amplitudes = self.x_hat[valid_flat_indices, time_step]
                    # Use source indices derived from valid flat indices for plotting
                    plot_est_source_indices = valid_flat_indices // 3


                # --- Plotting ---
                ax.scatter(gt_source_indices_orient, gt_amplitudes, color='blue', alpha=0.6, label='Ground Truth Active')
                ax.scatter(plot_est_source_indices, est_amplitudes, color='red', marker='x', alpha=0.6, label='Estimated Active')
                ax.set_xlabel('Source Index') # Label only needed on bottom plot due to sharex
                ax.set_ylabel('Amplitude')
                ax.set_title(f'Active Sources Comparison ({orientations[i]} Orientation, Time Step {time_step})')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.5)
                ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)

            # Add shared x-label
            fig.text(0.5, 0.04, 'Source Index', ha='center', va='center')
            plt.tight_layout(rect=[0, 0.05, 1, 0.96]) 
            fig.suptitle(f"Active Sources Comparison (Free Orientation, Time Step {time_step})", fontsize=16)

            save_path = os.path.join(self.experiment_dir, f'active_sources_single_time_step_{time_step}.png')
            plt.savefig(save_path)
            self.logger.debug(f"Saved active sources plot to {save_path}")
            plt.close(fig)

        else: # Fixed orientation (assuming x and x_hat are 2D: n_sources, n_times)
            if self.x.ndim != 2 or self.x_hat.ndim != 2:
                    raise ValueError(f"Fixed orientation plotting expects 2D x ({self.x.shape}) and x_hat ({self.x_hat.shape})")
            if self.x.shape[0] != self.x_hat.shape[0]:
                    raise ValueError(f"Shape mismatch between x ({self.x.shape}) and x_hat ({self.x_hat.shape})")

            n_sources = self.x.shape[0]
            max_index_x_hat = n_sources - 1

            gt_active_sources = np.where(self.x[:, time_step] != 0)[0]
            gt_amplitudes = self.x[gt_active_sources, time_step]

            # Filter active_set for valid indices
            valid_mask = self.active_set <= max_index_x_hat
            active_set_plot = self.active_set[valid_mask]
            if not np.all(valid_mask):
                invalid_indices = self.active_set[~valid_mask]
                self.logger.warning(f"Fixed Orientation: Found indices in active_set {invalid_indices.tolist()} "
                                f"that are out of bounds for x_hat (max index: {max_index_x_hat}). Filtering them out for plotting.")

            est_amplitudes = self.x_hat[active_set_plot, time_step]

            plt.figure(figsize=(12, 6))
            plt.scatter(gt_active_sources, gt_amplitudes, color='blue', alpha=0.6, label='Ground Truth Active')
            plt.scatter(active_set_plot, est_amplitudes, color='red', marker='x', alpha=0.6, label='Estimated Active')
            plt.xlabel('Source Index')
            plt.ylabel('Amplitude')
            plt.title(f'Active Sources Comparison (Fixed Orientation, Time Step {time_step})')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.5)
            plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
            plt.tight_layout(rect=[0, 0.05, 1, 0.96])
            save_path = os.path.join(self.experiment_dir, f'active_sources_single_time_step_{time_step}.png')
            plt.savefig(save_path)
            self.logger.debug(f"Saved active sources plot to {save_path}")
            plt.close()

    def plot_posterior_covariance_matrix(self):
        """
        Plot the posterior covariance matrix.
        """
        if self.orientation_type == 'free':
            # Check if posterior_cov shape is compatible with free orientation slicing
            n_active_components = self.posterior_cov.shape[0]
            if n_active_components % 3 != 0:
                self.logger.warning(f"Free orientation: posterior_cov shape {self.posterior_cov.shape}, first dimension is not divisible by 3.")
                # Fallback to plotting the whole matrix if slicing is not possible
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(self.posterior_cov, cmap='viridis', aspect='auto')
                fig.colorbar(im, ax=ax, label='Covariance Value')
                ax.set_title('Posterior Covariance Matrix (Free Orientation - Full)')
                ax.set_xlabel('Active Component Index')
                ax.set_ylabel('Active Component Index')
                plt.tight_layout(rect=[0, 0.05, 1, 0.96])

            else:
                fig, axes = plt.subplots(3, 1, figsize=(10, 18))
                orientations = ['X', 'Y', 'Z']
                # Determine shared color limits across the subplots
                vmin = np.min(self.posterior_cov)
                vmax = np.max(self.posterior_cov)

                images = [] # Store image objects for colorbar
                for i, ax in enumerate(axes):
                    # Select the block corresponding to the orientation
                    # This assumes active_set components are ordered [src0_x, src0_y, src0_z, src1_x, ...]
                    # which might not be true. A safer plot might show the full matrix.
                    # Let's plot the diagonal blocks for now, assuming structure.
                    try:
                        cov_matrix_block = self.posterior_cov[i::3, i::3]
                        im = ax.imshow(cov_matrix_block, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
                        images.append(im)
                        ax.set_title(f'Diagonal Block - Orientation {orientations[i]}')
                        ax.set_xlabel('Source Index (within orientation)')
                        ax.set_ylabel('Source Index (within orientation)')
                    except IndexError:
                        self.logger.warning(f"Could not extract block {i}::3 for orientation {orientations[i]}. Skipping subplot.")
                        ax.set_title(f'Orientation {orientations[i]} - Error')

                plt.tight_layout(rect=[0, 0.05, 1, 0.96])

                # Add colorbar spanning all axes, using the first image's mappable
                if images:
                    fig.colorbar(images[0], ax=axes.ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.04, label='Covariance Value')


        else: # Fixed orientation
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(self.posterior_cov, cmap='viridis', aspect='auto')
            plt.colorbar(im, label='Covariance Value')
            ax.set_title('Posterior Covariance Matrix (Fixed Orientation)')
            ax.set_xlabel('Active Source Index')
            ax.set_ylabel('Active Source Index')
            plt.tight_layout(rect=[0, 0.05, 1, 0.96])

        try:
            plt.savefig(os.path.join(self.experiment_dir, 'posterior_covariance_matrix.png'))
            self.logger.info(f"Posterior covariance matrix plot saved to {self.experiment_dir}/posterior_covariance_matrix.png")
        except Exception as e:
            self.logger.error(f"Failed to save posterior covariance matrix plot: {e}")
        finally:
            plt.close(fig) 


# ------------------------------
    def _compute_confidence_intervals(self, mean, cov, confidence_level=0.95):
        """
        Compute confidence intervals based on the diagonal of the covariance matrix.
        Assumes inputs correspond only to the active components.

        Parameters:
        - mean (np.ndarray): Mean array for active components, shape (n_active_components, n_times).
        - cov (np.ndarray): Covariance matrix for active components, shape (n_active_components, n_active_components).
        - confidence_level (float): Confidence level for the intervals (e.g., 0.95 for 95%).

        Returns:
        - ci_lower (np.ndarray): Lower bounds of confidence intervals, shape (n_active_components, n_times).
        - ci_upper (np.ndarray): Upper bounds of confidence intervals, shape (n_active_components, n_times).
        """
        # Calculate the Z-score corresponding to the confidence level for a normal distribution
        # Example: 0.95 -> z = 1.96
        alpha = 1.0 - confidence_level
        z = np.abs(np.percentile(np.random.normal(0, 1, 1000000), [alpha / 2 * 100, (1 - alpha / 2) * 100]))[1]
        self.logger.debug(f"Z-score for confidence level {confidence_level}: {z:.4f}")

        # Ensure covariance matrix is positive semi-definite for variance calculation
        # Note: _make_psd might modify cov in place if not careful, consider passing a copy if needed elsewhere.
        # However, we only need the diagonal here, so modifying cov might be acceptable if not used later.
        cov_psd = self._make_psd(cov.copy()) # Work on a copy to avoid modifying original cov

        # Extract diagonal variances
        variances = np.diag(cov_psd)

        # Handle potential negative variances after PSD adjustment (should ideally not happen with _make_psd)
        # variances[variances < 0] = 0
        # self.logger.debug(f"Number of non-positive variances after PSD adjustment: {np.sum(variances <= 0)}")

        # Calculate standard deviation for each active component
        std_dev = np.sqrt(variances)

        # Expand dimensions for broadcasting: (n_active_components,) -> (n_active_components, 1)
        std_dev = std_dev[:, np.newaxis]

        # Calculate confidence intervals: mean +/- z * std_dev
        ci_lower = mean - z * std_dev
        ci_upper = mean + z * std_dev

        self.logger.debug(f"Computed CI shapes: lower={ci_lower.shape}, upper={ci_upper.shape}")

        # The distinction between 'fixed' and 'free' is not needed here,
        # as the inputs `mean` and `cov` are already specific to the active components.
        # The interpretation of these components (fixed source vs. free orientation component)
        # happens in other functions like _count_values_within_ci.

        return ci_lower, ci_upper

    def _count_values_within_ci(self, x, ci_lower, ci_upper):
        """
        Count the number of ground truth values that lie within the confidence intervals for each time point.
        Assumes input arrays correspond only to the active components.

        Parameters:
        - x (np.ndarray): Ground truth source activity for active components, shape (n_active_components, n_times).
        - ci_lower (np.ndarray): Lower bounds of confidence intervals for active components, shape (n_active_components, n_times).
        - ci_upper (np.ndarray): Upper bounds of confidence intervals for active components, shape (n_active_components, n_times).

        Returns:
        - count_within_ci (np.ndarray): Count of values within confidence intervals.
            - For "fixed" orientation: 1D array (n_times,) with counts per time point.
            - For "free" orientation: 2D array (3, n_times) with counts per orientation (X, Y, Z) per time point.
        """
        if x.shape[0] != len(self.active_set) or ci_lower.shape[0] != len(self.active_set) or ci_upper.shape[0] != len(self.active_set):
             raise ValueError("Input array dimensions do not match the length of the active_set.")

        n_times = x.shape[1]

        if self.orientation_type == "fixed":
            # Sum over all active sources for each time point
            count_within_ci = np.sum((x >= ci_lower) & (x <= ci_upper), axis=0)
            self.logger.debug(f"Fixed orientation counts shape: {count_within_ci.shape}")

        elif self.orientation_type == "free":
            # Initialize counts for each orientation (X, Y, Z) and each time point
            count_within_ci = np.zeros((3, n_times))

            # Determine the orientation for each row in the input arrays based on the original active_set indices
            orientations = self.active_set % 3 # Shape: (n_active_components,)

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


    def _plot_ci_times(self, x, x_hat, active_set, ci_lower, ci_upper, confidence_level, figsize=(20, 15)):
        """
        Plot the estimated source activity with confidence intervals for active components and save them.
        Assumes input arrays correspond only to the active components.

        Parameters:
        - x (np.ndarray): Ground truth source activity for active components, shape (n_active_components, n_times).
        - x_hat (np.ndarray): Estimated source activity for active components, shape (n_active_components, n_times).
        - active_set (np.ndarray): Original indices (flattened for free orientation) of active components, shape (n_active_components,).
        - ci_lower (np.ndarray): Lower bounds of confidence intervals for active components, shape (n_active_components, n_times).
        - ci_upper (np.ndarray): Upper bounds of confidence intervals for active components, shape (n_active_components, n_times).
        - confidence_level (float): Confidence level for the intervals.
        - figsize (tuple): Size of the plot.
        """
        logger = self.logger if hasattr(self, 'logger') and self.logger else logging.getLogger(__name__)

        # Create the base directory for confidence intervals
        confidence_intervals_dir = os.path.join(self.experiment_dir, 'CI')
        os.makedirs(confidence_intervals_dir, exist_ok=True)
        logger.debug(f"Saving CI plots to: {confidence_intervals_dir}")

        n_active_components, n_times = x.shape

        if n_active_components == 0:
            logger.warning("No active components to plot for CI times.")
            return

        if self.orientation_type == "free":
            orientations = ['X', 'Y', 'Z']
            # Map active component index (0 to n_active_components-1) to original source index and orientation
            original_source_indices = active_set // 3
            original_orient_indices = active_set % 3

            for t in range(n_times):
                time_point_dir = os.path.join(confidence_intervals_dir, f't{t}')
                os.makedirs(time_point_dir, exist_ok=True)

                fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, sharey=True) # Share axes
                # Track if legend labels have been added for each subplot
                legend_labels_added = [False, False, False]

                for i in range(n_active_components): # Loop through active components
                    source_idx = original_source_indices[i]
                    orient_idx = original_orient_indices[i]
                    ax = axes[orient_idx]

                    # Determine if labels should be added (only for the first point on each subplot)
                    add_label = not legend_labels_added[orient_idx]

                    # Use source_idx for x-coordinate
                    ax.scatter(source_idx, x_hat[i, t], marker='x', s=50, color='red',
                                label='Posterior Mean' if add_label else "")
                    # Use fill_between for the CI bar
                    ax.fill_between(
                        [source_idx - 2, source_idx + 2], # x-range for the bar
                        ci_lower[i, t],
                        ci_upper[i, t],
                        color='green', # Match scatter color
                        alpha=0.8,
                        label='Confidence Interval' if add_label else ""
                    )
                    ax.scatter(source_idx, x[i, t], s=30, color='blue', alpha=0.7,
                                label='Ground Truth' if add_label else "")

                    # Mark that labels have been added for this subplot
                    if add_label:
                        legend_labels_added[orient_idx] = True

                # Configure axes after plotting all points for this time step
                all_plotted_source_indices = sorted(list(set(original_source_indices)))
                for j, (ax, orient) in enumerate(zip(axes, orientations)):
                    ax.set_title(f'Orientation {orient}')
                    ax.axhline(0, color='grey', lw=0.8, ls='--')

                    # Calculate total unique sources plotted on this specific axis
                    sources_on_this_axis = {original_source_indices[k] for k in range(n_active_components) if original_orient_indices[k] == j}
                    n_sources_this_axis = len(sources_on_this_axis)
                    # Add legend with total sources in the title
                    ax.legend(title=f"Total Sources: {n_sources_this_axis}", loc='best')

                    ax.grid(False)
                    # Set ticks only for sources actually plotted
                    ax.set_xticks(all_plotted_source_indices)
                    ax.set_xticklabels([str(idx) for idx in all_plotted_source_indices], rotation=45, ha='right')
                    # Limit x-axis slightly beyond plotted sources
                    if all_plotted_source_indices:
                            ax.set_xlim(min(all_plotted_source_indices) - 1, max(all_plotted_source_indices) + 1)

                fig.text(0.5, 0.04, 'Original Source Index', ha='center', va='center')
                fig.text(0.04, 0.5, 'Activity', va='center', rotation='vertical')
                fig.suptitle(f'Confidence Intervals (Level={confidence_level:.2f}, Time={t})', fontsize=16)
                plt.tight_layout(rect=[0.05, 0.05, 1, 0.95]) # Adjust rect for titles

                save_path = os.path.join(time_point_dir, f'ci_t{t}_clvl{round(confidence_level, 2)}.png')
                plt.savefig(save_path)
                logger.debug(f"Saved CI plot: {save_path}")
                plt.close(fig)

        else: # Fixed orientation
            original_source_indices = active_set # These are the source indices

            for t in range(n_times):
                time_point_dir = os.path.join(confidence_intervals_dir, f't{t}')
                os.makedirs(time_point_dir, exist_ok=True)

                fig, ax = plt.subplots(figsize=figsize)
                legend_labels_added = False # Track if labels added for this plot

                for i in range(n_active_components): # Loop through active components
                    source_idx = original_source_indices[i]

                    # Determine if labels should be added (only for the first point)
                    add_label = not legend_labels_added

                    # Use source_idx for x-coordinate
                    ax.scatter(source_idx, x_hat[i, t], marker='x', s=50, color='red',
                                label='Posterior Mean' if add_label else "")
                    ax.fill_between(
                        [source_idx - 0.4, source_idx + 0.4], # Adjust width
                        ci_lower[i, t],
                        ci_upper[i, t],
                        color='red',
                        alpha=0.3,
                        label='Confidence Interval' if add_label else ""
                    )
                    ax.scatter(source_idx, x[i, t], s=30, color='blue', alpha=0.7,
                                label='Ground Truth' if add_label else "")

                    # Mark that labels have been added
                    if add_label:
                        legend_labels_added = True

                # Configure axis after plotting
                all_plotted_source_indices = sorted(list(set(original_source_indices)))
                ax.set_title(f'Confidence Intervals (Level={confidence_level:.2f}, Time={t})')
                ax.axhline(0, color='grey', lw=0.8, ls='--')

                # Add legend with total active sources in the title
                ax.legend(title=f'Total Active Sources: {n_active_components}', loc='best')

                ax.grid(False) 
                ax.set_xticks(all_plotted_source_indices)
                ax.set_xticklabels([str(idx) for idx in all_plotted_source_indices], rotation=45, ha='right')
                ax.set_xlabel('Original Source Index')
                ax.set_ylabel('Activity')
                if all_plotted_source_indices:
                        ax.set_xlim(min(all_plotted_source_indices) - 1, max(all_plotted_source_indices) + 1)

                plt.tight_layout(rect=[0.05, 0.05, 1, 0.96]) # Adjust rect

                save_path = os.path.join(time_point_dir, f'ci_t{t}_clvl{round(confidence_level, 2)}.png')
                plt.savefig(save_path)
                logger.debug(f"Saved CI plot: {save_path}")
                plt.close(fig)


    def _plot_proportion_of_hits(
        self,
        confidence_levels,
        CI_count_per_confidence_level,
        total_sources,
        time_point=0,
        filename='proportion_of_hits',
    ):
        """
        Internal method to plot the proportion of hits within confidence intervals for a specific time point.

        Parameters:
        - confidence_levels (list or np.ndarray): Confidence levels to plot.
        - CI_count_per_confidence_level (np.ndarray): Array with counts of values within confidence intervals.
            - For "fixed": shape (n_levels, n_times).
            - For "free": shape (n_levels, 3, n_times).
        - total_sources (int): Total number of sources (denominator for proportion).
                                For 'free', this is typically the number of unique sources.
                                For 'fixed', this is typically the number of active sources.
        - time_point (int): The specific time point to plot.
        - filename (str): Name of the file to save the plot.
        """

        if self.orientation_type == 'free':
            # Create subplots for the three orientations (X, Y, Z)
            fig, axes = plt.subplots(3, 1, figsize=(6, 18), sharex=True, sharey=True)
            orientations = ['X', 'Y', 'Z']

            for i, ax in enumerate(axes):
                # Extract hits for the current orientation and time point
                # Ensure time_point is within bounds
                if time_point >= CI_count_per_confidence_level.shape[2]:
                        self.logger.error(f"time_point {time_point} is out of bounds for CI_count_per_confidence_level with shape {CI_count_per_confidence_level.shape}")
                        plt.close(fig)
                        return
                hits = CI_count_per_confidence_level[:, i, time_point] # Correct indexing order
                proportions = hits / total_sources  # Normalize hits to proportions

                # Plot proportions and diagonal line y=x
                ax.plot(confidence_levels, proportions, marker='o', linestyle='-', color='blue', label='Proportion of Hits')
                ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='y=x')

                # Set axis labels, title, and grid
                ax.set_ylabel('Proportion of Hits')
                ax.grid(True)
                ax.set_xticks(confidence_levels)
                ax.set_xticklabels([f'{cl:.0%}' for cl in confidence_levels]) # Use percentage format
                ax.set_title(f'Orientation {orientations[i]} (Time Point {time_point})')
                ax.legend(loc='lower right')

                # Ensure axes are square
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.05, 1.05) # Corrected typo here
                ax.set_aspect('equal', adjustable='box')

            # Add x-axis label to the last subplot
            axes[-1].set_xlabel('Confidence Level')

            # Add a title for the entire figure
            fig.suptitle(f'Proportion of Hits at Time Point {time_point} (Free Orientation)', fontsize=14)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for the title
            plt.savefig(os.path.join(self.experiment_dir, filename + '.png'))
            plt.close(fig)

        else: # Fixed orientation
            hits = CI_count_per_confidence_level[:, time_point]
            proportions = hits / total_sources  # Normalize hits to proportions

            fig, ax = plt.subplots(figsize=(6, 6))  # Square figure
            ax.plot(confidence_levels, proportions, marker='o', linestyle='-', color='blue', label='Proportion of Hits')
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='y=x')

            # Set axis labels, title, and grid
            ax.set_xlabel('Confidence Level')
            ax.set_ylabel('Proportion of Hits')
            ax.set_title(f'Proportion of Hits at Time Point {time_point} (Fixed Orientation)')
            ax.grid(True)
            ax.set_xticks(confidence_levels)
            ax.set_xticklabels([f'{cl:.0%}' for cl in confidence_levels]) # Use percentage format
            ax.legend(loc='lower right')

            # Ensure axes are square
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect('equal', adjustable='box')

            fig.tight_layout(rect=[0.05, 0.05, 1, 0.96])
            plt.savefig(os.path.join(self.experiment_dir, filename + '.png'))
            plt.close(fig)

        self.logger.info(f"Proportion of hits plot saved to {os.path.join(self.experiment_dir, filename + '.png')}")



    def visualize_confidence_intervals(self, confidence_levels=None, time_point=0):
        """
        Visualize confidence intervals and save the results. Handles both fixed and free orientation.

        Parameters:
        - confidence_levels (list, optional): List of confidence levels to visualize. If None, defaults to 10 levels from 0.1 to 0.99.
        - time_point (int): Time point to visualize for the proportion of hits plot.
        """
        if confidence_levels is None:
            confidence_levels = np.linspace(0.1, 0.99, 10)

        # --- Prepare data based on orientation type ---
        if self.orientation_type == 'free':
            # Shapes: x=(5124, 3, 10), x_hat=(5124, 3, 10), active_set=(1515,), posterior_cov=(1515, 1515)
            n_sources, n_orient, n_times = self.x.shape
            n_total_components = n_sources * n_orient

            # Reshape to (n_total_components, n_times)
            x_proc = self.x.reshape(n_total_components, n_times)
            x_hat_proc = self.x_hat.reshape(n_total_components, n_times)
            self.logger.debug(f"Free orientation: Reshaped x to {x_proc.shape}, x_hat to {x_hat_proc.shape}")

            # Index flattened data using active_set
            active_x = x_proc[self.active_set]         # Shape: (1515, 10)
            active_x_hat = x_hat_proc[self.active_set] # Shape: (1515, 10)

            # Denominator for proportion plot (number of unique sources in active set)
            total_sources_for_plot = len(np.unique(self.active_set // 3))

        elif self.orientation_type == 'fixed':
            # Shapes: x=(5124, 10), x_hat=(5124, 10), active_set=(1515,), posterior_cov=(1515, 1515)
            n_sources, n_times = self.x.shape
            n_total_components = n_sources

            # Data is already 2D
            x_proc = self.x
            x_hat_proc = self.x_hat
            self.logger.debug(f"Fixed orientation: Using x {x_proc.shape}, x_hat {x_hat_proc.shape}")

            # Index data using active_set
            active_x = x_proc[self.active_set]         # Shape: (1515, 10)
            active_x_hat = x_hat_proc[self.active_set] # Shape: (1515, 10)

            # Denominator for proportion plot (number of active sources)
            total_sources_for_plot = len(self.active_set)

        self.logger.debug(f"Indexed data using active_set. Shapes: "
                        f"active_x={active_x.shape}, active_x_hat={active_x_hat.shape}")
        self.logger.debug(f"Total sources for proportion plot denominator: {total_sources_for_plot}")

        # --- Loop through confidence levels ---
        self.logger.info("Computing and creating figures for confidence intervals; for each confidence level and time point. This may take a while...")
        
        CI_count_per_confidence_level = []
        for confidence_level in confidence_levels:
            # Compute CIs using active estimated data and the full posterior_cov for the active set
            # _compute_confidence_intervals should handle orientation internally based on self.orientation_type
            ci_lower, ci_upper = self._compute_confidence_intervals(
                active_x_hat,               # Shape (1515, 10)
                self.posterior_cov,         # Shape (1515, 1515)
                confidence_level=confidence_level
            ) # ci_lower/upper shape depends on _compute_confidence_intervals logic

            # Count hits using active ground truth data
            # _count_values_within_ci should handle orientation internally
            count_within_ci = self._count_values_within_ci(
                active_x,                   # Shape (1515, 10)
                ci_lower,
                ci_upper
            ) # count_within_ci shape depends on _count_values_within_ci logic

            # Plot CIs over time
            # _plot_ci_times should handle orientation internally
            self._plot_ci_times(
                active_x,                   # Shape (1515, 10)
                active_x_hat,               # Shape (1515, 10)
                self.active_set,            # Shape (1515,) - Original indices
                ci_lower,
                ci_upper,
                confidence_level,
            )
            CI_count_per_confidence_level.append(count_within_ci)

        # --- Plot Proportion of Hits ---
        CI_count_per_confidence_level = np.array(CI_count_per_confidence_level)
        self.logger.debug(f"Shape of CI_count_per_confidence_level array: {CI_count_per_confidence_level.shape}")

        # _plot_proportion_of_hits should handle orientation internally
        self._plot_proportion_of_hits(
            confidence_levels=confidence_levels,
            CI_count_per_confidence_level=CI_count_per_confidence_level,
            total_sources=total_sources_for_plot, # Use the calculated denominator
            time_point=time_point,
        )

# ------------------------------
    def plot_source_estimates(self, posterior_cov, orientations):
        """
        Plot source estimates and save the visualizations.

        Parameters:
        - posterior_cov (np.ndarray): Posterior covariance matrix.
        - experiment_dir (str): Path to the experiment directory.
        - orientations (list): List of orientations for visualization.
        """
        posterior_var = np.diag(posterior_cov)
        z_score = self.x_hat[:, 0] / (np.sqrt(np.abs(posterior_var)) + 1e-10)

        stc_x_t0 = mne.SourceEstimate(self.x[:, 0], vertices=self.vertices, tmin=0, tstep=0)
        stc_x_hat_t0 = mne.SourceEstimate(self.x_hat[:, 0], vertices=self.vertices, tmin=0, tstep=0)
        stc_variance = mne.SourceEstimate(posterior_var, vertices=self.vertices, tmin=0, tstep=0)
        stc_zscore = mne.SourceEstimate(z_score, vertices=self.vertices, tmin=0, tstep=0)

        source_estimates = [
            (stc_x_t0, 'Ground Truth'),
            (stc_x_hat_t0, 'Posterior Mean'),
            (stc_variance, 'Posterior Variance'),
            (stc_zscore, 'Z-Score')
        ]

        for stc, title in source_estimates:
            brain = stc.plot(hemi="both", subject='fsaverage', subjects_dir="/Users/orabe/0.braindata/MNE-sample-data/subjects", spacing='ico4', title=title)
            for orientation in orientations:
                orientation_dir = os.path.join(self.experiment_dir, 'brain', orientation)
                os.makedirs(orientation_dir, exist_ok=True)
                brain.show_view(orientation)
                brain.save_image(os.path.join(orientation_dir, f'{title.replace(" ", "_").lower()}_{orientation}.png'))
            brain.close()   
   
