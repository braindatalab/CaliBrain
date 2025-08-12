
import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from itertools import combinations, zip_longest
from matplotlib.patches import Ellipse
import matplotlib.lines as mlines # For creating custom legend handles
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
        full_posterior_cov = np.zeros((n_total_components, n_total_components), dtype=posterior_cov.dtype)
        self.logger.debug(f"Initialized full_posterior_cov with shape {full_posterior_cov.shape}")
    
        # Embed the active set covariance using nested loops (safe for unsorted active_indices)
        #    Alternative: If performance critical and active_indices is sorted:
        #    idx = np.ix_(self.active_indices, self.active_indices)
        #    full_posterior_cov[idx] = self.posterior_cov
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
    def _make_psd(self, cov: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
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
        mean = self.x_hat[self.active_indices]
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
                    # This assumes active_indices components are ordered [src0_x, src0_y, src0_z, src1_x, ...]
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
    def _compute_confidence_intervals(self, mean : np.ndarray, cov : np.ndarray, confidence_level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """Compute confidence intervals based on the diagonal of the covariance matrix.
        Assumes inputs correspond only to the active components.

        Parameters
        ----------
        mean : np.ndarray
            Mean array for active components, shape (n_sources, n_times).
        cov : np.ndarray
            Covariance matrix for active components, shape (n_sources, n_sources).
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
        # if x.shape[0] != len(self.active_indices) or ci_lower.shape[0] != len(self.active_indices) or ci_upper.shape[0] != len(self.active_indices):
        #      raise ValueError("Input array dimensions do not match the length of the active_indices.")

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


    def _plot_ci_times(self, x, x_hat, ci_lower, ci_upper, confidence_level, figsize=(12, 6)):
        """
        Plot the estimated source activity with confidence intervals for active components and save them.
        Assumes input arrays correspond only to the active components.

        Parameters:
        - x (np.ndarray): Ground truth source activity for active components, shape (n_active_components, n_times).
        - x_hat (np.ndarray): Estimated source activity for active components, shape (n_active_components, n_times).
        - active_indices (np.ndarray): Original indices (flattened for free orientation) of active components, shape (n_active_components,).
        - ci_lower (np.ndarray): Lower bounds of confidence intervals for active components, shape (n_active_components, n_times).
        - ci_upper (np.ndarray): Upper bounds of confidence intervals for active components, shape (n_active_components, n_times).
        - confidence_level (float): Confidence level for the intervals.
        - figsize (tuple): Size of the plot.
        """
        # Create the base directory for confidence intervals
        confidence_intervals_dir = os.path.join(self.experiment_dir, 'CI')
        os.makedirs(confidence_intervals_dir, exist_ok=True)
        self.logger.debug(f"Saving CI plots to: {confidence_intervals_dir}")

        n_active_components, n_times = x.shape

        if n_active_components == 0:
            self.logger.warning("No active components to plot for CI times.")
            return

        if self.orientation_type == "free":
            # TODO: Code has been adapted. It handles fixed orientation correctly, but free orientation needs to be checked.
            
            orientations = ['X', 'Y', 'Z']
            # Map active component index (0 to n_active_components-1) to original source index and orientation
            original_source_indices = self.active_indices // 3
            original_orient_indices = self.active_indices % 3

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
                                label=f'Non-Zero Posterior Mean - Estimated active ({len(self.active_indices)} sources)' if add_label else "")
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
                                label=f'Non-Zero Posterior Mean (({len(self.active_indices)} estimated sources)' if add_label else "")

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
                    ax.legend(title=f"Total Sources: {self.n_total_sources}", loc='best')

                    ax.grid(False)
                    # Set ticks only for sources actually plotted
                    # ax.set_xticks(all_plotted_source_indices)
                    # ax.set_xticklabels([str(idx) for idx in all_plotted_source_indices], rotation=45, ha='right')
                    # Limit x-axis slightly beyond plotted sources
                    if all_plotted_source_indices:
                            ax.set_xlim(min(all_plotted_source_indices) - 1, max(all_plotted_source_indices) + 1)

                fig.text(0.5, 0.04, 'Original Source Index', ha='center', va='center')
                fig.text(0.04, 0.5, 'Activity', va='center', rotation='vertical')
                fig.suptitle(f'Confidence Intervals (Level={confidence_level:.2f}, Time={t})', fontsize=16)
                plt.tight_layout(rect=[0.05, 0.05, 1, 0.95]) # Adjust rect for titles

                save_path = os.path.join(time_point_dir, f'ci_t{t}_clvl{round(confidence_level, 2)}.png')
                plt.savefig(save_path)
                self.logger.debug(f"Saved CI plot: {save_path}")
                plt.close(fig)

        else: # Fixed orientation
            fig, ax = plt.subplots(figsize=figsize)

            ax.scatter(self.active_indices, x_hat, marker='x', s=50, color='red', label=f'Non-Zero Posterior Mean ({len(self.active_indices)} estimated sources)')
            
            ax.scatter(self.x_active_indices, x, s=30, color='blue', alpha=0.7,
                        label=f'Non-Zero Ground Truth ({len(self.x_active_indices)} simulated Sources)')
            
            for i, src_ix in enumerate(self.active_indices):
                ax.fill_between(
                    [src_ix - 10, src_ix + 10], # Adjust width
                    ci_lower[i, 0],
                    ci_upper[i, 0],
                    color='red',
                    alpha=0.3,
                    label='Confidence Interval' if i == 0 else ""
                )

            all_plotted_source_indices = sorted(list(set(self.active_indices)))
            ax.set_title(f'Confidence Intervals (Level={confidence_level:.2f}')
            ax.axhline(0, color='grey', lw=0.8, ls='--')

            ax.legend(title=f'Total Sources: {self.n_total_sources}', loc='best')
            ax.grid(True, alpha=0.5) 
            ax.set_xlabel('Index of Active (Non-zero) Sources')
            ax.set_ylabel('Amplitude of averaged sources (across time) and their estimates')
            ax.set_xlim(min(all_plotted_source_indices) - 1, max(all_plotted_source_indices) + 1)
            plt.tight_layout(rect=[0.05, 0.05, 1, 0.96]) # Adjust rect

            save_path = os.path.join(confidence_intervals_dir, f'clvl{round(confidence_level, 2)}.png')
            plt.savefig(save_path)
            self.logger.debug(f"Saved CI plot: {save_path}")
            plt.close(fig)

    def vizualise_calibration_curve(
        self,
        empirical_coverage,
        results=None, # This dictionary is expected to contain the metrics
        which_legend="active_indices", # or "all_sources"
        filename='calibration_curve' 
    ):
        """
        Visualizes the calibration curve.

        Parameters:
        - empirical_coverage (np.ndarray): 1D array of empirical coverage values,
                                            corresponding to each confidence level in self.confidence_levels.
        - results (dict): Dictionary possibly containing calibration metrics.
        - which_legend (str): Specifies which set of metrics to display in the legend.
        - filename (str): Base name for the saved plot file.
        """            
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot the empirical coverage line and scatter points
        ax.plot(self.confidence_levels, empirical_coverage, label="Empirical Coverage", marker='o', linestyle='-')
        ax.scatter(self.confidence_levels, empirical_coverage, color='blue', s=50, zorder=5)

        # Plot the ideal calibration line (diagonal)
        ax.plot(self.confidence_levels, self.confidence_levels, '--', label="Ideal Calibration", color='gray')
        
        # Fill the area between empirical and ideal calibration
        ax.fill_between(
            self.confidence_levels, 
            empirical_coverage, 
            self.confidence_levels, 
            color='orange', 
            alpha=0.3, 
            label="AUC Deviation Area" # Changed label for clarity
        )
        
        ax.set_xlabel("Nominal Confidence Level")
        ax.set_ylabel("Empirical Coverage")
        ax.set_title(filename.replace('_', ' ').title())
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.set_aspect('equal', adjustable='box')

        # Prepare legend: start with existing plot elements
        handles, labels = ax.get_legend_handles_labels()
        

        # Determine which set of metrics to display
        if which_legend == "active_indices":
            metrics_to_display = {
                'AUC_deviation_active_indices': 'AUC area',
                'max_positive_deviation_active_indices': 'Max Positive Dev.',
                'max_negative_deviation_active_indices': 'Max Negative Dev.',
                'max_absolute_deviation_active_indices': 'Max Abs Dev.',
            }
        elif which_legend == "all_sources":
            metrics_to_display = {
                'AUC_deviation_all_sources': 'AUC area',
                'max_positive_deviation_all_sources': 'Max Positive Dev.',
                'max_negative_deviation_all_sources': 'Max Negative Dev.',
                'max_absolute_deviation_all_sources': 'Max Abs Dev.',
            }
        else:
            self.logger.error(f"Unknown which_legend value: {which_legend}. Expected 'active_indices' or 'all_sources'.")
            return

        if results:
            separator_handle = mlines.Line2D([], [], color='none', marker='', linestyle='None', label="---------------------------")
            handles.append(separator_handle)
            labels.append(separator_handle.get_label())

            for key, display_name in metrics_to_display.items():
                if key in results and results[key] is not None:
                    value = results[key]
                    dummy_handle = mlines.Line2D([], [], color='none', marker='', linestyle='None', label=f"{display_name}: {value:.3f}")
                    handles.append(dummy_handle)
                    labels.append(f"{display_name}: {value:.3f}")

        # Create the legend with potentially added metric values
        ax.legend(handles, labels, loc='best', fontsize='small')
    
        fig.tight_layout(rect=[0.05, 0.05, 1, 0.96]) 

        save_path = os.path.join(self.experiment_dir, f"{filename}.png")
        plt.savefig(save_path)
        plt.close(fig)

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
        all_ci_lower_list = []
        all_ci_upper_list = []
        collected_counts_within_ci_list = []

        self.logger.info("Computing confidence intervals and hit counts for each confidence level.")
        for cl_idx, confidence_level_val in enumerate(self.confidence_levels):
            self.logger.debug(f"Processing confidence level {cl_idx + 1}/{len(self.confidence_levels)}: {confidence_level_val:.2f}")
            
            # x_hat here is the active_x_hat
            # self.posterior_cov is the covariance for the active set
            ci_lower, ci_upper = self._compute_confidence_intervals(
                mean=x_hat, 
                cov=posterior_cov, 
                confidence_level=confidence_level_val
            )

            # x here is the active_x
            count_within_ci = self._count_values_within_ci(
                # x[self.x_active_indices],
                # ci_lower[self.x_active_indices],
                # ci_upper[self.x_active_indices]
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
        
        # return ci_lower_stacked[:, self.active_indices], ci_upper_stacked[:, self.active_indices], counts_array
        return ci_lower_stacked, ci_upper_stacked, counts_array

    def visualize_confidence_intervals(self, ci_lower, ci_upper, x, x_hat):
        """
        Visualizes confidence intervals over time for active components, using pre-computed data.

        Parameters:
        - x (np.ndarray): Ground truth source activity for active components.
            Shape (n_active_components, n_times).
        - x_hat (np.ndarray): Estimated source activity for active components.
            Shape (n_active_components, n_times).
        - confidence_intervals_tuple (tuple): 
            Tuple returned by get_confidence_intervals_data, containing (ci_lower_stacked, ci_upper_stacked).
        """
        self.logger.info("Plotting CI values for each confidence level. This may take a while...")
        for idx, confidence_level_val in enumerate(self.confidence_levels):
            self.logger.debug(f"Plotting CI times for confidence level: {confidence_level_val:.2f}")
            ci_lower_current = ci_lower[idx] 
            ci_upper_current = ci_upper[idx] 
            
            self._plot_ci_times(
                x,
                x_hat,
                ci_lower_current,
                ci_upper_current,
                confidence_level_val,
            )
        self.logger.info("CI times visualization process finished.")


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
   
