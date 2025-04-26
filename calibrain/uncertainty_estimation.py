
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
        Create a full covariance matrix and validate its shape.

        Returns:
        - full_posterior_cov (np.ndarray): Full posterior covariance matrix.
        """
        x_hat_t0 = self.x_hat[:, 0]
        full_posterior_cov = np.zeros((x_hat_t0.flatten().shape[0],
                                       x_hat_t0.flatten().shape[0]))
        for i, idx_i in enumerate(self.active_set):
            for j, idx_j in enumerate(self.active_set):
                full_posterior_cov[idx_i, idx_j] = self.posterior_cov[i, j]

        non_zero_values = full_posterior_cov[full_posterior_cov != 0]
        assert non_zero_values.size == self.posterior_cov.size == self.active_set.size ** 2
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
        plt.tight_layout()
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
        plt.tight_layout()
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
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, 'top_relevant_CE_pairs.png'))
        plt.close()
        
# ------------------------------
    def plot_active_sources_single_time_step(self, time_step=0):
        """
        Plot the active sources for a single time step, comparing ground truth and estimated sources.
        """
        if self.orientation_type == 'free':
            fig, axes = plt.subplots(3, 1, figsize=(12, 18))
            orientations = ['X', 'Y', 'Z']
            for i, ax in enumerate(axes):
                gt_active_sources = np.where(self.x[:, time_step] != 0)[0] // 3
                gt_amplitudes = self.x[gt_active_sources * 3 + i, time_step]
                est_amplitudes = self.x_hat[self.active_set[i::3], time_step]

                ax.scatter(gt_active_sources, gt_amplitudes, color='blue', alpha=0.6, label='Ground Truth')
                ax.scatter(self.active_set[i::3] // 3, est_amplitudes, color='red', marker='x', alpha=0.6, label='Estimated')
                ax.set_xlabel('Source Index')
                ax.set_ylabel('Amplitude')
                ax.set_title(f'Active Sources for GT and Estimated Sources ({orientations[i]} Orientation)')
                ax.legend(loc='upper left')
                ax.grid(True, alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(self.experiment_dir, 'active_sources_single_time_step.png'))
            plt.close()
        else:
            gt_active_sources = np.where(self.x[:, time_step] != 0)[0]
            gt_amplitudes = self.x[gt_active_sources, time_step]
            est_amplitudes = self.x_hat[self.active_set, time_step]

            plt.figure(figsize=(12, 6))
            plt.scatter(gt_active_sources, gt_amplitudes, color='blue', alpha=0.6, label='Ground Truth')
            plt.scatter(self.active_set, est_amplitudes, color='red', marker='x', alpha=0.6, label='Estimated')
            plt.xlabel('Source Index')
            plt.ylabel('Amplitude')
            plt.title(f'Active Sources for GT and Estimated Sources (Fixed Orientation, time step {time_step})')
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(self.experiment_dir, 'active_sources_single_time_step.png'))
            plt.close()

    def plot_posterior_covariance_matrix(self):
        """
        Plot the posterior covariance matrix.
        """
        if self.orientation_type == 'free':
            fig, axes = plt.subplots(3, 1, figsize=(10, 18))
            orientations = ['X', 'Y', 'Z']
            vmin = np.min(self.posterior_cov)
            vmax = np.max(self.posterior_cov)
            for i, ax in enumerate(axes):
                cov_matrix = self.posterior_cov[i::3, i::3]
                im = ax.imshow(cov_matrix, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
                ax.set_title(f'Orientation {orientations[i]}')
                ax.set_xlabel('Sources')
                ax.set_ylabel('Sources')
            fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
            plt.tight_layout()
            plt.savefig(os.path.join(self.experiment_dir, 'posterior_covariance_matrix.png'))
            plt.close()
            
        else:
            plt.figure(figsize=(10, 8))
            im = plt.imshow(self.posterior_cov, cmap='viridis', aspect='auto')
            plt.colorbar(im, label='Covariance Value')
            plt.title('Posterior Covariance Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(self.experiment_dir, 'posterior_covariance_matrix.png'))
            plt.close()

# ------------------------------
    def _compute_confidence_intervals(self, mean, cov, confidence_level=0.95):
        """
        Compute confidence intervals for a given mean and covariance matrix.
    
        Parameters:
        - mean (np.ndarray): Mean array with shape (n_sources, n_times).
        - cov (np.ndarray): Covariance matrix with shape (n_sources, n_sources).
        - confidence_level (float): Confidence level for the intervals.
    
        Returns:
        - ci_lower (np.ndarray): Lower bounds of confidence intervals with shape (n_sources, n_times).
        - ci_upper (np.ndarray): Upper bounds of confidence intervals with shape (n_sources, n_times).
        """
        z = np.abs(np.percentile(np.random.normal(0, 1, 1000000),
                                 [(1 - confidence_level) / 2 * 100, (1 + confidence_level) / 2 * 100]))[1]
    
        if self.orientation_type == "fixed":
            cov = self._make_psd(cov)
            diag_cov = np.diag(cov)  # Extract diagonal elements of the covariance matrix
            std_dev = np.sqrt(diag_cov)  # Standard deviation for each source
            std_dev = std_dev[:, np.newaxis]  # Expand dimensions to match (n_sources, n_times)
            ci_lower = mean - z * std_dev
            ci_upper = mean + z * std_dev
        elif self.orientation_type == "free":
            ci_lower = np.zeros_like(mean)
            ci_upper = np.zeros_like(mean)
            for i in range(3):
                mean_orient = mean[i::3]
                diag_cov_orient = np.diag(cov[i::3, i::3])
                std_dev_orient = np.sqrt(diag_cov_orient)
                std_dev_orient = std_dev_orient[:, np.newaxis]  # Expand dimensions to match (n_sources, n_times)
                ci_lower[i::3] = mean_orient - z * std_dev_orient
                ci_upper[i::3] = mean_orient + z * std_dev_orient
        else:
            raise ValueError("Unsupported orientation type")
    
        return ci_lower, ci_upper

    def _count_values_within_ci(self, x, ci_lower, ci_upper):
        """
        Count the number of ground truth values that lie within the confidence intervals for each time point.
    
        Parameters:
        - x (np.ndarray): Ground truth source activity with shape (n_sources, n_times).
        - ci_lower (np.ndarray): Lower bounds of confidence intervals with shape (n_sources, n_times).
        - ci_upper (np.ndarray): Upper bounds of confidence intervals with shape (n_sources, n_times).
    
        Returns:
        - count_within_ci (np.ndarray): Count of values within confidence intervals for each time point.
            - For "fixed" orientation: A 1D array with counts for each time point.
            - For "free" orientation: A 2D array with counts for each orientation (X, Y, Z) and each time point.
        """
        if self.orientation_type == "fixed":
            # Count values within confidence intervals for each time point
            count_within_ci = np.sum((x >= ci_lower) & (x <= ci_upper), axis=0)  # Sum over sources for each time point
        elif self.orientation_type == "free":
            # Initialize counts for each orientation (X, Y, Z) and each time point
            n_times = x.shape[1]
            count_within_ci = np.zeros((3, n_times))
            for i in range(3):
                # Extract values for the current orientation
                x_orient = x[i::3, :]  # Shape: (n_sources // 3, n_times)
                ci_lower_orient = ci_lower[i::3, :]  # Shape: (n_sources // 3, n_times)
                ci_upper_orient = ci_upper[i::3, :]  # Shape: (n_sources // 3, n_times)
    
                # Count values within confidence intervals for the current orientation and each time point
                count_within_ci[i, :] = np.sum((x_orient >= ci_lower_orient) & (x_orient <= ci_upper_orient), axis=0)
        else:
            raise ValueError("Unsupported orientation type")
    
        return count_within_ci

    def _plot_ci_times(self, x, x_hat, active_set, ci_lower, ci_upper, confidence_level, figsize=(20, 15)):
        """
        Plot the estimated source activity with confidence intervals and save them in structured folders.
    
        Parameters:
        - x: Ground truth source activity (n_sources, n_times).
        - x_hat: Estimated source activity (n_sources, n_times).
        - active_set: Indices of active sources.
        - ci_lower: Lower bounds of confidence intervals (n_sources, n_times).
        - ci_upper: Upper bounds of confidence intervals (n_sources, n_times).
        - confidence_level: Confidence level for the intervals.
        - figsize: Size of the plot.
        """
        # Create the base directory for confidence intervals
        confidence_intervals_dir = os.path.join(self.experiment_dir, 'CI')
        os.makedirs(confidence_intervals_dir, exist_ok=True)
    
        n_sources, n_times = x.shape
    
        if self.orientation_type == "free":
            # Plot for each time point and orientation
            orientations = ['X', 'Y', 'Z']
            for t in range(n_times):
                # Create a folder for the current time point
                time_point_dir = os.path.join(confidence_intervals_dir, f't{t}')
                os.makedirs(time_point_dir, exist_ok=True)
    
                fig, axes = plt.subplots(3, 1, figsize=figsize, sharey=True)
                for j, (ax, orient) in enumerate(zip(axes, orientations)):
                    for i in range(n_sources // 3):
                        idx = i * 3 + j
                        ax.scatter(i, x_hat[idx, t], marker='x', color='red', label='Posterior Mean' if i == 0 else "")
                        ax.fill_between(
                            [i - 0.5, i + 0.5],
                            ci_lower[idx, t],
                            ci_upper[idx, t],
                            alpha=0.3,
                            label='Confidence Interval' if i == 0 else ""
                        )
                        ax.scatter(i, x[idx, t], s=10, color='blue', label='Ground Truth' if i == 0 else "")
                    ax.set_title(f'Confidence Level {confidence_level:.2f} (Time Point {t}, Orientation {orient})')
                    ax.axhline(0, color='grey', lw=0.8, ls='-')
                    ax.legend(loc='upper right', title=f'(Total Sources: {n_sources // 3})')
                    ax.set_xticks(np.arange(n_sources // 3))
                    ax.set_xticklabels([f'{idx // 3}' for idx in active_set[j::3]], rotation=45)
                fig.text(0.5, 0.04, 'Source index', ha='center')
                fig.text(0.04, 0.5, 'Estimated Activity', va='center', rotation='vertical')
                plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    
                # Save the figure in the time point folder
                save_path = os.path.join(time_point_dir, f'confidence_intervals_t{t}_clvl{round(confidence_level, 2)}.png')
                plt.savefig(save_path)
                plt.close(fig)
        else:
            # Plot for each time point (fixed orientation)
            for t in range(n_times):
                # Create a folder for the current time point
                time_point_dir = os.path.join(confidence_intervals_dir, f't{t}')
                os.makedirs(time_point_dir, exist_ok=True)
    
                fig, ax = plt.subplots(figsize=figsize)
                for i in range(n_sources):
                    ax.scatter(i, x_hat[i, t], marker='x', color='red', label='Posterior Mean' if i == 0 else "")
                    ax.fill_between(
                        [i - 0.5, i + 0.5],
                        ci_lower[i, t],
                        ci_upper[i, t],
                        alpha=0.3,
                        label='Confidence Interval' if i == 0 else ""
                    )
                    ax.scatter(i, x[i, t], s=10, color='blue', label='Ground Truth' if i == 0 else "")
                ax.set_xticks(np.arange(n_sources))
                ax.set_xticklabels([f'{idx}' for idx in active_set], rotation=45)
                ax.set_title(f'Confidence Level {confidence_level:.2f} (Time Point {t})')
                ax.axhline(0, color='grey', lw=0.8, ls='-')
                ax.legend(loc='upper right', title=f'(Total Sources: {n_sources}, Active Sources: {len(active_set)})')
                fig.text(0.5, 0.04, 'Source index', ha='center')
                fig.text(0.04, 0.5, 'Estimated Activity', va='center', rotation='vertical')
                plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    
                # Save the figure in the time point folder
                save_path = os.path.join(time_point_dir, f'confidence_intervals_t{t}_clvl{round(confidence_level, 2)}.png')
                plt.savefig(save_path)
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
        - CI_count_per_confidence_level (np.ndarray): Array of shape (confidence_levels, n_times) with counts of values within confidence intervals.
        - total_sources (int): Total number of sources.
        - time_point (int): The specific time point to plot.
        - filename (str): Name of the file to save the plot.
        """
        save_path = os.path.join(self.experiment_dir, f'{filename}_t{time_point}.png')
            
        if self.orientation_type == 'free':
            # Create subplots for the three orientations (X, Y, Z)
            fig, axes = plt.subplots(3, 1, figsize=(6, 18), sharex=True, sharey=True)
            orientations = ['X', 'Y', 'Z']
    
            for i, ax in enumerate(axes):
                # Extract hits for the current orientation and time point
                hits = CI_count_per_confidence_level[:, time_point, i]
                proportions = hits / total_sources  # Normalize hits to proportions
    
                # Plot proportions and diagonal line y=x
                ax.plot(confidence_levels, proportions, marker='o', linestyle='-', color='blue', label='Proportion of Hits')
                ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='y=x')
    
                # Set axis labels, title, and grid
                ax.set_ylabel('Proportion of Hits')
                ax.grid(True)
                ax.set_xticks(confidence_levels)
                ax.set_xticklabels([f'{int(cl * 100)}%' for cl in confidence_levels])
                ax.set_title(f'Orientation {orientations[i]} (Time Point {time_point})')
                ax.legend(loc='lower right')
    
                # Ensure axes are square
                ax.set_xlim(-0.05, 1.05)  # Add margin to the right of the x-axis
                ax.set_ylim-(0.05, 1.05)  # Add margin to the top of the y-axis
                ax.set_aspect('equal', adjustable='box')
    
            # Add x-axis label to the last subplot
            axes[-1].set_xlabel('Confidence Level')
    
            # Add a title for the entire figure
            fig.suptitle(f'Proportion of Hits at Time Point {time_point} (Free Orientation)', fontsize=14)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for the title
            plt.savefig(save_path)
            plt.close(fig)
    
        else:
            # Fixed orientation: single plot
            hits = CI_count_per_confidence_level[:, time_point]
            proportions = hits / total_sources  # Normalize hits to proportions
    
            # Add artificial zero values to proportions
            # extended_proportions = np.concatenate(([0], proportions, [0]))
    
            fig, ax = plt.subplots(figsize=(6, 6))  # Square figure
            ax.plot(confidence_levels, proportions, marker='o', linestyle='-', color='blue', label='Proportion of Hits')
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='y=x')
    
            # Set axis labels, title, and grid
            ax.set_xlabel('Confidence Level')
            ax.set_ylabel('Proportion of Hits')
            ax.set_title(f'Proportion of Hits at Time Point {time_point} (Fixed Orientation)')
            ax.grid(True)
            ax.set_xticks(confidence_levels)
            ax.set_xticklabels([f'{int(cl * 100)}%' for cl in confidence_levels])
            ax.legend(loc='lower right')
    
            # Ensure axes are square
            ax.set_xlim(-0.05, 1.05)  # Add margin to the right of the x-axis
            ax.set_ylim(-0.05, 1.05)  # Add margin to the top of the y-axis
            ax.set_aspect('equal', adjustable='box')
    
            fig.tight_layout()
            plt.savefig(save_path)
            plt.close(fig)
    
        print(f"Plot saved to {save_path}")
    
    def visualize_confidence_intervals(self, confidence_levels=None, time_point=0):
        """
        Visualize confidence intervals and save the results.
        Parameters:
        - confidence_levels (list, optional): List of confidence levels to visualize. If None, defaults to 10 levels from 0.1 to 0.99.
        - time_point (int): Time point to visualize.
        """
        self.logger.info("Plotting confidence intervals...")

        if confidence_levels is None:
            confidence_levels = np.linspace(0.1, 0.99, 10)
     
        CI_count_per_confidence_level = []
        for confidence_level in confidence_levels:
            ci_lower, ci_upper = self._compute_confidence_intervals(
                self.x_hat[self.active_set],
                self.posterior_cov,
                confidence_level=confidence_level
            )
            count_within_ci = self._count_values_within_ci(
                self.x[self.active_set],
                ci_lower, ci_upper
            )

            self._plot_ci_times(
                self.x[self.active_set],
                self.x_hat[self.active_set],
                self.active_set,
                ci_lower,
                ci_upper,
                confidence_level,
            )
            CI_count_per_confidence_level.append(count_within_ci)

        CI_count_per_confidence_level = np.array(CI_count_per_confidence_level)

        # Call the internal method to plot the proportion of hits
        self._plot_proportion_of_hits(
            confidence_levels=confidence_levels,
            CI_count_per_confidence_level=CI_count_per_confidence_level,
            total_sources=self.x[self.active_set].shape[0],
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
   
