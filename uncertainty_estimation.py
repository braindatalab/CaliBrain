import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

def construct_source_wise_credible_intervals(x_hat, posterior_cov, confidence_level=0.95):
    """
    Construct credible intervals for each source individually (i.e., marginal distributions).
    
    Parameters:
    - x_hat: (n_sources, n_times) predicted source time courses
    - posterior_cov: (n_sources, n_sources) full posterior covariance matrix
    - confidence_level: float, nominal confidence level (e.g., 0.95)
    
    Returns:
    - ci_lower: (n_sources,) lower bounds of credible intervals for each source
    - ci_upper: (n_sources,) upper bounds of credible intervals for each source
    - x_hat_avg: (n_sources,) time-averaged source estimates for each source
    - std_avg: (n_sources,) time-averaged posterior standard deviations for each source
    """
    # Time-average the source estimates for each source
    x_hat_avg = np.mean(x_hat, axis=1)
    
    # Compute time-averaged posterior standard deviation for each source
    n_times = x_hat.shape[1]
    posterior_variance = np.diag(posterior_cov)  # Extract variance for each source
    std_avg = np.sqrt(posterior_variance / n_times)  # Time-averaged standard deviation for each source
    
    # Calculate critical value (i.e., z-score)
    z = norm.ppf(0.5 + confidence_level / 2)
    
    # Construct credible intervals for each source independently
    ci_lower = x_hat_avg - z * std_avg  # Lower bound for each source
    ci_upper = x_hat_avg + z * std_avg  # Upper bound for each source
    
    return ci_lower, ci_upper, x_hat_avg, std_avg

def compute_calibration_curve(x_hat, posterior_cov, x_true, confidence_levels=None):
    """
    Compute the calibration curve for source estimators using source-wise credible intervals.
    
    Parameters:
    - x_hat: (n_sources, n_times) predicted source time courses
    - posterior_cov: (n_sources, n_sources) full covariance matrix.
    - x_true: (n_sources, n_times) ground truth source time courses
    - confidence_levels: array-like, nominal confidence levels to evaluate
    
    Returns:
    - nominal_confidences: np.ndarray of confidence levels
    - empirical_coverages: np.ndarray of empirical coverages
    - std_avg: (n_sources,) time-averaged posterior standard deviations
    """
    if confidence_levels is None:
        confidence_levels = np.linspace(0.5, 0.95, 10)
    
    # Get time-averaged estimates and std
    _, _, x_hat_avg, std_avg = construct_source_wise_credible_intervals(
        x_hat, posterior_cov, 0.5  # Confidence level doesn't matter for getting mean and std
    )
    
    # Time-average the true sources (i.e., ground truth sources)
    x_true_avg = np.mean(x_true, axis=1)
    
    nominal_confidences = np.array(confidence_levels)
    empirical_coverages = np.zeros_like(nominal_confidences)
    
    for i, c in enumerate(nominal_confidences):
        # Calculate critical value for this confidence level
        z = norm.ppf(0.5 + c / 2)
        
        # Construct credible intervals using precomputed values
        ci_lower = x_hat_avg - z * std_avg
        ci_upper = x_hat_avg + z * std_avg
        
        # Check coverage
        within_ci = (x_true_avg >= ci_lower) & (x_true_avg <= ci_upper)
        empirical_coverages[i] = np.mean(within_ci)
    
    return nominal_confidences, empirical_coverages

def plot_calibration_curve(nominal_confidences, empirical_coverages, title="Calibration Curve"):
    """
    Plot the calibration curve comparing nominal vs. empirical confidence.

    Parameters
    ----------
    nominal_confidences : ndarray
        X-axis values (nominal confidence levels).
    empirical_coverages : ndarray
        Y-axis values (empirical coverages).
    title : str
        Title of the plot.
    """
    smooth_x = np.linspace(0.01, 0.99, 300)
    spline = make_interp_spline(nominal_confidences, empirical_coverages, k=2)
    smooth_y = spline(smooth_x)

    plt.figure(figsize=(7, 6))
    plt.plot(smooth_x, smooth_y, label='Calibration Curve', color='blue')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    plt.xlabel('Nominal Confidence Level')
    plt.ylabel('Empirical Coverage')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_calibration_curve2(
    self,
    confidence_levels,
    empirical_coverage,
    result=None, # This dictionary is expected to contain the metrics
    which_legend="active_indices", # or "all_sources"
    file_name='calibration_curve',
    save_path=None,
    show=True,
):
    """
    Visualizes the calibration curve.

    Parameters:
    - empirical_coverage (np.ndarray): 1D array of empirical coverage values,
                                        corresponding to each confidence level in self.confidence_levels.
    - results (dict): Dictionary possibly containing calibration metrics.
    - which_legend (str): Specifies which set of metrics to display in the legend.
    - file_name (str): Base name for the saved plot file.
    """            
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the empirical coverage line and scatter points
    ax.plot(confidence_levels, empirical_coverage, label="Empirical Coverage", marker='o', linestyle='-')
    ax.scatter(confidence_levels, empirical_coverage, color='blue', s=50, zorder=5)

    # Plot the ideal calibration line (diagonal)
    ax.plot(confidence_levels, confidence_levels, '--', label="Ideal Calibration", color='gray')

    # Fill the area between empirical and ideal calibration
    ax.fill_between(
        confidence_levels, 
        empirical_coverage, 
        confidence_levels, 
        color='orange', 
        alpha=0.3, 
        label="AUC Deviation Area"
    )
    
    ax.set_xlabel("Nominal Confidence Level")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title(file_name.replace('_', ' ').title())
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.set_aspect('equal', adjustable='box')

    # Prepare legend: start with existing plot elements
    handles, labels = ax.get_legend_handles_labels()
    
    # Determine which set of metrics to display
    if which_legend == "active_indices":
        metrics_to_display = {
            'auc_deviation_active_indices': 'AUC area',
            'max_positive_deviation_active_indices': 'Max Positive Dev.',
            'max_negative_deviation_active_indices': 'Max Negative Dev.',
            'max_absolute_deviation_active_indices': 'Max Abs Dev.',
        }
    elif which_legend == "all_sources":
        metrics_to_display = {
            'auc_deviation_all_sources': 'AUC area',
            'max_positive_deviation_all_sources': 'Max Positive Dev.',
            'max_negative_deviation_all_sources': 'Max Negative Dev.',
            'max_absolute_deviation_all_sources': 'Max Abs Dev.',
        }
    else:
        self.logger.error(f"Unknown which_legend value: {which_legend}. Expected 'active_indices' or 'all_sources'.")
        return

    if result:
        separator_handle = mlines.Line2D([], [], color='none', marker='', linestyle='None', label="---------------------------")
        handles.append(separator_handle)
        labels.append(separator_handle.get_label())

        for key, display_name in metrics_to_display.items():
            if key in result and result[key] is not None:
                value = result[key]
                dummy_handle = mlines.Line2D([], [], color='none', marker='', linestyle='None', label=f"{display_name}: {value:.3f}")
                handles.append(dummy_handle)
                labels.append(f"{display_name}: {value:.3f}")

    # Create the legend with potentially added metric values
    ax.legend(handles, labels, loc='best', fontsize='small')
    fig.tight_layout(rect=[0.05, 0.05, 1, 0.96]) 

    self._handle_figure_output(fig, file_name, save_path, show)



# Example usage with random data
n_sources = 20
n_times = 100

np.random.seed(42)
x_hat = np.random.randn(n_sources, n_times)
posterior_cov = np.diag(np.abs(np.random.rand(n_sources)))  # Diagonal covariance for simplicity
x_true = np.random.randn(n_sources, n_times)

nominal_confidences, empirical_coverages = compute_calibration_curve(x_hat, posterior_cov, x_true)

plot_calibration_curve(nominal_confidences, empirical_coverages, title="Calibration Curve (Random Data)")

plot_calibration_curve2(
    None,
    nominal_confidences,
    empirical_coverages,
    # result={
    #     'auc_deviation_active_indices': 0.05,
    #     'max_positive_deviation_active_indices': 0.1,
    #     'max_negative_deviation_active_indices': 0.08,
    #     'max_absolute_deviation_active_indices': 0.1,
    # },
    which_legend="active_indices",
    file_name='calibration_curve_example',
    save_path=None,
    show=True,
)