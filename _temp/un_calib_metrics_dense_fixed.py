# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 15:29:12 2025

@author: User
"""


# --- After running eLORETA ---
# x_hat: (n_sources, n_times)
# posterior_cov: (n_sources, n_sources)
# x: (n_sources, n_times)    # simulated ground truth sources


# =============================================================================
# UNCERTAINTY QUANTIFICATION AND CALIBRATION METRICS FOR DENSE SOLVERS 
# =============================================================================
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# -------------------------------
# 1. Compute Calibration Curve
# -------------------------------

def compute_calibration_curve_dense(x_hat, posterior_cov, x_true, confidence_levels=None):
    """
    Compute the calibration curve for dense estimators (e.g., eLORETA) using all sources.

    Parameters:
    - x_hat: (n_sources, n_times) predicted source time courses
    - posterior_cov: (n_sources, n_sources) full covariance matrix over all sources
    - x_true: (n_sources, n_times) ground truth source time courses
    - confidence_levels: array-like, nominal confidence levels to evaluate

    Returns:
    - nominal_confidences: np.ndarray of confidence levels
    - empirical_coverages: np.ndarray of empirical coverages (over all sources)
    - std: np.ndarray of posterior std devs (over all sources)
    """
    if confidence_levels is None:
        confidence_levels = np.linspace(0.5, 0.95, 10)

    # Mean over time
    x_hat_avg = np.mean(x_hat, axis=1)
    x_true_avg = np.mean(x_true, axis=1)

    # Time-averaged posterior std
    posterior_variance = np.diag(posterior_cov)
    n_times = x_hat.shape[1]
    std = np.sqrt(posterior_variance / n_times)

    nominal_confidences = np.array(confidence_levels)
    empirical_coverages = np.zeros_like(nominal_confidences)

    for i, c in enumerate(nominal_confidences):
        z = norm.ppf(0.5 + c / 2)
        ci_lower = x_hat_avg - z * std
        ci_upper = x_hat_avg + z * std
        within_ci = (x_true_avg >= ci_lower) & (x_true_avg <= ci_upper)
        empirical_coverages[i] = np.mean(within_ci)

    return nominal_confidences, empirical_coverages, std


# -------------------------------
# 2. Plot Calibration Curve
# -------------------------------

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


# -------------------------------
# 3. Compute Uncertainty & Calibration Metrics
# -------------------------------

def compute_uncertainty_and_calibration_metrics_dense(nominal_confidences, empirical_coverages, std):
    """
    Compute scalar metrics for uncertainty and calibration (dense solvers).

    Parameters:
    - nominal_confidences (np.ndarray): Nominal confidence levels.
    - empirical_coverages (np.ndarray): Corresponding empirical coverages.
    - std (np.ndarray): Posterior standard deviations (over all sources).

    Returns:
    - dict: Dictionary of scalar metrics including calibration and uncertainty.
    """
    deviations = empirical_coverages - nominal_confidences
    aad = np.mean(np.abs(deviations))         # Average Absolute Deviation
    asd = np.mean(deviations)                 # Average Signed Deviation
    max_dev = np.max(deviations)              # Maximum Underconfidence
    neg_min_dev = -np.min(deviations)         # Maximum Overconfidence
    mean_std = np.mean(std)                   # Uncertainty metric

    return {
        "AAD": aad,
        "ASD": asd,
        "Max Underconfidence": max_dev,
        "Max Overconfidence": neg_min_dev,
        "Mean Posterior Std (All)": mean_std
    }


nom_conf, emp_cov, std = compute_calibration_curve_dense(x_hat, posterior_cov, x)
plot_calibration_curve(nom_conf, emp_cov)
metrics = compute_uncertainty_and_calibration_metrics_dense(nom_conf, emp_cov, std)
print("Uncertainty and Calibration Metrics (Dense):", metrics)
