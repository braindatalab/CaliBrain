"""
09. Metric Evaluation
=====================

This tutorial mainly explains the ``MetricEvaluator`` class. It demonstrates
the high-level ``MetricEvaluator`` API for quantitative evaluation of
uncertainty and calibration outputs.

It covers:

- calibration curves via ``MetricEvaluator.calibration_curve``;
- calibration summary metrics via ``MetricEvaluator.calibration_metrics_4``;
- combined error and uncertainty summaries via ``MetricEvaluator.evaluate_all``;
- fixed-orientation and free-orientation EEG examples.
"""



# %%
# Scientific motivation
# ---------------------
#
# After source estimation and uncertainty estimation, CaliBrain needs a compact
# way to answer three questions:
#
# - how accurate are the reconstructed source signals?
# - how large is the posterior uncertainty on average?
# - how well do empirical coverages match nominal coverages?
#
# ``MetricEvaluator`` is the high-level class that summarizes these quantities.
# It wraps ``UncertaintyEstimator`` and exposes workflow-facing evaluation
# methods. This is intentionally more general than the named calibration modes
# such as ``post_oracle``, ``post_pooled``, ``post_pooled_mismatch``, and
# ``post_fixed``: those modes define common recalibration workflows, whereas
# evaluation can be performed in multiple ways once predictions and uncertainty
# summaries are available.

import matplotlib.pyplot as plt
import numpy as np
from mne.io.constants import FIFF

from calibrain import (
    MetricEvaluator,
    SensorSimulator,
    SourceEstimator,
    SourceSimulator,
    UncertaintyEstimator,
    gamma_map_sflex,
)


RANDOM_SEED = 71

# %%
# Build a lightweight evaluation fixture
# --------------------------------------
#
# The tutorial generates two small synthetic examples:
#
# - one fixed-orientation example;
# - one free-orientation EEG example.
#
# Both use the active ``gamma_map_sflex`` solver and the same uncertainty grid.

erp_config = {
    "tmin": -0.1,
    "tmax": 0.8,
    "stim_onset": 0.0,
    "sfreq": 100,
    "fmin": 2,
    "fmax": 8,
    "amplitude_distribution": {
        "median": 8.0,
        "sigma": 0.15,
        "clip": [2.0, 20.0],
    },
    "random_erp_timing": False,
    "erp_min_length": 20,
}

nominal_coverages = np.linspace(0.0, 1.0, 11)
ue = UncertaintyEstimator(nominal_coverages=nominal_coverages)
metric_evaluator = MetricEvaluator(ue)
source_simulator = SourceSimulator(ERP_config=erp_config)
sensor_simulator = SensorSimulator()
times = np.arange(erp_config["tmin"], erp_config["tmax"], 1.0 / erp_config["sfreq"])

rng = np.random.default_rng(RANDOM_SEED)
n_sensors = 16
n_sources = 32
src_coords = rng.normal(scale=0.04, size=(n_sources, 3))

leadfield_fixed = rng.normal(scale=0.03, size=(n_sensors, n_sources))
leadfield_fixed /= np.maximum(
    np.linalg.norm(leadfield_fixed, axis=0, keepdims=True),
    np.finfo(float).eps,
)
leadfield_fixed *= 0.6

leadfield_free_eeg = rng.normal(scale=0.015, size=(n_sensors, n_sources, 3))
leadfield_free_eeg /= np.maximum(
    np.linalg.norm(leadfield_free_eeg, axis=0, keepdims=True),
    np.finfo(float).eps,
)
leadfield_free_eeg *= 0.4

sensor_simulator.set_sensor_metadata(
    kind=FIFF.FIFFV_EEG_CH,
    units=FIFF.FIFF_UNIT_V,
    unitmult=FIFF.FIFF_UNITM_MU,
    coil_type=FIFF.FIFFV_COIL_EEG,
)

# %%
# Fixed-orientation evaluation
# ----------------------------
#
# For fixed orientation, ``MetricEvaluator`` works with:
#
# - ``x_true`` and ``x_hat`` of shape ``(N, T)``;
# - uncertainty as either ``posterior_var`` or full ``posterior_cov``.

x_true_fixed, active_fixed = source_simulator.simulate(
    n_sources=n_sources,
    nnz=4,
    orientation_type="fixed",
    seed=RANDOM_SEED,
)

y_fixed_clean, y_fixed_noisy, fixed_noise, fixed_eta = sensor_simulator.simulate(
    x=x_true_fixed,
    L=leadfield_fixed,
    alpha_SNR=0.7,
    sensor_white_noise_std=0.2,
    seed=RANDOM_SEED,
)
fixed_noise_var = float(np.var(fixed_noise))

fixed_estimator = SourceEstimator(
    solver=gamma_map_sflex,
    solver_params={"max_iter": 150, "tol": 1e-7, "sigma": 0.01, "src_coords": src_coords},
    noise_var=fixed_noise_var,
    n_orient=1,
)
fixed_estimator.fit(leadfield_fixed, y_fixed_noisy)
fixed_result = fixed_estimator.predict()

fixed_curve = metric_evaluator.calibration_curve(
    x_true=x_true_fixed,
    x_hat=fixed_result["posterior_mean"],
    posterior_uncert=fixed_result["posterior_cov"],
    setting="fixed",
    mode="aggregated",
)
fixed_summary = metric_evaluator.evaluate_all(
    x_true=x_true_fixed,
    x_hat=fixed_result["posterior_mean"],
    posterior_uncert=fixed_result["posterior_cov"],
    setting="fixed",
    mode="aggregated",
)

print("fixed calibration curve keys:", sorted(fixed_curve.keys()))
print("fixed calibration metrics:", fixed_curve["metrics_4"])
print("fixed evaluate_all keys:", sorted(fixed_summary.keys()))
print("fixed mean posterior std:", fixed_summary["mean_posterior_std"])

# %%
# Free-orientation EEG evaluation
# -------------------------------
#
# For free-orientation EEG, ``MetricEvaluator`` supports two interval types:
#
# - ``marginal``: pooled component-wise intervals;
# - ``full_cov``: local 3D covariance blocks.
#
# Both are evaluated below in aggregated mode.

x_true_free, active_free = source_simulator.simulate(
    n_sources=n_sources,
    nnz=4,
    orientation_type="free",
    coil_type=FIFF.FIFFV_COIL_EEG,
    seed=RANDOM_SEED + 1,
)

y_free_clean, y_free_noisy, free_noise, free_eta = sensor_simulator.simulate(
    x=x_true_free,
    L=leadfield_free_eeg,
    alpha_SNR=0.7,
    sensor_white_noise_std=0.05,
    seed=RANDOM_SEED + 1,
)
free_noise_var = float(np.var(free_noise))

free_estimator = SourceEstimator(
    solver=gamma_map_sflex,
    solver_params={"max_iter": 150, "tol": 1e-7, "sigma": 0.01, "src_coords": src_coords},
    noise_var=free_noise_var,
    n_orient=3,
)
free_estimator.fit(leadfield_free_eeg, y_free_noisy)
free_result = free_estimator.predict()

free_curve_marginal = metric_evaluator.calibration_curve(
    x_true=x_true_free,
    x_hat=free_result["posterior_mean_reshaped"],
    posterior_uncert=free_result["posterior_cov"],
    setting="eeg_free",
    mode="aggregated",
    free_interval_type="marginal",
)
free_curve_full_cov = metric_evaluator.calibration_curve(
    x_true=x_true_free,
    x_hat=free_result["posterior_mean_reshaped"],
    posterior_uncert=free_result["posterior_cov"],
    setting="eeg_free",
    mode="aggregated",
    free_interval_type="full_cov",
)
free_summary = metric_evaluator.evaluate_all(
    x_true=x_true_free,
    x_hat=free_result["posterior_mean_reshaped"],
    posterior_uncert=free_result["posterior_cov"],
    setting="eeg_free",
    mode="aggregated",
    free_interval_type="full_cov",
)

print("free marginal calibration metrics:", free_curve_marginal["metrics_4"])
print("free full_cov calibration metrics:", free_curve_full_cov["metrics_4"])
print("free evaluate_all mse:", free_summary["mse"])
print("free evaluate_all mean posterior std:", free_summary["mean_posterior_std"])

# %%
# Compare calibration summary metrics directly
# --------------------------------------------
#
# ``calibration_metrics_4`` can also be called directly on nominal and empirical
# coverage arrays. This is useful when calibration curves are already available
# and only the summary metrics need to be recomputed.

fixed_metrics_direct = metric_evaluator.calibration_metrics_4(
    fixed_curve["nominal"],
    fixed_curve["empirical"],
)
free_metrics_direct = metric_evaluator.calibration_metrics_4(
    free_curve_full_cov["nominal"],
    free_curve_full_cov["empirical"],
)

print("fixed direct metrics:", fixed_metrics_direct)
print("free direct metrics:", free_metrics_direct)

# %%
# Plot calibration and evaluation summaries
# -----------------------------------------
#
# The first panel compares the fixed and free-EEG calibration curves. The
# second panel summarizes the default calibration metrics.

metric_names = [
    "mean_absolute_deviation",
    "mean_signed_deviation",
    "max_underconfidence_deviation",
    "max_overconfidence_deviation",
]
fixed_metric_values = [fixed_curve["metrics_4"][name] for name in metric_names]
free_metric_values = [free_curve_full_cov["metrics_4"][name] for name in metric_names]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot([0, 1], [0, 1], "--", color="0.5", label="perfect calibration")
axes[0].plot(fixed_curve["nominal"], fixed_curve["empirical"], marker="o", label="fixed")
axes[0].plot(
    free_curve_full_cov["nominal"],
    free_curve_full_cov["empirical"],
    marker="s",
    label="free EEG full_cov",
)
axes[0].plot(
    free_curve_marginal["nominal"],
    free_curve_marginal["empirical"],
    marker="^",
    label="free EEG marginal",
)
axes[0].set(
    xlabel="Nominal coverage",
    ylabel="Empirical coverage",
    title="Calibration curves",
)
axes[0].legend(loc="best")

bar_positions = np.arange(len(metric_names))
bar_width = 0.38
axes[1].bar(bar_positions - bar_width / 2, fixed_metric_values, width=bar_width, label="fixed")
axes[1].bar(bar_positions + bar_width / 2, free_metric_values, width=bar_width, label="free EEG full_cov")
axes[1].set(
    xticks=bar_positions,
    xticklabels=[
        "MAD",
        "MSD",
        "Max under",
        "Max over",
    ],
    ylabel="Metric value",
    title="Calibration summary metrics",
)
axes[1].legend(loc="best")
fig.tight_layout()

# %%
# Inspect the combined evaluation output
# --------------------------------------
#
# ``evaluate_all`` combines error metrics, uncertainty summary, and calibration
# summary into one dictionary.

for key in ["mse", "mae", "rmse", "rmae", "mean_posterior_std", "calibration"]:
    print(f"fixed evaluate_all[{key!r}] =", fixed_summary[key])

# %%
# Summary
# -------
#
# ``MetricEvaluator`` is the high-level evaluation class used after uncertainty
# estimation and calibration.
#
# In this tutorial it was used to:
#
# - compute aggregated calibration curves;
# - summarize them with the default four calibration metrics;
# - compare ``marginal`` and ``full_cov`` free-EEG uncertainty;
# - collect reconstruction and uncertainty summaries with ``evaluate_all``.
