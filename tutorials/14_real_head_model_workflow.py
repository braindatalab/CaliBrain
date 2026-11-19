"""
14. Workflow with a Real Head Model
===================================

The previous tutorial ran the whole calibration chain on a small synthetic
leadfield. This tutorial runs the same chain on a real head model: the
``fsaverage`` template, with 1284 cortical sources and 64 EEG channels. It then
shows what the free-orientation simulation does with the real cortical normals.

The example data are downloaded on first use (about 39 MB, see
:doc:`Leadfield Construction </auto_tutorials/05_leadfield_building>`), and the
run takes about 20 seconds.
"""

# %%
# Get the example data
# --------------------
#
# Getting the data ready takes the three independent steps from the leadfield
# tutorial: download the forward solution, reduce it to 1284 sources, then
# create the leadfield files from it. Each is skipped when its files already exist.

import matplotlib.pyplot as plt
import numpy as np

from calibrain import (
    LeadfieldBuilder,
    MetricEvaluator,
    SensorSimulator,
    SourceEstimator,
    SourceSimulator,
    UncertaintyCalibrator,
    UncertaintyEstimator,
    extract_leadfields,
    fetch_example_data,
    gamma_map_sflex,
    reduce_forward_solutions,
)
from calibrain.utils import get_leadfield_dir

RANDOM_SEED = 91

leadfield_dir = get_leadfield_dir()
fetch_example_data()  # step 1: download the forward solution
reduce_forward_solutions("fsaverage")  # step 2: reduce it to 1284 sources
if not (leadfield_dir / "fsaverage_free_leadfield.npz").exists():
    extract_leadfields("fsaverage")  # step 3: create the leadfield files

# %%
# Load the real leadfield
# -----------------------
#
# ``LeadfieldBuilder`` in ``load`` mode returns the leadfield together with its
# metadata: the sensor units, the source coordinates, and for the free
# orientation the cortical normals. The EEG leadfield is scaled to
# :math:`\mu V / nAm`.

builder = LeadfieldBuilder(leadfield_dir=leadfield_dir)
fixed = builder.get_leadfield(
    subject="fsaverage", orientation_type="fixed", retrieve_mode="load", return_metadata=True
)
free = builder.get_leadfield(
    subject="fsaverage", orientation_type="free", retrieve_mode="load", return_metadata=True
)

L = fixed.leadfield
src_coords = fixed.src_coords
n_sensors, n_sources = L.shape

print("fixed leadfield shape:", L.shape)
print("free leadfield shape:", free.leadfield.shape)
print("source coordinates shape:", src_coords.shape)

# %%
# The sources lie on the real cortical surface of the template, 642 per
# hemisphere. The two views below show their coordinates.

fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
for ax, (i, j, title) in zip(axes, [(0, 1, "top view (x, y)"), (0, 2, "side view (x, z)")]):
    ax.scatter(src_coords[:, i] * 100, src_coords[:, j] * 100, s=6, color="0.4")
    ax.set(xlabel="x (cm)", ylabel=("y (cm)" if j == 1 else "z (cm)"), title=title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle="--", alpha=0.3)
fig.tight_layout()

# %%
# Simulate, estimate, and calibrate
# ---------------------------------
#
# The chain is the one from the end-to-end tutorial: source activity, sensor
# data, posterior estimation with ``gamma_map_sflex`` (which also uses the source
# coordinates), then an uncertainty representation and isotonic recalibration. A
# small helper builds one dataset from a seed. The train and eval splits differ
# only in the seed, as in the ``post_oracle`` setting.

erp_config = {
    "tmin": -0.1,
    "tmax": 0.8,
    "stim_onset": 0.0,
    "sfreq": 100,
    "fmin": 2,
    "fmax": 8,
    "amplitude_distribution": {"median": 8.0, "sigma": 0.15, "clip": [2.0, 20.0]},
    "random_erp_timing": False,
    "erp_min_length": 20,
}
nnz = 5
alpha_snr = 0.7

source_simulator = SourceSimulator(ERP_config=erp_config)
sensor_simulator = SensorSimulator()
sensor_simulator.set_sensor_metadata(
    kind=fixed.sensor_kind,
    units=fixed.sensor_units,
    unitmult=fixed.sensor_unitmult,
    coil_type=fixed.coil_type,
)
uncertainty_estimator = UncertaintyEstimator(nominal_coverages=np.linspace(0.0, 1.0, 11))
metric_evaluator = MetricEvaluator(uncertainty_estimator)


def simulate_and_estimate(seed):
    """Simulate one dataset on the real leadfield and estimate the sources."""
    x_true, active = source_simulator.simulate(
        n_sources=n_sources, nnz=nnz, orientation_type="fixed", seed=seed
    )
    _, y_noisy, noise, _ = sensor_simulator.simulate(
        x=x_true, L=L, alpha_SNR=alpha_snr, sensor_white_noise_std=0.2, seed=seed
    )
    noise_var = max(float(np.var(noise)), 1e-12)
    estimator = SourceEstimator(
        solver=gamma_map_sflex,
        solver_params={"max_iter": 100, "tol": 1e-6, "sigma": 0.01, "src_coords": src_coords},
        noise_var=noise_var,
        n_orient=1,
    )
    estimator.fit(L, y_noisy)
    result = estimator.predict()
    return {
        "orientation_type": "fixed",
        "coil_type": None,
        "x_true": x_true,
        "x_hat": result["posterior_mean"],
        "posterior_cov": result["posterior_cov"],
        "posterior_var": uncertainty_estimator.posterior_variance_from_cov(result["posterior_cov"]),
        "n_sources": x_true.shape[0],
        "n_times": x_true.shape[1],
        "seed": seed,
        "nnz": nnz,
        "alpha_SNR": alpha_snr,
        "solver": "gamma_map_sflex",
        "noise_type": "oracle",
        "active_sources": active,
        "noise_var": noise_var,
    }


train_dataset = simulate_and_estimate(RANDOM_SEED)
eval_dataset = simulate_and_estimate(RANDOM_SEED + 1)

print("estimated sources shape:", eval_dataset["x_hat"].shape)
print("posterior covariance shape:", eval_dataset["posterior_cov"].shape)

# %%
# The estimate on the eval split, next to the sources that were really active.
# The color shows the size of the estimated activity (root mean square over
# time) at every source.

active = eval_dataset["active_sources"]
estimated_size = np.sqrt((eval_dataset["x_hat"] ** 2).mean(axis=1))

fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
for ax, (i, j, title) in zip(axes, [(0, 1, "top view (x, y)"), (0, 2, "side view (x, z)")]):
    points = ax.scatter(
        src_coords[:, i] * 100, src_coords[:, j] * 100, c=estimated_size, s=10, cmap="viridis"
    )
    ax.scatter(
        src_coords[active, i] * 100,
        src_coords[active, j] * 100,
        s=90,
        facecolors="none",
        edgecolors="red",
        label="true active sources",
    )
    ax.set(xlabel="x (cm)", ylabel=("y (cm)" if j == 1 else "z (cm)"), title=title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=8)
fig.colorbar(points, ax=axes, label="estimated activity (nAm, rms over time)", shrink=0.8)

# %%
# Coverage before and after calibration
# -------------------------------------
#
# The aggregated interval routine gives the empirical coverage for every nominal
# level. The calibrator fits an isotonic map on the train split and applies it to
# the eval split.

pre_curve = uncertainty_estimator.calibration_curve_intervals_aggregated(
    x_true=eval_dataset["x_true"],
    x_hat=eval_dataset["x_hat"],
    posterior_var=eval_dataset["posterior_var"],
)
calibrator = UncertaintyCalibrator(uncertainty_estimator, metric_evaluator)
calibration_result = calibrator.calibrate(
    train_data=train_dataset, test_data=eval_dataset, fit=True
)
post_block = calibration_result["post_calibration"]

print("pre empirical coverages: ", np.round(pre_curve["empirical_coverages"], 2))
print("post empirical coverages:", np.round(post_block["empirical_coverages"], 2))

fig, ax = plt.subplots(figsize=(4.6, 4.4))
ax.plot([0, 1], [0, 1], "--", color="0.5", label="perfect calibration")
ax.plot(pre_curve["nominal_coverages"], pre_curve["empirical_coverages"], "o-", label="pre-calibration")
ax.plot(post_block["nominal_coverages"], post_block["empirical_coverages"], "s-", label="post-calibration")
ax.set(xlabel="Nominal coverage", ylabel="Empirical coverage", xlim=(0, 1), ylim=(0, 1), title="Real head model")
ax.set_aspect("equal", adjustable="box")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(loc="best")
fig.tight_layout()

# %%
# Read the figure against the dashed diagonal, which is perfect calibration.
# With these seeds, the intervals before calibration are conservative: the
# empirical coverage is above the nominal level at most levels. After calibration
# the curve lies much closer to the diagonal.

# %%
# Free orientation along the real cortical normals
# ------------------------------------------------
#
# In a free-orientation simulation each active source carries one waveform along
# its cortical normal, so every active source block has rank one. The loaded
# free leadfield provides the real normals (and, for MEG, the local basis).

x_free, active_free = source_simulator.simulate(
    n_sources=n_sources,
    nnz=nnz,
    orientation_type="free",
    coil_type=free.coil_type,
    seed=RANDOM_SEED,
    cortical_normals=free.cortical_normals,
    Q_basis=free.Q_basis,
)

source = int(active_free[0])
block = x_free[source]  # (3 components, n_times)
u, singular_values, _ = np.linalg.svd(block, full_matrices=False)

print("free simulation shape:", x_free.shape)
print("singular values of one active source block:", np.round(singular_values, 3))
print("cosine between its direction and the cortical normal:", round(abs(u[:, 0] @ free.cortical_normals[source]), 3))

# %%
# For EEG the free leadfield applied along the normal is the fixed leadfield, so
# the sensor data of the free simulation equal those of the fixed simulation with
# the same seed (up to the single precision of the stored forward solution).

x_fixed, active_fixed = source_simulator.simulate(
    n_sources=n_sources, nnz=nnz, orientation_type="fixed", seed=RANDOM_SEED
)
y_fixed = L @ x_fixed
y_free = np.einsum("mnk,nkt->mt", free.leadfield, x_free)

print("same active sources:", np.array_equal(np.sort(active_fixed), np.sort(active_free)))
print("largest relative sensor difference:", f"{np.abs(y_free - y_fixed).max() / np.abs(y_fixed).max():.1e}")

# %%
# Summary
# -------
#
# With the real ``fsaverage`` head model the workflow is the same as with the
# synthetic one, only larger:
#
# - the example data were downloaded and prepared in three independent steps;
# - ``LeadfieldBuilder`` loaded a real leadfield with its coordinates, units, and
#   cortical normals;
# - simulation, estimation, uncertainty, and calibration ran on 1284 real source
#   positions;
# - the free-orientation simulation placed rank-one activity along the real
#   normals, consistent with the fixed leadfield.
#
# The four CamCAN subjects used by the default workflow configurations are a
# separate dataset, ``fetch_example_data("camcan")``, with MEG forward solutions.
# They are subject to the CamCAN terms (see the Dataset Notice). The orientation
# tutorial shows how the uncertainty representations differ for free orientation.
