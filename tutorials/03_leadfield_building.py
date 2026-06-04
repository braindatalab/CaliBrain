"""
03. Building and Loading Leadfields
===================================

This tutorial demonstrates the three leadfield retrieval modes used by
``LeadfieldBuilder``:

- ``retrieve_mode="random"``: create a synthetic random leadfield;
- ``retrieve_mode="simulate"``: run a simulation pipeline;
- ``retrieve_mode="load"``: load a stored leadfield NPZ file.

The full MNE simulation pipeline requires subject surfaces, BEM files, sensor
information, and forward-solution configuration. To keep this tutorial
executable during documentation builds, the simulate example uses a small
subclass that returns a deterministic synthetic leadfield while preserving the
same ``get_leadfield(..., retrieve_mode="simulate")`` interface.
"""

# sphinx_gallery_thumbnail_number = 1

# %%
# Scientific motivation
# ---------------------
#
# A leadfield maps source amplitudes to sensor measurements. In fixed
# orientation, the leadfield has shape ``(n_sensors, n_sources)``. In free
# orientation, CaliBrain represents it as ``(n_sensors, n_sources, n_components)``.
#
# Units matter because the leadfield connects source dipole moments to sensor
# measurements:
#
# - source amplitudes are represented in ``nAm``;
# - MEG magnetometer leadfields are typically reported in ``fT / nAm``;
# - EEG leadfields are typically reported in ``µV / nAm``.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mne.io.constants import FIFF

from calibrain import LeadfieldBuilder
from calibrain.utils import get_data_path


RANDOM_SEED = 23
OUTPUT_DIR = Path("results/tutorials/03_leadfield_building")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
# Use the CaliBrain data-root helper
# ----------------------------------
#
# CaliBrain resolves local datasets with ``get_data_path``. With no argument,
# it uses ``CALIBRAIN_DATA`` if the environment variable is set, otherwise the
# repository-level ``data`` directory. Here we pass a tutorial-specific path so
# the example is self-contained and does not depend on local paper-scale data.

data_root = get_data_path(OUTPUT_DIR / "example_dataset")
leadfield_dir = data_root / "1284src_leadfield"
leadfield_dir.mkdir(parents=True, exist_ok=True)

print("tutorial data root:", data_root)
print("tutorial leadfield directory:", leadfield_dir)

# %%
# Random retrieval mode
# ---------------------
#
# ``retrieve_mode="random"`` is useful for lightweight tests. It does not
# represent a physical head model. The generated values are arbitrary unless
# explicit metadata and scaling are attached later.

builder = LeadfieldBuilder(leadfield_dir=leadfield_dir)

random_fixed = builder.get_leadfield(
    subject="demo",
    orientation_type="fixed",
    retrieve_mode="random",
    n_sensors=12,
    n_sources=24,
    return_metadata=True,
)

random_free = builder.get_leadfield(
    subject="demo",
    orientation_type="free",
    retrieve_mode="random",
    n_sensors=12,
    n_sources=24,
    return_metadata=True,
)

print("random fixed shape:", random_fixed.leadfield.shape)
print("random free shape:", random_free.leadfield.shape)

# %%
# Simulate retrieval mode
# -----------------------
#
# In production, ``retrieve_mode="simulate"`` runs the full MNE-based pipeline:
# source space, BEM model, sensor info, forward solution, and leadfield
# extraction. The subclass below provides a tiny deterministic simulation for
# documentation. It sets MEG metadata so the result is interpretable as
# ``fT / nAm``.


class DemoLeadfieldBuilder(LeadfieldBuilder):
    """Small deterministic leadfield simulator for documentation examples."""

    def __init__(self, *, leadfield_dir, seed=RANDOM_SEED):
        super().__init__(config=None, leadfield_dir=leadfield_dir)
        self.seed = seed

    def simulate(self):
        rng = np.random.default_rng(self.seed)
        n_sensors = 12
        n_sources = 24
        source_positions = rng.normal(scale=0.04, size=(n_sources, 3))
        sensor_positions = rng.normal(scale=0.08, size=(n_sensors, 3))
        distances = np.linalg.norm(
            sensor_positions[:, None, :] - source_positions[None, :, :],
            axis=2,
        )
        leadfield = 1.0 / np.maximum(distances, 0.02) ** 2
        leadfield *= rng.choice([-1.0, 1.0], size=leadfield.shape)
        leadfield /= np.linalg.norm(leadfield, axis=0, keepdims=True)

        self.sensor_kind = FIFF.FIFFV_MEG_CH
        self.sensor_units = FIFF.FIFF_UNIT_T
        self.sensor_unitmult = FIFF.FIFF_UNITM_F
        self.coil_type = FIFF.FIFFV_COIL_VV_MAG_T1
        self.src_coords = source_positions
        self.Q_basis = np.eye(n_sources)
        return leadfield


simulated_builder = DemoLeadfieldBuilder(leadfield_dir=leadfield_dir)
simulated = simulated_builder.get_leadfield(
    subject="demo",
    orientation_type="fixed",
    retrieve_mode="simulate",
    return_metadata=True,
)

print("simulated shape:", simulated.leadfield.shape)
print("simulated sensor unit:", simulated.sensor_units)
print("simulated sensor unit multiplier:", simulated.sensor_unitmult)
print("simulated coil type:", simulated.coil_type)

# %%
# Store a leadfield for load mode
# -------------------------------
#
# ``retrieve_mode="load"`` expects NPZ files named
# ``<subject>_<orientation>_leadfield.npz`` in the configured leadfield
# directory. The stored file should contain a ``leadfield`` array and may also
# contain sensor/source metadata.

subject = "demo_subject"
load_path = leadfield_dir / f"{subject}_fixed_leadfield.npz"
np.savez(
    load_path,
    leadfield=simulated.leadfield,
    sensor_kind=simulated.sensor_kind,
    sensor_units=simulated.sensor_units,
    sensor_unitmult=simulated.sensor_unitmult,
    coil_type=simulated.coil_type,
    src_coords=simulated.src_coords,
    Q_basis=simulated.Q_basis,
)

loaded = builder.get_leadfield(
    subject=subject,
    orientation_type="fixed",
    retrieve_mode="load",
    return_metadata=True,
)

print("loaded shape:", loaded.leadfield.shape)
print("loaded equals saved:", np.allclose(loaded.leadfield, simulated.leadfield))

# %%
# Visualize leadfield magnitudes
# ------------------------------
#
# A simple quality-control plot is the per-source leadfield norm. Large
# deviations can indicate scaling, orientation, or file-loading problems.

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(np.linalg.norm(random_fixed.leadfield, axis=0), label="random fixed")
ax.plot(np.linalg.norm(simulated.leadfield, axis=0), label="simulated fixed")
ax.plot(np.linalg.norm(loaded.leadfield, axis=0), "--", label="loaded fixed")
ax.set(
    xlabel="Source index",
    ylabel="Leadfield column norm",
    title="Leadfield magnitude check",
)
ax.legend(loc="best")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "leadfield_norms.png", dpi=150)

# %%
# Summary
# -------
#
# The retrieval modes serve different purposes:
#
# - ``random``: fast synthetic matrices for testing shape logic;
# - ``simulate``: full MNE forward-model construction in production;
# - ``load``: standard workflow mode for precomputed leadfield NPZ files.
#
# For paper-scale workflows, use ``get_data_path`` to locate the local dataset
# root and point ``LeadfieldBuilder`` to the directory containing
# ``*_fixed_leadfield.npz`` or ``*_free_leadfield.npz`` files.

print(f"Saved leadfield tutorial figure in: {OUTPUT_DIR}")
