"""
06. Leadfield Construction
==========================

This tutorial mainly explains the ``LeadfieldBuilder`` class. It demonstrates
the three leadfield retrieval modes used by ``LeadfieldBuilder``:

- ``retrieve_mode="random"``: create a synthetic random leadfield;
- ``retrieve_mode="simulate"``: run a simulation pipeline;
- ``retrieve_mode="load"``: load a leadfield through the standard loading API.

The full MNE simulation pipeline requires subject surfaces, BEM files, sensor
information, and forward-solution configuration. To keep this tutorial
executable during documentation builds, the simulate example uses a small
subclass that returns a deterministic synthetic leadfield while preserving the
same ``get_leadfield(..., retrieve_mode="simulate")`` interface.

The last section loads real example data. It downloads the forward solution of
the ``fsaverage`` template (about 39 MB, once), reduces it to 1284 sources, and
creates the leadfield files from it.
"""



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
# - source coordinates associated with the leadfield are represented in ``m``.
#
# This means:
#
# - ``(fT / nAm) × nAm = fT`` at MEG sensors;
# - ``(µV / nAm) × nAm = µV`` at EEG sensors.

import matplotlib.pyplot as plt
import numpy as np
from mne.io.constants import FIFF
from tempfile import TemporaryDirectory

from calibrain import LeadfieldBuilder, extract_leadfields, fetch_example_data, reduce_forward_solutions
from calibrain.utils import get_data_path, get_leadfield_dir


RANDOM_SEED = 23

# %%
# Use the CaliBrain data-root helper
# ----------------------------------
#
# CaliBrain resolves local datasets with ``get_data_path``. With no argument,
# it uses ``<repository>/data`` in a source checkout, and a per-user directory
# (for example ``~/.cache/calibrain``) for an installed package. To use another
# folder, pass it as ``data_dir`` to ``fetch_example_data`` and ``extract_leadfields``.

data_root = get_data_path()
leadfield_dir = get_leadfield_dir()

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
# extraction. That path requires external anatomy and MNE configuration, so this
# lightweight tutorial shows the expected output structure with a synthetic
# stand-in that matches the usual fixed-orientation shape and metadata.

rng = np.random.default_rng(RANDOM_SEED)
n_sensors = 12
n_sources = 24
source_positions = rng.normal(scale=0.04, size=(n_sources, 3))
sensor_positions = rng.normal(scale=0.08, size=(n_sensors, 3))
distances = np.linalg.norm(
    sensor_positions[:, None, :] - source_positions[None, :, :],
    axis=2,
)
simulated_leadfield = 1.0 / np.maximum(distances, 0.02) ** 2
simulated_leadfield *= rng.choice([-1.0, 1.0], size=simulated_leadfield.shape)
simulated_leadfield /= np.linalg.norm(simulated_leadfield, axis=0, keepdims=True)
simulated_q_basis = np.eye(n_sources)

print("simulate-like shape:", simulated_leadfield.shape)
print("simulate-like sensor unit:", FIFF.FIFF_UNIT_T)
print("simulate-like sensor unit multiplier:", FIFF.FIFF_UNITM_F)
print("simulate-like coil type:", FIFF.FIFFV_COIL_VV_MAG_T1)

# %%
# Exercise the load interface
# ---------------------------
#
# In the full workflow, ``retrieve_mode="load"`` reads a precomputed leadfield
# dataset from ``leadfield_dir``. Here we create a temporary fixture and then
# load it through the standard ``LeadfieldBuilder`` API.

tmpdir = TemporaryDirectory()
temp_builder = LeadfieldBuilder(leadfield_dir=tmpdir.name)
np.savez(
    f"{tmpdir.name}/demo_subject_fixed_leadfield.npz",
    leadfield=simulated_leadfield,
    sensor_kind=FIFF.FIFFV_MEG_CH,
    sensor_units=FIFF.FIFF_UNIT_T,
    sensor_unitmult=FIFF.FIFF_UNITM_F,
    coil_type=FIFF.FIFFV_COIL_VV_MAG_T1,
    src_coords=source_positions,
    Q_basis=simulated_q_basis,
)
loaded = temp_builder.get_leadfield(
    subject="demo_subject",
    orientation_type="fixed",
    retrieve_mode="load",
    return_metadata=True,
)

print("loaded shape:", loaded.leadfield.shape)
print("loaded equals simulate-like:", np.allclose(loaded.leadfield, simulated_leadfield))
tmpdir.cleanup()

# %%
# Load the example data
# ---------------------
#
# The example data are not part of the package. They are downloaded on demand
# from the `CaliBrain-data <https://github.com/braindatalab/CaliBrain-data>`_
# repository. Getting them ready takes three independent steps, which the previous
# tutorial, :doc:`Preparing the Example Data </auto_tutorials/05_preparing_the_example_data>`,
# explains one by one:
#
# 1. ``fetch_example_data()`` downloads the forward solution of the ``fsaverage``
#    template with the full source space (about 39 MB) into the data root and
#    checks its SHA-256 checksum. Nothing is downloaded if the file is already there.
# 2. ``reduce_forward_solutions()`` reduces it to the 1284 sources (642 per
#    hemisphere) that CaliBrain works with. The sources are chosen with a fixed
#    seed, so everyone gets the same ones.
# 3. ``extract_leadfields()`` creates the leadfield files from the reduced forward
#    solution. They are created locally and not downloaded, because their format
#    depends on the version of CaliBrain.

fetch_example_data()  # step 1: download the forward solution
reduce_forward_solutions("fsaverage")  # step 2: reduce it to 1284 sources
if not (leadfield_dir / "fsaverage_free_leadfield.npz").exists():
    extract_leadfields("fsaverage")  # step 3: create the leadfield files

# %%
# Now the standard ``load`` mode reads the real leadfield. For free orientation
# the metadata also holds the cortical normals and the local source basis that the
# free-orientation simulation needs. The
# :doc:`real head model workflow </auto_tutorials/15_real_head_model_workflow>`
# uses this leadfield for a complete simulation and calibration run.

example = builder.get_leadfield(
    subject="fsaverage",
    orientation_type="free",
    retrieve_mode="load",
    return_metadata=True,
)

print("example leadfield shape:", example.leadfield.shape)
print("cortical normals shape:", example.cortical_normals.shape)
print("Q_basis shape:", example.Q_basis.shape)
print("source coordinates shape:", example.src_coords.shape)

# %%
# Summary
# -------
#
# The retrieval modes serve different purposes:
#
# - ``random``: fast synthetic matrices for testing shape logic;
# - ``simulate``: full MNE forward-model construction in production;
# - ``load``: standard workflow mode for leadfield files, for example the ones
#   created from the example data in the previous section.
#
# For larger workflow runs, use ``fetch_example_data()``,
# ``reduce_forward_solutions()`` and ``extract_leadfields()`` to prepare the
# example data, or point
# ``LeadfieldBuilder`` to a directory that holds your own
# ``*_fixed_leadfield.npz`` or ``*_free_leadfield.npz`` files. The CamCAN example
# subjects are a separate dataset, ``fetch_example_data("camcan")``, and are
# subject to the CamCAN terms (see the Dataset Notice).
