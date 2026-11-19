"""
05. Preparing the Example Data
==============================

The simulations, solvers, and workflows of CaliBrain need two kinds of files for
every subject: a *forward solution* (an MNE ``-fwd.fif`` file with the source space
and the gain matrix) and *leadfield files* (``.npz`` files that CaliBrain reads). This
tutorial follows the example data from the download to files that are ready to use:

1. ``fetch_example_data()`` downloads the forward solution;
2. ``reduce_forward_solutions()`` reduces it to the 1284 sources CaliBrain works with;
3. ``extract_leadfields()`` creates the leadfield files from it.

It also shows how to check that everything is ready, and how to run the same steps
on your own forward solutions. The first run downloads about 39 MB and takes about
half a minute; later runs only check that the files exist.
"""

# %%
# The data folder
# ---------------
#
# CaliBrain keeps its data in one folder with three subfolders. ``get_data_path()``
# returns it: ``<repository>/data`` in a source checkout and a per-user folder (for
# example ``~/.cache/calibrain``) for an installed package. To use another folder,
# pass it as ``data_dir`` to the three functions below. CaliBrain does not read
# environment variables.
#
# - ``fwd/``: the forward solutions with the full source space (step 1);
# - ``1284src_fwd/``: the forward solutions reduced to 1284 sources (step 2);
# - ``1284src_leadfield/``: the leadfield files (step 3).

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.io.constants import FIFF

from calibrain import (
    LeadfieldBuilder,
    extract_leadfields,
    fetch_example_data,
    reduce_forward_solutions,
)
from calibrain.utils import get_data_path, get_fwd_dir, get_leadfield_dir, restrict_fwd_to_sources

data_dir = get_data_path()
full_fwd_dir = data_dir / "fwd"
reduced_fwd_dir = get_fwd_dir()
leadfield_dir = get_leadfield_dir()

print("data folder:         ", data_dir)
print("full forward:        ", full_fwd_dir)
print("reduced forward:     ", reduced_fwd_dir)
print("leadfield files:     ", leadfield_dir)


def show(folder):
    """Print the files of ``folder`` with their sizes."""
    for path in sorted(folder.glob("*")):
        print(f"  {path.name:<42} {path.stat().st_size / 1e6:7.1f} MB")


# %%
# Step 1: download the forward solution
# -------------------------------------
#
# ``fetch_example_data()`` downloads the forward solution of the ``fsaverage``
# template from the `CaliBrain-data <https://github.com/braindatalab/CaliBrain-data>`_
# repository and unpacks it into ``fwd/``. The download is checked against a
# SHA-256 checksum that is stored in CaliBrain, so a given version of CaliBrain always
# gets the data it was tested with. Nothing is downloaded when the file already exists
# (``force=True`` downloads it again).

fetch_example_data()
show(full_fwd_dir)

# %%
# The file is an ordinary MNE forward solution. It has about 5100 sources (2562 per
# hemisphere) on the cortical surface, and 64 EEG channels.

fwd = mne.read_forward_solution(full_fwd_dir / "fsaverage-fwd.fif", verbose="error")

print("sources:         ", fwd["nsource"])
channel_types, channel_counts = np.unique(fwd["info"].get_channel_types(), return_counts=True)
print("channels:        ", fwd["nchan"], {str(t): int(c) for t, c in zip(channel_types, channel_counts)})
print("fixed orientation:", mne.forward.is_fixed_orient(fwd))
print("gain matrix shape:", fwd["sol"]["data"].shape, "(channels, sources x 3 components)")

# %%
# Step 2: reduce it to 1284 sources
# ---------------------------------
#
# The solvers of CaliBrain work with a source space of 1284 sources: 642 per
# hemisphere, chosen at random with a fixed seed. ``reduce_forward_solutions()``
# reads every forward solution in ``fwd/`` and writes the reduced one to
# ``1284src_fwd/``. A subject whose reduced forward solution already exists is skipped
# (``force=True`` overwrites it).
#
# The reduction is not part of the download, because it depends on the version of
# CaliBrain. With the default ``seed=42`` it gives the same sources on every computer.

reduce_forward_solutions("fsaverage")
show(reduced_fwd_dir)

# %%
# The figure shows the source positions before and after the reduction, seen from above.


def source_positions(forward):
    """Positions (in m) of the sources of a forward solution, both hemispheres."""
    return np.vstack([hemi["rr"][hemi["vertno"]] for hemi in forward["src"]])


reduced = mne.read_forward_solution(reduced_fwd_dir / "fsaverage-fwd.fif", verbose="error")

fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4), sharex=True, sharey=True)
for ax, forward, title in zip(axes, (fwd, reduced), ("full source space", "reduced (1284 sources)")):
    xyz = source_positions(forward)
    ax.scatter(xyz[:, 0] * 100, xyz[:, 1] * 100, s=3, color="0.3")
    ax.set(title=f"{title}: {forward['nsource']} sources", xlabel="x (cm)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3)
axes[0].set_ylabel("y (cm)")
fig.tight_layout()

# %%
# The seed decides which sources are kept. The same seed always gives the same
# sources, another seed gives other ones. ``restrict_fwd_to_sources`` is the function
# that ``reduce_forward_solutions`` uses; here it works in memory.


def kept_vertices(seed):
    forward = restrict_fwd_to_sources(fwd, n_keep=642, seed=seed)
    return [hemi["vertno"] for hemi in forward["src"]]


same_seed = all(np.array_equal(a, b) for a, b in zip(kept_vertices(42), kept_vertices(42)))
other_seed = all(np.array_equal(a, b) for a, b in zip(kept_vertices(42), kept_vertices(7)))
matches_file = all(np.array_equal(a, hemi["vertno"]) for a, hemi in zip(kept_vertices(42), reduced["src"]))

print("seed 42 twice gives the same sources:", same_seed)
print("seed 42 and seed 7 give the same sources:", other_seed)
print("seed 42 gives the sources of the reduced file:", matches_file)

# %%
# Step 3: create the leadfield files
# ----------------------------------
#
# ``extract_leadfields()`` reads the reduced forward solutions and writes two leadfield
# files per subject to ``1284src_leadfield/``: ``<subject>_fixed_leadfield.npz`` (one
# component per source, normal to the cortex) and ``<subject>_free_leadfield.npz``
# (three components per source). They are created on your computer and not
# downloaded, because their format depends on the version of CaliBrain.
#
# The channel type is chosen from the subject name: ``eeg`` for ``fsaverage`` and ``mag``
# (magnetometers) for the CamCAN subjects. For other subjects, pass ``coil_name``.

if not (leadfield_dir / "fsaverage_free_leadfield.npz").exists():
    extract_leadfields("fsaverage")
show(leadfield_dir)

# %%
# Check that everything is ready
# ------------------------------
#
# ``LeadfieldBuilder`` in ``load`` mode reads the leadfield files. With
# ``return_metadata=True`` it also returns the sensor type and units, the source
# coordinates, and for the free orientation the cortical normals.

builder = LeadfieldBuilder(leadfield_dir=leadfield_dir)
fixed = builder.get_leadfield(
    subject="fsaverage", orientation_type="fixed", retrieve_mode="load", return_metadata=True
)
free = builder.get_leadfield(
    subject="fsaverage", orientation_type="free", retrieve_mode="load", return_metadata=True
)

print("fixed leadfield:", fixed.leadfield.shape, "(sensors, sources)")
print("free leadfield: ", free.leadfield.shape, "(sensors, sources, components)")
print("EEG sensors:", fixed.sensor_kind == FIFF.FIFFV_EEG_CH, "| unit is volt:", fixed.sensor_units == FIFF.FIFF_UNIT_V)
print("source coordinates:", fixed.src_coords.shape)
print("cortical normals:", free.cortical_normals.shape)

# %%
# These checks tell you which step is missing for a subject:

for subject in ("fsaverage",):
    print(f"{subject}: downloaded={(full_fwd_dir / f'{subject}-fwd.fif').exists()}, "
          f"reduced={(reduced_fwd_dir / f'{subject}-fwd.fif').exists()}, "
          f"leadfield files={all((leadfield_dir / f'{subject}_{o}_leadfield.npz').exists() for o in ('fixed', 'free'))}")

# %%
# Using your own forward solutions
# --------------------------------
#
# Steps 2 and 3 do not care where a forward solution comes from. To use your own,
# save it with MNE as ``<data folder>/fwd/<subject>-fwd.fif`` and run the two steps with
# that folder as ``data_dir``. The subject name is the file name without ``-fwd.fif``.
#
# The example below writes the ``fsaverage`` forward solution as ``my_subject`` into a
# temporary folder, so that your data folder is not touched. For a subject that CaliBrain
# does not know, the channel type must be given, so ``extract_leadfields`` first fails
# with a message that says so.

with TemporaryDirectory() as tmp:
    own_dir = Path(tmp)
    (own_dir / "fwd").mkdir()
    mne.write_forward_solution(own_dir / "fwd" / "my_subject-fwd.fif", fwd, verbose="error")

    reduce_forward_solutions(data_dir=own_dir)

    try:
        extract_leadfields(data_dir=own_dir)
    except ValueError as error:
        print("\nwithout coil_name:", error)

    saved = extract_leadfields(data_dir=own_dir, coil_name="eeg")
    print("\nwritten files:", [path.relative_to(own_dir).as_posix() for path in saved])

    own = LeadfieldBuilder(leadfield_dir=own_dir / "1284src_leadfield").get_leadfield(
        subject="my_subject", orientation_type="fixed", retrieve_mode="load", return_metadata=True
    )
    print("leadfield of my_subject:", own.leadfield.shape)

# %%
# What a forward solution must have to work with CaliBrain:
#
# - free orientation (three components per source). A fixed-orientation forward solution
#   gives a fixed leadfield, but a free leadfield cannot be recovered from it, and
#   ``extract_leadfields`` stops with an error;
# - a source space on the cortical surface, because the free leadfield stores the
#   cortical normals;
# - channels of the type you pass as ``coil_name``: ``"eeg"``, ``"mag"`` (magnetometers),
#   or ``"grad"`` (gradiometers). Channels of other types are dropped.
#
# The CamCAN subjects
# -------------------
#
# The default workflow configurations use four CamCAN participants. Their forward
# solutions are a separate dataset (about 108 MB), downloaded in the same way:
#
# .. code-block:: python
#
#     fetch_example_data("camcan")
#     reduce_forward_solutions()   # every subject in <data folder>/fwd
#     extract_leadfields()         # every subject in <data folder>/1284src_fwd
#
# They are derived from CamCAN data and remain subject to the CamCAN terms; read the
# Dataset Notice before you use them.
#
# If something fails
# ------------------
#
# - ``ValueError`` from ``fetch_example_data`` about a checksum: the download is
#   incomplete or damaged. Download it again with ``fetch_example_data(force=True)``.
# - ``FileNotFoundError: No forward solutions ... found in .../1284src_fwd`` from
#   ``extract_leadfields``: step 2 is missing. Run ``reduce_forward_solutions()``.
# - ``FileNotFoundError: No forward solutions ... found in .../fwd`` from
#   ``reduce_forward_solutions``: step 1 is missing, or the files are not in the data
#   folder you passed as ``data_dir``.
# - Leadfield files from an earlier version of CaliBrain do not have all the fields that
#   the current version needs (for example the cortical normals). Create them again with
#   ``extract_leadfields()``, which overwrites the existing files.

# %%
# Summary
# -------
#
# .. list-table::
#    :header-rows: 1
#
#    * - Step
#      - Function
#      - Reads
#      - Writes
#    * - 1
#      - ``fetch_example_data()``
#      - the CaliBrain-data release
#      - ``fwd/``
#    * - 2
#      - ``reduce_forward_solutions()``
#      - ``fwd/``
#      - ``1284src_fwd/``
#    * - 3
#      - ``extract_leadfields()``
#      - ``1284src_fwd/``
#      - ``1284src_leadfield/``
#
# The next tutorial, :doc:`Leadfield Construction </auto_tutorials/06_leadfield_building>`,
# explains ``LeadfieldBuilder``, and :doc:`Workflow with a Real Head Model
# </auto_tutorials/15_real_head_model_workflow>` runs the whole calibration chain on the
# data prepared here.
