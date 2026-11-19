.. meta::
   :description: Configure the local example data and forward-model files used by CaliBrain simulation and calibration workflows.

Example Dataset
===============

CaliBrain workflows require local forward-model and leadfield files before
larger simulation studies can be run. This page describes the example dataset
layout expected by the current codebase and how to point CaliBrain to that
data.

.. note::

   These example data are provided solely to help users become familiar with
   CaliBrain workflows and data handling. They are not intended for evaluating
   the performance of any EEG, MEG, MRI, forward-modeling, or acquisition
   system.

Dataset access in the current workflow is local and explicit. The workflow
configuration points to the directories that contain forward solutions and
leadfield files.

.. note::

   See the :doc:`Dataset Notice <data_notice>` for the distinction between the
   CaliBrain software license and third-party dataset terms.

Data root
---------

The local data root is expected to contain precomputed forward solutions and
leadfield matrices. In practice, the workflow configuration should point to the
directories that contain those files.

The example data are not part of the PyPI package or the git repository. They are
published as release assets of the
`CaliBrain-data <https://github.com/braindatalab/CaliBrain-data>`_ repository. Getting
them ready takes two independent steps:

.. code-block:: python

   from calibrain.datasets import fetch_example_data
   from calibrain.extract_leadfield import extract_leadfields

   fetch_example_data()   # step 1: download the forward solution
   extract_leadfields()   # step 2: create the leadfield files from it

**Step 1** downloads ``calibrain-example-data-fsaverage.zip`` (about 36 MB), verifies its
SHA-256 checksum, and unpacks the forward solution into
``<data directory>/1284src_fwd``. Nothing is downloaded if the file already exists;
pass ``force=True`` to download it again. The data are computed from the ``fsaverage``
template and contain no participant data. The CamCAN subjects are not included; see
the :doc:`Dataset Notice <data_notice>`.

**Step 2** creates the leadfield files in ``<data directory>/1284src_leadfield``. They
are not downloaded, because their format depends on the version of CaliBrain: creating
them locally means they always match the installed version. On the command line,
``python -m calibrain.extract_leadfield`` does the same. By default it processes every
subject that has a forward solution; ``--subjects`` and ``--data-dir`` select others.

CaliBrain resolves the data directory with ``calibrain.utils.get_data_path``:

- If the ``CALIBRAIN_DATA`` environment variable is set, that directory is used.
- Otherwise, in a source checkout, ``<repository>/data`` is used.
- Otherwise (a PyPI installation), a per-user directory is used, for example
  ``~/.cache/calibrain`` on Linux.

.. code-block:: bash

   # Linux and macOS
   export CALIBRAIN_DATA=/path/to/calibrain_data

.. code-block:: powershell

   # Windows (PowerShell)
   $env:CALIBRAIN_DATA = "C:\path\to\calibrain_data"


Available local example data
----------------------------

The current workflow configurations assume an example source space with 1284
sources and multiple subjects.

Forward solutions versus leadfields
-----------------------------------

The dataset layout distinguishes between two related but different objects:

- **Forward solution** (``fwd``): an MNE forward-model file, typically stored
  as ``*-fwd.fif``. It contains the full source-to-sensor model together with
  source-space and geometry information. It is used when source coordinates or
  other source-space metadata must be recovered.

- **Leadfield** (``leadfield``): a matrix or tensor extracted from the forward
  solution. It contains the numerical source-to-sensor mapping used directly by
  the simulation and inverse solvers. It is stored in compact ``.npz`` files
  for workflow use.

Full versus reduced representations
-----------------------------------

The local data may also distinguish between **full** and **reduced**
representations:

- **Full forward solution**: retains the original forward-model object and
  associated metadata. It is appropriate when source-space geometry,
  orientation structure, or coordinate recovery is needed.

- **Reduced leadfield**: stores only the numerical leadfield needed by the
  workflow. It is smaller and faster to load than the full forward solution and
  is intended for routine simulation and source-estimation runs.

Modality-specific overview
--------------------------

The workflow uses the same directory structure for EEG and MEG, but the
geometric interpretation differs by modality and orientation setting.

EEG
~~~

Typical EEG-related files in the local example dataset include:

.. list-table:: EEG-oriented example layout
   :header-rows: 1

   * - Path
     - Contents
   * - ``1284src_fwd/``
     - Forward solutions such as ``CC120166-fwd.fif`` and
       ``fsaverage-fwd.fif`` containing source-space geometry.
   * - ``1284src_leadfield/``
     - Reduced fixed and free-orientation leadfields such as
       ``CC120166_fixed_leadfield.npz`` and
       ``CC120166_free_leadfield.npz``.

In free-orientation EEG, each source location is represented with three local
components. This is the setting in which CaliBrain can work with both
``marginal`` and ``full_cov`` uncertainty representations.

MEG
~~~

Typical MEG-related files use the same local folders, but the reduced
leadfields are interpreted differently:

.. list-table:: MEG-oriented example layout
   :header-rows: 1

   * - Path
     - Contents
   * - ``1284src_fwd/``
     - Forward solutions used when source coordinates or other source-space
       information must be recovered.
   * - ``1284src_leadfield/``
     - Reduced fixed and reduced free-orientation leadfields used in
       simulation and inverse estimation.

In reduced free-orientation MEG, each source location is represented with two
tangential components rather than the full three-dimensional orientation used
for EEG. This changes both the leadfield shape and the geometry of the local
uncertainty representation.

Subject identifiers
-------------------

The default local example dataset commonly uses:

.. code-block:: text

   CC120166
   CC120264
   CC120309
   CC120313
   fsaverage

The exact subject list is controlled by the workflow configs. If a config
requests a subject whose forward or leadfield file is missing, data generation
or calibration will fail with a file-not-found error.

Minimal check
-------------

Use this short check before running data generation:

.. code-block:: python

   from pathlib import Path

   data_root = Path("/path/to/calibrain/data")
   print(data_root)
   print(sorted((data_root / "1284src_leadfield").glob("*_fixed_leadfield.npz")))

For a configured example dataset, the second line should print at least one
fixed-orientation leadfield file.

.. toctree::
   :hidden:

   data_notice
