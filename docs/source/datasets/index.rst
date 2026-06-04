Example Dataset
===============

CaliBrain workflows require local forward-model and leadfield files before
paper-scale simulations can be run. This page describes the example dataset
layout expected by the current codebase and how to point CaliBrain to that
data.

Unlike MNE-Python's public dataset registry, CaliBrain does not currently ship a
download helper that fetches data automatically. Dataset access is local and
explicit: users configure a data root with ``CALIBRAIN_DATA`` or pass paths in
the workflow configuration files.

Data root
---------

CaliBrain resolves the default data root with ``calibrain.utils.get_data_path``.
The lookup order is:

1. the ``CALIBRAIN_DATA`` environment variable, if set;
2. the repository-level ``data`` directory, otherwise.

For a local installation, set:

.. code-block:: bash

   export CALIBRAIN_DATA=/path/to/calibrain/data

The data root is expected to contain precomputed forward solutions and
leadfield matrices. These files are intentionally kept outside version control
because they are large and site-specific.

Available local example data
----------------------------

The current workflow configurations assume an example source space with 1284
sources and multiple subjects. A typical local data directory contains:

.. list-table:: Expected example dataset layout
   :header-rows: 1

   * - Path
     - Contents
     - Used by
   * - ``1284src_fwd/``
     - MNE forward solutions such as ``CC120166-fwd.fif`` and
       ``fsaverage-fwd.fif``.
     - Calibration EMD and source-coordinate lookup.
   * - ``1284src_leadfield/``
     - Reduced leadfield NPZ files such as
       ``CC120166_fixed_leadfield.npz`` and
       ``CC120166_free_leadfield.npz``.
     - Data-generation workflow and inverse solvers.
   * - ``fwd/``
     - Full or intermediate forward-solution files.
     - Leadfield extraction and reduction utilities.
   * - ``leadfield/``
     - Alternative fixed/free leadfield NPZ layout.
     - Legacy or exploratory scripts.

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

   from calibrain.utils import get_data_path

   data_root = get_data_path()
   print(data_root)
   print(sorted((data_root / "1284src_leadfield").glob("*_fixed_leadfield.npz")))

For a configured example dataset, the second line should print at least one
fixed-orientation leadfield file.

Workflow usage
--------------

The data-generation workflow reads leadfields from the configured
``leadfield_dir``:

.. code-block:: python

   CONFIG = {
       "leadfield_dir": "/path/to/calibrain/data/1284src_leadfield",
       "manifest_path": "/path/to/results/run_manifest/fixed.csv",
   }

The calibration workflow uses forward solutions to recover source coordinates
when source-space EMD is requested:

.. code-block:: text

   CALIBRAIN_DATA/1284src_fwd/<subject>-fwd.fif

Storage policy
--------------

Large forward solutions, leadfields, generated posterior summaries, aggregated
NPZ files, and calibration results should remain outside git. The repository
``.gitignore`` excludes local ``data/`` and ``results/`` directories by default.

Relationship to generated artifacts
-----------------------------------

The example dataset is an input to the workflow. It is distinct from generated
outputs:

.. list-table::
   :header-rows: 1

   * - Artifact
     - Created by
     - Purpose
   * - Forward solutions and leadfields
     - Prepared before running CaliBrain workflows.
     - Inputs for simulation, inverse estimation, and spatial metrics.
   * - Posterior H5 summaries
     - ``calibrain/workflows/data_generation.py``
     - Raw per-run solver output.
   * - Manifest CSV
     - ``calibrain/workflows/data_generation.py``
     - Auditable index of generated posterior summaries.
   * - Aggregated NPZ datasets
     - ``calibrain/workflows/aggregation.py``
     - Calibration-ready reduced uncertainty representation.
   * - Calibration JSON records
     - ``calibrain/workflows/calibration.py``
     - Pre/post calibration curves, metrics, and metadata.
