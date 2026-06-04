Workflows
=========


CaliBrain's current end-to-end analysis is organized into workflow scripts. The
scripts are configured by editing Python config files under ``configs/``; they do
not currently parse command-line ``--config`` flags.


Data generation
---------------


Entry point:

.. code-block:: bash

   python calibrain/workflows/data_generation.py


Default config:

.. code-block:: text

   configs/data_generation_default.py


Inputs:

- precomputed leadfields under the configured ``leadfield_dir``
- solver grids in ``CONFIG["estimators"]``
- ERP, source sparsity, orientation, subject, SNR, and noise grids
- random seed and number of runs

Outputs:

- per-run posterior summaries under ``posterior_dir``
- manifest CSV at ``manifest_path``
- log files under ``log_dir``

The manifest is the required input for aggregation.

Aggregation
-----------


Entry point:

.. code-block:: bash

   python calibrain/workflows/aggregation.py


Default config:

.. code-block:: text

   configs/aggregate_default.py


Inputs:

- manifest CSV from data generation
- metadata filters defining the requested split

Outputs:

- per-run aggregated NPZ datasets
- JSON sidecars describing the aggregation criteria and summary sources

Supported experiment selectors in the default config:

- ``precal``
- ``post_oracle``
- ``post_pooled``
- ``post_pooled_mismatch``
- ``post_fixed``

Calibration
-----------


Entry point:

.. code-block:: bash

   python calibrain/workflows/calibration.py


Default config:

.. code-block:: text

   configs/calibration_default.py


Inputs:

- aggregated training datasets
- aggregated evaluation datasets
- nominal coverage grid
- calibration strategy settings such as ``fit_calibration``, ``fit_once``, and
  ``free_interval_type``

Outputs:

- calibration JSON summaries
- optional calibration plots

Paper figures
-------------


Entry point:

.. code-block:: bash

   python calibrain/workflows/plot_paper_calibration_figures.py


Inputs:

- calibration JSON summaries
- aggregated NPZ files referenced by each JSON ``eval_source``

Outputs:

- publication-style calibration figures under
  ``results/figures/paper_calibration``

This script is specialized for the paper-style fixed/free comparison figures
and should be treated as analysis tooling rather than a general plotting API.
