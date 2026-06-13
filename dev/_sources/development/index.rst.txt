Development Guide
=================

This guide summarizes the current package architecture and documentation
maintenance conventions.

Project architecture
--------------------

The repository has three main layers:

* ``calibrain/``: importable package code.
* ``calibrain/workflows/``: reproducible scripts for data generation,
  aggregation, calibration, and paper figure plotting.
* ``configs/``: Python configuration files defining workflow inputs, filters,
  and output paths.

Important implementation modules:

* ``leadfield_builder.py``: loading and constructing leadfield data.
* ``source_simulation.py``: source-level signal simulation.
* ``sensor_simulation.py``: forward projection and sensor-noise simulation.
* ``source_estimation.py``: inverse solvers and estimator wrappers.
* ``uncertainty_estimation.py``: credible intervals, ellipsoids, and coverage
  curves.
* ``uncertainty_calibration.py``: isotonic recalibration and train/eval
  calibration logic.
* ``metric_evaluation.py``: calibration and spatial evaluation metrics.
* ``calibration_dataset.py`` and ``run_manifest.py``: storage and manifest
  data model.

Coding standards
----------------

Contributions should preserve the following conventions:

* Keep workflow artifacts explicit and inspectable on disk.
* Prefer manifest-based discovery over filesystem scanning.
* Keep raw posterior summaries complete; reduce storage only in aggregation.
* Use NumPy-style docstrings for public functions and classes.
* Keep configuration-driven experiments reproducible by documenting all paths,
  filters, seeds, and split definitions.

Testing strategy
----------------

Recommended tests should cover:

* shape and metadata contracts for fixed and free-orientation source arrays.
* manifest parsing and metadata filtering.
* aggregation output schemas for fixed and free orientation.
* calibration behavior for ``fit_calibration=False``, fitted mappings, and
  ``fit_once=True``.
* regression checks for calibration JSON schema fields used by plotting.

Documentation maintenance
-------------------------

Build the documentation from the repository root:

.. code-block:: bash

   cd docs
   make html

The redesigned documentation removes historical generated gallery artifacts
that were built against older APIs. New examples should be small, runnable, and
should include expected outputs.

Continuous integration recommendations
--------------------------------------

A complete CI setup should include:

* import smoke tests for ``calibrain``.
* unit tests for storage and calibration utilities.
* a lightweight docs build using ``sphinx-build -b html docs/source docs/build/html``.
* optional workflow smoke tests on very small synthetic grids.

.. toctree::
   :maxdepth: 2
   :caption: Development topics

   contributing
   changelog
