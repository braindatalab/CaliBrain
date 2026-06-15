CaliBrain
=========

.. image:: https://img.shields.io/badge/PyPI%20downloads-not%20published-lightgrey
   :target: https://pypi.org/project/calibrain/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/calibrain.svg
   :target: https://pypi.org/project/calibrain/
   :alt: Supported Python versions

.. image:: https://readthedocs.org/projects/calibrain/badge/?version=latest
   :target: https://calibrain.readthedocs.io/en/latest/
   :alt: Documentation status

.. image:: https://github.com/braindatalab/CaliBrain/actions/workflows/docs-check.yml/badge.svg
   :target: https://github.com/braindatalab/CaliBrain/actions/workflows/docs-check.yml
   :alt: Documentation build

.. image:: https://img.shields.io/pypi/l/calibrain.svg
   :target: https://github.com/braindatalab/CaliBrain/blob/main/LICENSE
   :alt: License

.. image:: https://img.shields.io/github/v/release/braindatalab/CaliBrain
   :target: https://github.com/braindatalab/CaliBrain/releases
   :alt: Latest GitHub release

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.20703249.svg
   :target: https://doi.org/10.5281/zenodo.20703249
   :alt: DOI

CaliBrain is a Python package for simulation-based evaluation of uncertainty
estimation and calibration in EEG/MEG inverse source imaging. It combines
source and sensor simulation, inverse modeling, posterior uncertainty
extraction, and isotonic recalibration into a reproducible workflow for
comparing methods under controlled conditions.

Current workflow
----------------

The current analysis pipeline is organized around explicit workflow scripts and
Python config files:

1. ``calibrain/workflows/data_generation.py``
   generates simulated runs, solves inverse problems, writes per-run HDF5
   posterior summaries, and updates a manifest CSV.
2. ``calibrain/workflows/aggregation.py``
   reads the manifest, filters runs by metadata, and writes calibration-ready
   NPZ datasets plus JSON sidecars.
3. ``calibrain/workflows/calibration.py``
   fits or evaluates isotonic recalibration maps and writes calibration JSON
   summaries.
4. ``calibrain/workflows/plot_paper_calibration_figures.py``
   reads calibration JSON outputs and generates paper-style figures.

The workflow configs live in ``configs/``. Inspect and edit those files before
running large experiments, because they define solver grids, split definitions,
data paths, and output locations.

Core package components
-----------------------

* ``LeadfieldBuilder``: load or construct leadfield data.
* ``SourceSimulator`` and ``SensorSimulator``: simulate source activity and
  sensor measurements.
* ``gamma_map_sflex``, ``gamma_lambda_map_sflex``, ``BMN``, and
  ``BMN_joint``: inverse solvers.
* ``UncertaintyEstimator``: construct coverage curves for intervals,
  ellipses, and ellipsoids.
* ``UncertaintyCalibrator``: fit isotonic nominal-coverage recalibration maps.
* ``MetricEvaluator``: compute calibration and source-space metrics.
* ``DataGenerator``: orchestrate simulation and posterior-summary generation.

Installation
------------

From a local checkout:

.. code-block:: bash

   git clone https://github.com/braindatalab/CaliBrain.git
   cd CaliBrain
   python -m pip install -e .

For documentation work:

.. code-block:: bash

   python -m pip install -e ".[docs]"

The package metadata defines the distribution name as ``calibrain``. It is not
currently published on PyPI or conda-forge, so installation is presently from a
source checkout or a local environment specification such as ``environment.yml``.

Build documentation
-------------------

.. code-block:: bash

   cd docs
   make html

The rendered site is written to ``docs/build/html/index.html``.

Documentation
-------------

The redesigned documentation is under ``docs/source`` and is based on the
current source tree and workflow scripts. Executable tutorials in
``tutorials/`` are rendered with Sphinx-Gallery during ``make html``.

The canonical published documentation is hosted on Read the Docs:
https://calibrain.readthedocs.io/

Citation
--------

If you use CaliBrain in academic work, please cite the software archive:

``Orabe, M., Huseynov, I., Nagarajan, S., & Haufe, S. (2026). CaliBrain (v1.0.1). Zenodo. https://doi.org/10.5281/zenodo.20703249``

License
-------

CaliBrain is distributed under the BSD 3-Clause License. See ``LICENSE``.
