.. meta::
   :description: How the CaliBrain repository is organized, how a run flows through the package, and how the tests, continuous integration, and documentation work.

Development Guide
=================

This section is for people who change the code or the documentation of CaliBrain.

- :doc:`contributing` describes how to make a change: setup, branches, commit
  messages, checks, pull requests, and the maintainer checklists for releases and
  example data.
- :doc:`changelog` lists the changes of every release.
- This page explains how the repository is organized, how a run passes through the
  package, and how the tests, the continuous integration, and the documentation work.

Repository layout
-----------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Path
     - Contents
   * - ``calibrain/``
     - The installable package (see the next section).
   * - ``calibrain/workflows/``
     - The workflow entry points: ``run_data_generation``, ``aggregate_posteriors``,
       ``run_calibration``, and ``plot_paper_calibration_figures``.
   * - ``configs/``
     - Python configuration files for the workflows (``data_generation_default.py``,
       ``aggregate_default.py``, ``calibration_default.py``) and ``configs/README.md``.
   * - ``tutorials/``
     - The tutorials ``00_*.py`` to ``14_*.py``. Sphinx-Gallery runs them when the
       documentation is built.
   * - ``tests/``
     - The pytest suite.
   * - ``docs/``
     - The Sphinx documentation (``docs/source``).
   * - ``scripts/``
     - Maintainer scripts: ``build_example_data_archives.py``,
       ``check_version_switcher.py``, ``generate_readme_example.py``, and
       ``make_container_constraints.py``.
   * - ``container/``
     - The container image: ``Dockerfile``, the Apptainer definition ``calibrain.def``,
       the pinned package versions, and the install and test scripts. See
       :doc:`../installation/container_installation`.
   * - ``viz/``
     - Scripts that plot results of workflow runs (``plot_*.py``).
   * - ``data/``
     - The example data. It is not in git and is created by the data steps described in
       :doc:`../datasets/index`.
   * - ``results/``
     - The default output folder of the workflows. Only the paper figures in
       ``results/figures`` are tracked by git.

The package modules
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Module
     - Main objects
     - Purpose
   * - ``datasets.py``
     - ``fetch_example_data``
     - Download the example forward solutions and check their checksum.
   * - ``reduce_fwd.py``
     - ``reduce_forward_solutions``
     - Reduce forward solutions to the 1284-source space.
   * - ``extract_leadfield.py``
     - ``extract_leadfields``
     - Write the leadfield files (``.npz``) from forward solutions.
   * - ``leadfield_builder.py``
     - ``LeadfieldBuilder``, ``LeadfieldData``
     - Load, simulate, or randomly generate a leadfield, with its metadata (units,
       source coordinates, cortical normals).
   * - ``utils.py``
     - ``get_data_path``, ``get_fwd_dir``, ``get_leadfield_dir``, ``load_python_config``
     - The data folders, configuration loading, and small helpers.
   * - ``source_simulation.py``
     - ``SourceSimulator``
     - Simulate source activity (ERP waveforms) for fixed and free orientation.
   * - ``sensor_simulation.py``
     - ``SensorSimulator``
     - Project source activity to the sensors and add noise at a given SNR.
   * - ``source_estimation.py``
     - ``SourceEstimator``, ``gamma_map_sflex``, ``gamma_lambda_map_sflex``, ``BMN``,
       ``BMN_joint``
     - The inverse solvers and their wrapper.
   * - ``uncertainty_estimation.py``
     - ``UncertaintyEstimator``
     - Credible intervals and ellipsoids and empirical coverage curves.
   * - ``uncertainty_calibration.py``
     - ``UncertaintyCalibrator``
     - Fit and apply the isotonic recalibration.
   * - ``metric_evaluation.py``
     - ``MetricEvaluator``
     - Estimation, uncertainty, and calibration metrics.
   * - ``data_generation.py``
     - ``DataGenerator``
     - Run solver, data, and noise parameter grids and save one posterior summary per run.
   * - ``calibration_dataset.py``
     - ``PosteriorSummary``
     - Read posterior-summary files and filter them by metadata.
   * - ``run_manifest.py``
     - ``ManifestRow``, ``summaries_from_manifest``
     - Read the manifest CSV that lists the posterior summaries of a run.
   * - ``calibration_storage.py``
     - ``save_calibration_record``, ``load_calibration_record``
     - Save and load the JSON files that the calibration workflow writes.
   * - ``visualization.py``
     - ``Visualizer``
     - Plots of sources, sensors, and uncertainty.

How a run flows through the package
-----------------------------------

The three workflows run one after the other. Each is started with a configuration
(a Python file from ``configs/`` or a dictionary) and each stage reads the files that
the previous one wrote.

1. ``run_data_generation(config)`` loops over the solver, data, and noise grids and
   over ``nruns`` repetitions. For every run it simulates sources and sensor data from a
   leadfield, estimates the sources, writes a posterior-summary HDF5 file to
   ``posterior_dir`` (default ``results/posterior_summaries``), and records it in the
   manifest CSV at ``manifest_path`` (default ``results/run_manifest.csv``).
2. ``aggregate_posteriors(config)`` reads the manifest (``manifest_path`` is required),
   selects posterior summaries by their metadata, and writes one NPZ dataset per run
   for the calibration step.
3. ``run_calibration(config)`` loads those datasets, fits and evaluates the isotonic
   calibration according to the configured mode, and writes JSON records and
   optional plots to ``output_dir`` (default ``results/calibration_eval``).

``plot_paper_calibration_figures`` draws the figures of the calibration results.
``configs/README.md`` describes the configuration keys.

Conventions
-----------

- **Units.** Source amplitudes are in nAm and source coordinates in m. MEG leadfields
  are in fT/nAm and EEG leadfields in µV/nAm.
- **Array shapes.** A fixed-orientation leadfield has the shape
  ``(n_sensors, n_sources)``, a free-orientation one ``(n_sensors, n_sources,
  n_components)``. Free-orientation source activity has the shape
  ``(n_sources, n_components, n_times)``.
- **Random numbers.** Simulation and estimation functions take a ``seed``. Tutorials
  and tests fix their seeds.
- **Data folders.** Never write a path like ``data/1284src_fwd`` into code. Use
  ``get_data_path()``, ``get_fwd_dir()``, and ``get_leadfield_dir()``, and let the
  caller pass ``data_dir``. CaliBrain does not read environment variables.
- **Workflow discovery.** The aggregation step finds posterior summaries through the
  manifest CSV and does not scan the file system.
- **Docstrings.** Public functions and classes have NumPy-style docstrings
  (``sphinx.ext.napoleon`` renders them).
- **Logging.** Library code reports through the standard ``logging`` module. Classes
  take an optional ``logger`` argument.

Tests
-----

The tests are in ``tests/`` and run with ``python -m pytest``:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - File
     - What it covers
   * - ``test_datasets.py``
     - ``fetch_example_data`` against a local test server: checksum, unsafe archives,
       missing files, default folders.
   * - ``test_reduce_fwd.py``, ``test_extract_leadfield.py``
     - The data preparation steps, including that reducing ``fsaverage`` reproduces the
       example forward solution.
   * - ``test_example_data_archives.py``
     - The archive builder: no personal paths, reproducible zip files.
   * - ``test_workflow_data_dir.py``
     - The ``data_dir`` key of the data-generation and calibration workflows.
   * - ``test_leadfield_builder.py``
     - Loading leadfields, units, coordinates, and normals.
   * - ``test_simulation_and_generation.py``
     - ``SourceSimulator``, ``SensorSimulator``, and ``DataGenerator`` (including the
       posterior-summary files).
   * - ``test_uncertainty_estimation.py``, ``test_uncertainty_calibration.py``,
       ``test_metric_evaluation.py``, ``test_free_orientation_uncertainty.py``
     - Intervals, coverage curves, calibration modes, and metrics, for fixed and free
       orientation.
   * - ``test_container_files.py``
     - The Dockerfile, the definition file, and the pinned versions agree.
   * - ``test_package_metadata.py``
     - The version, the citation file, and ``switcher.json``.

Rules for tests:

- Tests that need the example data (``data/fwd``, ``data/1284src_fwd``, or the leadfield
  files) are skipped when it is missing. The continuous integration runs without it.
  To run them locally, prepare the data as described in :doc:`../datasets/index`.
- A test must never write into ``data/``. Use ``tmp_path``, and stub reading and
  writing of large files when the test is about the logic around them.
- Tests that download the real example data are marked ``network`` and only run with
  ``python -m pytest --network``.
- ``ruff`` is configured to find only syntax errors and undefined names (rules
  ``E9``, ``F63``, ``F7``, ``F82`` in ``pyproject.toml``). It does not check style.

Continuous integration
----------------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Workflow
     - What it does
   * - ``tests.yml``
     - On pull requests, on pushes to ``main`` and ``dev``, and manually: ``ruff check``;
       the tests on Python 3.10, 3.11, 3.12, 3.13, and 3.14; and the tests with the lowest
       dependency versions allowed by ``pyproject.toml`` (Python 3.10).
   * - ``docs-check.yml``
     - Builds the documentation with ``make html`` (``DOCS_VERSION=dev``, Python 3.10).
       It does not fail on warnings; Read the Docs does.
   * - ``container-check.yml``
     - On pull requests that change ``container/``: builds the image and the Apptainer
       definition from a wheel of the checkout and runs ``container/smoke_test.sh`` on them.
   * - ``publish-container.yml``
     - After ``publish-pypi.yml`` succeeds for a release (or manually): builds the image
       from the PyPI release, tests it with Docker and Apptainer, and pushes it to
       ``ghcr.io/braindatalab/calibrain``.
   * - ``publish-pypi.yml``
     - When a GitHub release is published (or manually): runs the tests, builds the
       distributions, checks them with ``twine check``, and uploads them to PyPI with
       trusted publishing.
   * - ``dependabot.yml``
     - Monthly update pull requests to ``dev`` for the versions of the GitHub actions.
       The Python dependencies are not updated automatically.

Read the Docs (``readthedocs.yml``) builds the documentation of ``dev``, ``main``
(``latest``), the ``stable`` branch, and the release tags, and fails on warnings.

Documentation
-------------

The sources are in ``docs/source``: ``index.rst``, the folders ``installation``,
``documentation`` (concepts and glossary), ``datasets``, ``tutorials``,
``api_reference``, ``cite``, and ``development``, and ``_static`` and ``_templates``.

Build it from the repository root, with warnings treated as errors as on Read the Docs:

.. code-block:: bash

   cd docs
   make html SPHINXOPTS="-W --keep-going"

The result is in ``docs/build/html``. Notes:

- **Tutorials are executed.** Sphinx-Gallery runs every file in ``tutorials/`` and the
  build fails if one raises an error. Tutorials 05, 06, and 15 download the example data
  (about 39 MB) into the data folder on the first run.
- **Adding a tutorial.** Name it ``NN_topic.py`` and start the file with a docstring
  that has a title with an underline. Then add its card and its entry in the toctree in
  ``docs/source/tutorials/index.rst`` by hand. Renaming or renumbering a tutorial also
  means updating the links to its page.
- **Adding a public function or class.** List it in ``docs/source/api_reference/index.rst``,
  in both the ``autosummary`` list and the toctree.
- **Generated files.** ``docs/source/auto_tutorials/`` and
  ``docs/source/api_reference/generated/`` are build output and are not in git. If a
  build reports pages that "aren't included in any toctree" after tutorials were renamed,
  delete the stale files in those folders and build again.
- **The version menu.** ``docs/source/_static/switcher.json`` lists the versions; see
  :ref:`the maintainer notes <doc-version-switcher>`.

.. toctree::
   :hidden:

   contributing
   changelog
