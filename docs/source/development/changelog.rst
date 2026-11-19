.. meta::
   :description: Release history and notable changes to the CaliBrain package, workflows, tests, and documentation.

Changelog
=========

This document records notable changes to the CaliBrain package.

.. note::
    This page summarizes notable package, workflow, and documentation changes.

Version 1.0.4 (unreleased)
--------------------------

*   [BREAKING] Free-orientation source simulation now generates one waveform per active source along its cortical normal (rank-one source blocks) instead of an independent waveform per orientation component. Results from earlier free-orientation runs are not comparable.
*   [BREAKING] ``SourceSimulator.simulate`` now requires the keyword-only argument ``cortical_normals`` for free-orientation simulations, plus ``Q_basis`` for reduced free MEG. Fixed-orientation simulation is unchanged.
*   [BREAKING] Existing free-orientation leadfield files must be regenerated with ``python -m calibrain.extract_leadfield``, because older files do not contain ``cortical_normals``.
*   [FEATURE] Added ``calibrain.datasets.fetch_example_data()``, which downloads example forward solutions from the CaliBrain-data repository on demand: the ``fsaverage`` template, and, as a separate dataset subject to the CamCAN terms, four CamCAN participants (``fetch_example_data("camcan")``). The download is checked against a SHA-256 checksum. The leadfield files are not downloaded: they depend on the CaliBrain version and are created from the forward solution in a separate step.
*   [FEATURE] Added ``extract_leadfields()`` (also ``python -m calibrain.extract_leadfield``) to create the leadfield files from the forward solutions in the data directory. By default it processes every subject that has a forward solution, and it accepts ``--subjects``, ``--data-dir`` and ``--coil``. Before, the command line failed unless all five example forward solutions were present.
*   [ENHANCEMENT] The default data directory of an installed package is now a per-user directory instead of a folder inside the Python environment. In a source checkout it is still ``<repository>/data``, and ``CALIBRAIN_DATA`` overrides both.
*   [FEATURE] Leadfield files, ``LeadfieldBuilder``, and ``DataGenerator`` now carry ``cortical_normals`` (and ``Q_basis`` for MEG). ``LeadfieldBuilder`` computes them for free leadfields built from a forward solution and stores them in cached files, and ``DataGenerator`` saves the normals in the posterior-summary H5 files.
*   [ENHANCEMENT] ``PyQt5``, ``pyvistaqt``, and ``ipywidgets`` are no longer required dependencies. 3D brain plotting now needs the optional extra, ``pip install "calibrain[viz]"``, and logs an install hint when it is missing.
*   [ENHANCEMENT] The package version is now defined only in ``calibrain/__init__.py``, and ``pyproject.toml`` reads it from there.
*   [ENHANCEMENT] The dependencies are now defined only in ``pyproject.toml``, with lower version bounds, and ``pooch`` is now a direct dependency. ``requirements.txt`` and ``docs/requirements.txt`` were removed, ``environment.yml`` only defines Python and ``pip``, and the unused ``myst-parser`` requirement was dropped. The ``docs`` and ``dev`` extras were updated (``dev`` now includes ``pytest-cov`` and ``ruff``).
*   [ENHANCEMENT] The figure scripts in ``viz/`` no longer need ``seaborn``. They use matplotlib and draw the same figures.
*   [ENHANCEMENT] ``python -m calibrain.extract_leadfield`` and ``python -m calibrain.reduce_fwd`` run the data preparation scripts from an installed package.
*   [ENHANCEMENT] ``extract_leadfield.py`` validates its inputs: free extraction from a fixed-only forward solution raises a clear ``ValueError``, and it requires surface source spaces and non-zero cortical normals.
*   [BUGFIX] ``extract_leadfield`` no longer stores local file paths (user and machine names) in the leadfield files it writes. Regenerate existing files before sharing them.
*   [BUGFIX] ``LeadfieldBuilder`` no longer fails with an ``AttributeError`` when ``save_path`` does not exist yet.
*   [BUGFIX] ``extract_leadfield.py`` and ``reduce_fwd.py`` no longer run their batch jobs, which overwrite data files, when they are imported.
*   [BUGFIX] Fixed an undefined ``STRIP_SIZE`` in ``viz/plot_overview_by_nnz.py``.
*   [BUGFIX] Removed the duplicate repository-root ``__init__.py``, which could prevent ``pytest`` from collecting the test suite.
*   [TEST] Added tests for free-orientation simulation, ``extract_leadfield``, ``LeadfieldBuilder`` geometry, the posterior-summary files, and package metadata.
*   [DOCS] Updated tutorials 04, 06, 08, 09, and 12 (metric evaluation) for the cortical-normal simulation interface.
*   [DOCS] Tutorial 05 now downloads the example data, creates the leadfield files from it, and loads them. Its description of the data directory was corrected, and tutorials 05 and 07 clean up their temporary folders.
*   [DOCS] Added tutorial 14, which runs the simulation, estimation, and calibration chain on the real ``fsaverage`` head model from the example data and shows free-orientation activity along the real cortical normals. The tutorials with a duplicate number 10 were renumbered: calibration is now 11, metric evaluation 12, and the end-to-end workflow 13. Links to the old tutorial page names change.
*   [DOCS] Expanded the installation guides (conda, Windows, optional extras, upgrading) and documented the ``CALIBRAIN_DATA`` data directory.
*   [DOCS] Added a release checklist for maintainers and a ``SECURITY.md``.
*   [DOCS] The documentation footer now lists all four authors. The version warning banner for old versions now works: the ``stable`` entry of ``switcher.json`` is marked as the preferred version. Added ``scripts/check_version_switcher.py``, which checks the version menu of every documented version against the live site, a test that keeps ``switcher.json`` clean and requires an entry for the current version, and maintainer notes on the version switcher and its Read the Docs redirects.
*   [DOCS] Added Documentation and Citation sections to the README, switched the pipeline images to PNG, updated the wording and author order for uncertainty quantification, and made the documentation default to the ``stable`` version.
*   [CI] Read the Docs builds now fail on warnings and install the ``docs`` extra.
*   [CI] The tests now run on Python 3.10 to 3.14 and with the lowest allowed dependency versions, a ``ruff`` check was added, the tests run before a release is uploaded to PyPI, and Dependabot checks the dependencies and GitHub Actions monthly.

Version 1.0.3 (2026-07-08)
--------------------------
*   [FEATURE] Added a new tutorials for ``UncertaintyEstimator``.
*   [TEST] Added an initial automated test suite covering ``UncertaintyEstimator``, ``UncertaintyCalibrator``, ``MetricEvaluator``, ``SourceSimulator``, ``SensorSimulator``, ``DataGenerator``, and EEG free-orientation uncertainty paths.
*   [CI] Added a dedicated GitHub Actions test workflow running ``pytest`` on the repository test suite.
*   [DOCS] Added a test-status badge to the README and expanded the related-software positioning text for MNE-Python, CUQIpy, and Uncertainty Toolbox.
*   [DOCS] Added repository-level community files required for external review workflows, including ``CODE_OF_CONDUCT.md`` and ``CONTRIBUTING.md``.
*   [DOCS] Updated the repository, citation page, and documentation landing page to use the current Zenodo DOI.

Version 1.0.2 (2026-06-16)
--------------------------

*   [DOCS] Reworked the installation documentation into separate beginner-oriented PyPI, pip, and conda guides with clearer step-by-step instructions.
*   [DOCS] Refined the tutorial and documentation structure around the current workflow, conceptual overview, glossary, datasets, and runnable gallery pages.
*   [DOCS] Added manual ``Examples using calibrain.XXX`` sections to API reference pages so core classes and workflow entry points link directly to relevant tutorials.

Version 1.0.1 (2026-06-16)
--------------------------

*   [BREAKING] Raised the supported Python version floor to ``>=3.10`` and aligned package metadata, documentation builds, and contributor setup instructions accordingly.
*   [DOCS] Reworked documentation CI so GitHub Actions validates documentation builds while Read the Docs remains the canonical published documentation host.
*   [DOCS] Fixed the documentation version switcher to use explicit Read the Docs ``latest``, ``stable``, and tagged release targets.
*   [BUGFIX] Completed package dependency declarations for wheel-based installation and clean-environment imports.
*   [FEATURE] Added Publishing GitHub Actions workflow for PyPI releases triggered from published GitHub releases, with manual dispatch retained as a fallback.
*   [FEATURE] Added Zenodo DOI citation metadata and surfaced the software DOI throughout the package documentation.
*   [FEATURE] Added a root ``CITATION.cff`` file with author metadata, ORCID identifiers, release metadata, and Zenodo DOI information.
*   [DOCS] Added Read the Docs, PyPI, downloads, license, release, workflow, and DOI badges to the README and documentation landing page.
*   [DOCS] Consolidated the root project README around ``README.md`` and removed the duplicate reStructuredText copy.
*   [DOCS] Standardized software citation text across the README, documentation landing page, and citation page using full author names and Zenodo DOI-based software references.
*   [ENHANCEMENT] Refined the package description and synchronized the README and docs landing page around the current package scope and workflow summary.


Version 1.0.0 (2026-06-11)
--------------------------

*   Version 1.0.0 marks a major consolidation of the CaliBrain package around a
    stable, production-ready workflow. The package is now centered on simulation-based
    uncertainty quantification and calibration in EEG/MEG inverse source imaging, with
    robust support for fixed and free-orientation source models.
*   [BREAKING] Consolidated the supported inverse workflow around ``gamma_map_sflex``,
    ``gamma_lambda_map_sflex``, ``BMN``, and ``BMN_joint``. Older methods such as
    ``gamma_map``, ``eLORETA``, and related cross-validation branches have been
    removed from the supported pipeline.
*   [BREAKING] Reworked calibration from within-subject, across-source fitting to pooled source datasets across subjects, with isotonic regression now fit on subject-level splits rather than a single subject at a time.
*   [FEATURE] Introduce five calibration modes (``precal``, ``post_oracle``, ``post_pooled``, ``post_pooled_mismatch``, and ``post_fixed``) for evaluating calibration performance under different fitting conditions.
*   [FEATURE] Expanded the core simulation stack with stronger support for source simulation, sensor simulation, and leadfield handling across fixed and free-orientation settings.
*   [FEATURE] Added and extended uncertainty quantification and calibration capabilities, including ``pointwise`` and ``aggregated`` uncertainty modes, ``full_cov`` and ``marginal`` free-orientation interval types, componentwise uncertainty handling, and the multiple calibration modes.
*   [FEATURE] Added manifest workflow modules for data generation, aggregation, calibration, and calibration-figure generation, and documented their main entry points and helper methods.
*   [ENHANCEMENT] Refined ``MetricEvaluator`` and related evaluation logic for ``mse``, ``mae``, ``rmse``, ``rmae``, ``mean_posterior_std``, ``emd``, ``mean_signed_deviation``, ``mean_absolute_deviation``, ``max_underconfidence_deviation``, and ``max_overconfidence_deviation``.
*   [BREAKING] Removed the legacy benchmarking class and replaced it with ``DataGenerator``, which wraps solver grids, ``SourceSimulator``, ``SensorSimulator``, ``LeadfieldBuilder``, and posterior-summary generation in a single workflow abstraction.
*   [BUGFIX] Fixed unit handling and source/sensor projection consistency across the simulation pipeline.


Version 0.1.2 (2025-08-19)
--------------------------

*   [FEATURE] Refactored entire codebase into a modular, class-based architecture: `SourceSimulator`, `SensorSimulator`, `MetricEvaluator`, `Visualizer`.
*   [FEATURE] Added empirical evaluation metrics:
    - **Uncertainty**: `mean_posterior_std`
    - **Calibration**: `mean_calibration_error`, `max_underconfidence_deviation`, `max_overconfidence_deviation`, `mean_absolute_deviation`, `mean_signed_deviation`
    - **Spatial Accuracy**: `emd`, `jaccard_error`, `mse`
    - **Detection Performance**: `euclidean_distance`, `f1`, `accuracy`
*   [FEATURE] Integrated `eLORETA` as a new inverse estimator for distributed source reconstruction.
*   [FEATURE] Introduced unit-aware visualization utilities with automatic label scaling for EEG/MEG/source signals.
*   [FEATURE] Added comprehensive examples and tutorials covering simulation, evaluation, and visualization.
*   [FEATURE] Overhauled and expanded documentation with updated installation and usage instructions.
*   [ENHANCEMENT] Streamlined simulation engine with clearer logic, improved configuration handling, and consistent SI unit usage.
*   [ENHANCEMENT] Improved leadfield projection with support for orientation handling and channel filtering.
*   [BUGFIX] Fixed inconsistent unit handling across simulation pipeline ([Issue #18](https://github.com/braindatalab/calibrain/issues/18)):
    - Converted source dipole moments from `nAm` to `Am`
    - Standardized EEG/MEG projections to SI units (V, T)
    - Added logic to format plot labels according to unit scaling


Version 0.1.1 (2025-05-24) 
--------------------------

*   [FEATURE] Include ERP signal generation. Add Multi trial simulation and refactor DataSimulator (`Issue #6 <https://github.com/braindatalab/CaliBrain/issues/6>`_, implemented in `PR #7 <https://github.com/braindatalab/CaliBrain/pull/7>`_).
*   [ENHANCEMENT] Refactored ERP signal generation for smoother waveforms and support for random Hanning window length/duration (`Commit 035d65c <https://github.com/braindatalab/CaliBrain/commit/035d65c0f434ae614d675eb3e03e0585a2bc6254>_`).
*   [BUGFIX] Enhance noise handling in data simulation


Version 0.1.0 (2025-04-28)
--------------------------

*   Initial release.
*   Added `LeadfieldSimulator` for simulating leadfields.
*   Added `DataSimulator` for generating synthetic data.
*   Added `SourceEstimator` for estimating source activity using gamma MAP and eLORETA.
*   Added `UncertaintyEstimator` for estimating uncertainty in source activity.
*   Added `DataGenerator` class for orchestrating source-estimation data generation.
*   Added `utils` module for utility functions.
