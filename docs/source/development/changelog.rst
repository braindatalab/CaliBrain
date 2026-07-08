Changelog
=========

This document records notable changes to the CaliBrain package.

.. note::
    This page summarizes notable package, workflow, and documentation changes.

Version 1.0.3 (2026-07-08)
--------------------------

*   [TEST] Added an initial automated test suite covering ``UncertaintyEstimator``, ``UncertaintyCalibrator``, ``MetricEvaluator``, ``SourceSimulator``, ``SensorSimulator``, ``DataGenerator``, and EEG free-orientation uncertainty paths.
*   [CI] Added a dedicated GitHub Actions test workflow running ``pytest`` on the repository test suite.
*   [DOCS] Added a test-status badge to the README and expanded the related-software positioning text for MNE-Python, CUQIpy, and Uncertainty Toolbox.
*   [DOCS] Added repository-level community files required for external review workflows, including ``CODE_OF_CONDUCT.md`` and ``CONTRIBUTING.md``.
*   [DOCS] Updated the repository, citation page, and documentation landing page to use the current Zenodo DOI ``10.5281/zenodo.20721580``.

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
    uncertainty estimation and calibration in EEG/MEG inverse source imaging, with
    robust support for fixed and free-orientation source models.
*   [BREAKING] Consolidated the supported inverse workflow around ``gamma_map_sflex``,
    ``gamma_lambda_map_sflex``, ``BMN``, and ``BMN_joint``. Older methods such as
    ``gamma_map``, ``eLORETA``, and related cross-validation branches have been
    removed from the supported pipeline.
*   [BREAKING] Reworked calibration from within-subject, across-source fitting to pooled source datasets across subjects, with isotonic regression now fit on subject-level splits rather than a single subject at a time.
*   [FEATURE] Introduce five calibration modes (``precal``, ``post_oracle``, ``post_pooled``, ``post_pooled_mismatch``, and ``post_fixed``) for evaluating calibration performance under different fitting conditions.
*   [FEATURE] Expanded the core simulation stack with stronger support for source simulation, sensor simulation, and leadfield handling across fixed and free-orientation settings.
*   [FEATURE] Added and extended uncertainty estimation and calibration capabilities, including ``pointwise`` and ``aggregated`` uncertainty modes, ``full_cov`` and ``marginal`` free-orientation interval types, componentwise uncertainty handling, and the multiple calibration modes.
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
