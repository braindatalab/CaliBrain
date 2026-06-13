Changelog
=========

This document records notable changes to the CaliBrain project.

.. note::

   This page is kept as a historical record. Older entries describe earlier
   package states and may mention methods or workflow paths that are no longer
   part of the current supported pipeline.

Current status
--------------

Version 1.0.0 marks a major consolidation of the CaliBrain package around a
stable, production-ready workflow. The package is now centered on simulation-based
uncertainty estimation and calibration in EEG/MEG inverse source imaging, with
robust support for fixed and free-orientation source models.

Supported inverse solvers
~~~~~~~~~~~~~~~~~~~~~~~~~

The current supported workflow pipeline is built around the following inverse
solvers:

* ``gamma_map_sflex`` — Sparse Flexible Gamma MAP
* ``gamma_lambda_map_sflex`` — Joint Gamma-Lambda MAP
* ``BMN`` — Bayesian Minimum Norm
* ``BMN_joint`` — Bayesian Minimum Norm with joint noise learning

Each solver supports both fixed-orientation and free-orientation source
configurations, with unit-aware handling of EEG/MEG signal scaling and posterior
uncertainty summaries.

Deprecated and removed methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Older methods such as ``gamma_map``, ``eLORETA``, and related cross-validation
branches have been removed from the supported pipeline. These may still appear
in historical changelog entries below but are no longer maintained or tested.

Unreleased
----------

*   [BREAKING] Raised the supported Python version floor to ``>=3.10`` and aligned package metadata, documentation builds, and contributor setup instructions accordingly.
*   [DOCS] Reworked documentation CI into separate build-check and deploy workflows for GitHub Actions.
*   [DOCS] Fixed the documentation version switcher to use explicit ``stable``, ``dev``, and tagged release targets, with version-aware Sphinx configuration and versioned GitHub Pages deployment paths.


Version 1.0.0 (2026-06-11)
--------------------------

*   [BREAKING] Consolidated the supported inverse workflow around     ``gamma_map_sflex``, ``gamma_lambda_map_sflex``, ``BMN``, and ``BMN_joint``.
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
