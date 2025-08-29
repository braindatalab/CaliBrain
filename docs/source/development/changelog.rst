Changelog
=========

This document records notable changes to the CaliBrain project.

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
*   Added `Benchmark` class for benchmarking source estimation methods.
*   Added `utils` module for utility functions.
