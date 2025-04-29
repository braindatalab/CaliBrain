Changelog
=========

This document records notable changes to the CaliBrain project.

Version 1.0.0 (2025-04-28)
--------------------------

*   Initial release.
*   Added `LeadfieldSimulator` for simulating leadfields.
*   Added `DataSimulator` for generating synthetic data.
*   Added `SourceEstimator` for estimating source activity using gamma MAP and eLORETA.
*   Added `UncertaintyEstimator` for estimating uncertainty in source activity.
*   Added `Benchmark` class for benchmarking source estimation methods.
*   Added `utils` module for utility functions.

Version 1.0.1 (Upcoming)
------------------------

*   Add support for additional source estimation methods (e.g. Baysian Minimum Norm)
*   Improve documentation and examples.
*   Add support for other noise models (baseline, cross-validation, joint-learning). 
*   Add support for multi CPU/GPU processing.
*   Add uncertainty calibration metrics
*   Add docstring.