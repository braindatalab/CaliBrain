Pre-calibration
===============

``precal`` evaluates raw posterior uncertainty without fitting an isotonic
recalibration map. It is the reference condition for all post-calibration
methods.

Split usage
-----------

Aggregation reads the manifest CSV and selects only evaluation runs:

- ``subject``: all configured heads.
- ``alpha_SNR``: ``0.5``.
- ``nnz``: ``5``.
- ``run_id``: 11--35.

This produces one evaluation directory per solver/noise/orientation setting.
Calibration then sets ``fit_calibration=False`` and evaluates each aggregated
NPZ directly. No training NPZ is read.

Assumptions
-----------

- The raw posterior representation is already available in aggregated NPZ
  files.
- Evaluation runs are held out from any calibration fitting, because no fitting
  occurs in this method.
- Results should be interpreted as the empirical coverage of the solver's
  original uncertainty estimate.

Outputs
-------

The calibration workflow writes JSON records containing:

- ``pre_calibration``: empirical coverage and calibration metrics.
- ``post_calibration``: identical or effectively raw evaluation information,
  because no recalibration map is fitted.
- ``emd``: optional spatial metric when source coordinates are available.

Use ``precal`` when reporting whether a solver is already calibrated before any
learned recalibration step.
