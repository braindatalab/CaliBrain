Post-oracle Calibration
=======================

``post_oracle`` fits an isotonic recalibration map using calibration runs from
the same head and the same experimental condition as the evaluation runs. It is
the matched-condition post-calibration baseline.

Split usage
-----------

For each head, aggregation creates two directories:

- Training: same head, ``alpha_SNR = 0.5``, ``nnz = 5``, ``run_id`` 1--10.
- Evaluation: same head, ``alpha_SNR = 0.5``, ``nnz = 5``, ``run_id`` 11--35.

Calibration fits one mapping per head, solver, noise model, orientation type,
and free-orientation interval type. The fitted map is evaluated on the
corresponding held-out evaluation runs from the same head.

Assumptions
-----------

- Calibration and evaluation data come from the same head and same test
  condition.
- Calibration runs and evaluation runs are disjoint.
- The method estimates the best-case recalibration performance under matched
  data-generating conditions; it should not be interpreted as cross-condition
  or cross-head generalization.

Outputs
-------

The workflow writes one calibration JSON record per evaluated NPZ. Each record
stores:

- ``pre_calibration``: raw empirical coverage on the evaluation split.
- ``train_empirical_coverages``: empirical coverage used to fit the isotonic
  map.
- ``post_calibration``: empirical coverage after applying the matched isotonic
  map to the evaluation split.

Use ``post_oracle`` as the main reference for how well recalibration can work
when the calibration data match the evaluation condition.
