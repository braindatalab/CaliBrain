Post-pooled Calibration
=======================

``post_pooled`` fits an isotonic recalibration map from a larger within-head
calibration pool. The training pool spans the configured SNR and sparsity grid,
but evaluation remains at the default test condition.

Split usage
-----------

For each head, aggregation creates:

- Training: same head, all configured ``alpha_SNR`` values, all configured
  ``nnz`` values, ``run_id`` 1--10.
- Evaluation: same head, ``alpha_SNR = 0.5``, ``nnz = 5``, ``run_id`` 11--35.

With the default grid, the training pool contains ``5 × 5 × 10 = 250`` runs per
head before aggregation. Calibration fits one isotonic map per head, solver,
noise model, orientation type, and free-orientation interval type.

Assumptions
-----------

- Calibration data are from the same head as evaluation data.
- The pooled SNR/NNZ grid is scientifically meaningful for the evaluated
  condition.
- The larger training pool may reduce variance, but it can introduce bias if
  the calibration curve changes substantially across SNR or sparsity.

Outputs
-------

The JSON output has the same structure as ``post_oracle``:

- raw evaluation coverage in ``pre_calibration``;
- pooled training coverage in ``train_empirical_coverages``;
- recalibrated evaluation coverage in ``post_calibration``.

Use ``post_pooled`` to test whether additional within-head calibration data
improves robustness relative to the matched but smaller ``post_oracle`` split.
