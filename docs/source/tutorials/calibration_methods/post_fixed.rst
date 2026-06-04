Post-fixed Calibration
======================

``post_fixed`` fits one isotonic recalibration map at the default condition and
reuses that same mapping across an SNR or sparsity sweep. This method tests
condition generalization while keeping the calibration rule fixed.

Split usage
-----------

For each head, aggregation creates:

- Training: same head, ``alpha_SNR = 0.5``, ``nnz = 5``, ``run_id`` 1--10.
- Evaluation for an SNR sweep: same head, all configured ``alpha_SNR`` values,
  ``nnz = 5``, ``run_id`` 11--35.
- Evaluation for an NNZ sweep: same head, ``alpha_SNR = 0.5``, all configured
  ``nnz`` values, ``run_id`` 11--35.

Calibration uses ``fit_once=True``. The workflow calls
``UncertaintyCalibrator.fit_mapping`` once on the default-condition training
pool, then applies ``UncertaintyCalibrator.evaluate_with_mapping`` to each
evaluation NPZ in the selected sweep.

Assumptions
-----------

- The calibration map is intentionally not refitted for each sweep point.
- Training and evaluation use the same head, solver, noise model, orientation
  type, and free-orientation interval type.
- Differences across the output curves reflect condition mismatch relative to
  the fixed training condition.

Outputs
-------

The workflow writes one JSON record per evaluated sweep dataset. The
``post_calibration`` block contains empirical coverage after applying the fixed
mapping, and its split metadata identifies the evaluation setting.

Use ``post_fixed`` when the scientific question is whether one calibration map
learned at the default condition remains useful when SNR or source sparsity
changes.
