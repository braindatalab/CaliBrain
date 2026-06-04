Post-pooled Mismatch Calibration
================================

``post_pooled_mismatch`` evaluates cross-head transfer of recalibration. The
held-out head is never used for fitting; calibration is fitted from the other
heads pooled across the SNR and sparsity grid.

Split usage
-----------

For each held-out head ``H``, aggregation creates:

- Training: all heads except ``H``, all configured ``alpha_SNR`` values, all
  configured ``nnz`` values, ``run_id`` 1--10.
- Evaluation: held-out head ``H``, ``alpha_SNR = 0.5``, ``nnz = 5``,
  ``run_id`` 11--35.

With four default heads and a 5 × 5 condition grid, the training pool contains
``3 × 5 × 5 × 10 = 750`` calibration runs before aggregation for each held-out
head.

Assumptions
-----------

- The held-out head is excluded from all calibration fitting.
- All pooled training datasets share solver, noise model, orientation type,
  coil type, and compatible posterior representation.
- The method intentionally introduces subject/head mismatch and should be
  interpreted as transfer performance, not as a matched calibration baseline.

Outputs
-------

For each held-out head, solver, noise model, and orientation setting, the
workflow writes JSON records containing:

- raw held-out-head evaluation coverage;
- pooled other-head training coverage;
- post-calibration coverage on the held-out head.

Use ``post_pooled_mismatch`` to quantify how sensitive the recalibration map is
to head-specific geometry, lead-field properties, and subject-level mismatch.
