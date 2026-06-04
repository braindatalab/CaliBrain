Calibration Methods
===================

CaliBrain separates uncertainty evaluation from uncertainty recalibration.
The implemented calibration methods differ only in how the calibration
training split is selected and how the fitted isotonic mapping is reused.
They all evaluate empirical coverage on held-out aggregated NPZ datasets and
write calibration JSON records.

The paper-style defaults use the following split convention:

- ``run_id`` 1--10 are calibration runs.
- ``run_id`` 11--35 are evaluation runs.
- The default test condition is ``alpha_SNR = 0.5`` and ``nnz = 5``.
- Four heads are used by default: ``CC120166``, ``CC120264``, ``CC120309``,
  and ``CC120313``.
- Training and evaluation splits must not mix orientation type, coil type,
  solver, or noise model.

.. list-table:: Implemented calibration methods
   :header-rows: 1
   :widths: 20 28 28 24

   * - Method
     - Calibration data
     - Evaluation data
     - Main question
   * - ``precal``
     - None
     - Test runs at the default condition
     - How calibrated is the raw posterior uncertainty?
   * - ``post_oracle``
     - Same head, default condition
     - Same head, default condition
     - What is the best matched recalibration baseline?
   * - ``post_pooled``
     - Same head, pooled SNR × NNZ grid
     - Same head, default condition
     - Does more within-head calibration data help?
   * - ``post_pooled_mismatch``
     - Other heads, pooled SNR × NNZ grid
     - Held-out head, default condition
     - Does calibration transfer across heads?
   * - ``post_fixed``
     - Same head, default condition
     - Same head, SNR or NNZ sweep
     - Does one fixed mapping generalize across conditions?

.. toctree::
   :maxdepth: 1

   01. Pre-calibration <precal>
   02. Post-oracle calibration <post_oracle>
   03. Post-pooled calibration <post_pooled>
   04. Post-pooled mismatch calibration <post_pooled_mismatch>
   05. Post-fixed calibration <post_fixed>
