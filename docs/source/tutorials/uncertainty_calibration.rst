Uncertainty Calibration
=======================


Pre-calibration
---------------


Pre-calibration evaluates the empirical coverage of a solver's raw posterior
uncertainty on the evaluation split. No isotonic model is fitted. In the config,
this corresponds to ``fit_calibration=False``.

Isotonic recalibration
----------------------


Let \(g(c)\) denote the empirical coverage obtained when intervals are built
with nominal coverage \(c\). CaliBrain fits a monotone approximation
\(\hat{g}\) using isotonic regression on calibration runs. It then numerically
inverts this map to obtain recalibrated coverage levels:

.. math::

   c_{\mathrm{recal}}(c) \approx \hat{g}^{-1}(c).


The evaluation split is then measured by constructing intervals at
\(c_{\mathrm{recal}}(c)\) while reporting empirical coverage against the
original target grid \(c\).

Implemented calibration strategies
----------------------------------


The default configs implement the following strategies:

- :doc:`calibration_methods/precal`: evaluate raw uncertainty on evaluation
  runs.
- :doc:`calibration_methods/post_oracle`: fit and evaluate at the same default
  condition for each head.
- :doc:`calibration_methods/post_pooled`: fit on a pooled within-head SNR/NNZ
  grid and evaluate at the default condition.
- :doc:`calibration_methods/post_pooled_mismatch`: fit on other heads and
  evaluate on the held-out head.
- :doc:`calibration_methods/post_fixed`: fit once at the default condition and
  reuse the same mapping across an SNR or NNZ sweep.

See :doc:`calibration_methods/index` for split definitions, assumptions, and
outputs for each method.

Interval types
--------------


For fixed orientation, calibration uses marginal intervals based on
``posterior_var``.

For free orientation, ``free_interval_type`` controls the uncertainty set:

- ``full_cov``: use per-source covariance blocks to form ellipses or ellipsoids.
- ``marginal``: use componentwise marginal intervals and pool over components.

Common pitfalls
---------------


- A flat calibration curve near low nominal coverage is not necessarily a
  plotting error. It can occur when many sources have exact zero truth, exact
  zero posterior mean, and zero posterior variance.
- Calibration maps can clamp to 0 or 1 when the empirical curve has no inverse
  inside the open interval.
- Do not mix fixed and free-orientation datasets in a training pool.
- Do not mix EEG and MEG free-orientation datasets in a training pool; their
  local component dimensions differ.
