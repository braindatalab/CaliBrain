Scientific Background
=====================


Forward and inverse model
-------------------------


CaliBrain uses the standard linear forward model for source imaging,

.. math::

   Y = L X + E,


where \(Y\) is the sensor data, \(L\) is the leadfield, \(X\) is source activity,
and \(E\) is noise. The inverse problem estimates \(X\) from \(Y\) and \(L\).
Because \(L\) is typically ill-conditioned and \(M \ll N\), the inverse problem
requires regularization or a prior model.

The current codebase implements several inverse solvers:

- ``gamma_map``: sparse Gamma-MAP estimation with fixed or grouped orientation
  coefficients.
- ``gamma_map_sflex``: Gamma-MAP with an sFLEX spatial basis.
- ``gamma_lambda_map_sflex``: sFLEX Gamma-MAP with joint noise learning.
- ``eloreta``: eLORETA-style distributed inverse estimation.
- ``BMN``: Bayesian minimum norm estimation.
- ``BMN_joint``: Bayesian minimum norm estimation with joint noise learning.

Orientation models
------------------


For fixed orientation, each source has one scalar time series:

.. math::

   X \in \mathbb{R}^{N \times T}.


For free orientation, each source has multiple local orientation components:

.. math::

   X \in \mathbb{R}^{N \times K \times T}.


The current pipeline uses \(K=3\) for EEG free-orientation models and \(K=2\)
for reduced-rank MEG free-orientation models. MEG reduced coordinates are
tracked with the per-source ``Q_basis`` array so results can be interpreted in a
consistent local source basis.

Calibration objective
---------------------


For each nominal coverage level \(c \in [0, 1]\), an uncertainty estimator
constructs a credible set \(C_i(c)\). Empirical coverage is

.. math::

   \hat{g}(c) =
   \frac{1}{n}
   \sum_{i=1}^{n}
   \mathbf{1}\left[x_i^{\mathrm{true}} \in C_i(c)\right].


In a calibrated model, \(\hat{g}(c)\) is close to \(c\). CaliBrain estimates
the forward calibration curve on calibration runs and uses isotonic regression
to learn a monotone map from target coverage to recalibrated nominal coverage.

Limitations
-----------


Calibration curves are conditional on the simulation design, leadfield set,
noise model, source sparsity, solver, and train/test split. They should not be
interpreted as universal guarantees for all EEG/MEG source imaging settings.
Coverage degeneracies can occur when many sources have exactly zero truth, zero
posterior mean, and zero posterior variance; in that case empirical coverage can
be high even at nominal coverage zero.
