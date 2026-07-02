Conceptual Overview
===================

CaliBrain is a simulation-based toolbox for uncertainty quantification and
uncertainty calibration in Bayesian M/EEG source imaging. Its scope is
deliberately specific: it studies inverse methods that provide closed-form
Gaussian posterior summaries, and it asks whether the reported posterior
credibility agrees with empirical coverage under controlled experimental
conditions.

The scientific target is therefore not only source reconstruction accuracy, but
also the reliability of posterior uncertainty. A posterior covariance is useful
only if the credible regions derived from it have interpretable nominal levels.

Scientific setting
------------------

The forward model is

.. math::

   y(t) = L x(t) + e(t),

where :math:`y(t)` is the sensor measurement, :math:`L` is the leadfield,
:math:`x(t)` is the source vector, and :math:`e(t)` is additive sensor noise.
Over :math:`T` time samples, CaliBrain works with the stacked model

.. math::

   Y = L X + E.

Because the M/EEG inverse problem is ill-posed, posterior uncertainty is not a
secondary diagnostic; it is part of the inference problem itself. CaliBrain
therefore evaluates Bayesian inverse solvers as posterior procedures that
return:

- a posterior mean;
- a posterior covariance;
- solver-specific hyperparameters or diagnostics.

Orientation-aware source models
-------------------------------

The framework unifies three source-model settings through the local source
dimension :math:`d_i`:

- fixed orientation: :math:`d_i = 1`;
- reduced free-orientation MEG: :math:`d_i = 2`;
- free-orientation EEG: :math:`d_i = 3`.

This distinction determines the uncertainty object associated with each source
location:

- fixed orientation uses one-dimensional credible intervals;
- reduced free-MEG uses two-dimensional credible ellipses in the local
  tangential plane;
- free-orientation EEG uses three-dimensional credible ellipsoids.

The point of this formulation is not cosmetic. It makes calibration comparable
across modalities and source models while preserving the correct local
geometry.

Posterior uncertainty representation
------------------------------------

The framework starts from the full Gaussian posterior and then uses local
marginal blocks for uncertainty analysis. For each source location
:math:`i`, CaliBrain works with the local posterior block
:math:`\Sigma_{ii}` rather than with only a scalar global summary.

Since the implemented calibration workflow is based on temporally aggregated
posterior summaries, the uncertainty analysis is performed on

.. math::

   \bar{x} = \frac{1}{T} \sum_{t=1}^{T} x(t),

with aggregated posterior

.. math::

   \bar{x}_i \mid Y \sim \mathcal{N}(\bar{\mu}_i, \bar{\Sigma}_{ii}).

Under the current model used in the package, the local covariance is static
over time, so the aggregated covariance scales by :math:`1/T`.

For nominal credibility level :math:`c \in (0, 1)`, CaliBrain defines the
source-wise credible region

.. math::

   \mathcal{C}_i(c)
   =
   \left\{
   z \in \mathbb{R}^{d_i}
   :
   (z - \bar{\mu}_i)^\top \bar{\Sigma}_{ii}^{-1} (z - \bar{\mu}_i)
   \le \chi^2_{d_i}(c)
   \right\}.

This quadratic-form construction yields the interval/ellipse/ellipsoid cases
as dimension-matched specializations of the same definition.

Why CaliBrain distinguishes dense and sparse Bayesian solvers
-------------------------------------------------------------

The toolbox focuses on two structurally different Type-II Bayesian solver
families:

- ``BMN`` and ``BMN_joint`` as dense shared-variance models;
- ``gamma_map_sflex`` and ``gamma_lambda_map_sflex`` as sparse source-wise
  variance models with sFLEX support expansion.

This distinction matters scientifically because sparse Bayesian learning can
prune sources so aggressively that posterior variances collapse toward zero at
inactive locations. Without additional structure, this can make credible-region
construction degenerate or ill-defined.

CaliBrain addresses that issue by using sparse basis field expansions
(``sFLEX``), which impose sparsity in coefficient space while restoring full
source-space support. The result is a source-space posterior covariance that
remains usable for uncertainty quantification and calibration.

Calibration target
------------------

For a nominal credibility level :math:`c`, CaliBrain evaluates whether the
aggregated ground-truth source block falls inside the corresponding credible
region. The empirical coverage is

.. math::

   \hat{g}(c)
   =
   \frac{1}{N}
   \sum_{i=1}^{N}
   \mathbf{1}\!\left[\bar{x}^{\mathrm{true}}_i \in \mathcal{C}_i(c)\right].

The calibration curve is the graph of :math:`\hat{g}(c)` against :math:`c`.

- a curve above the diagonal indicates underconfidence;
- a curve below the diagonal indicates overconfidence.

This definition is shared across fixed orientation, reduced free-MEG, and
free-EEG settings; what changes is only the local uncertainty geometry.

Post-hoc recalibration
----------------------

When nominal credibility and empirical coverage disagree systematically,
CaliBrain applies post-hoc isotonic recalibration. The procedure is:

1. estimate empirical calibration curves on training runs;
2. fit a monotone isotonic regression map;
3. invert that fitted map numerically;
4. evaluate the recalibrated nominal levels on held-out runs.

This recalibration does not change the posterior mean, posterior covariance, or
uncertainty representation. It changes only the mapping from nominal
credibility to evaluated coverage.

The workflow modes implemented in the package differ only in how the training
and evaluation runs are chosen around this common recalibration step:

- ``precal`` evaluates raw empirical coverage without fitting a map;
- ``post_oracle`` fits and evaluates under matched conditions;
- ``post_pooled`` fits on pooled matched conditions and evaluates on a target
  condition;
- ``post_pooled_mismatch`` fits on intentionally mismatched pooled conditions;
- ``post_fixed`` fits once at a reference condition and reuses that map across a
  sweep of evaluation settings.

Current workflow
----------------

The current workflow is organized around these high-level stages:

1. ``SourceSimulator`` generates ground-truth source activity.
2. ``LeadfieldBuilder`` provides a leadfield.
3. ``SensorSimulator`` projects sources to sensors and adds Gaussian noise.
4. ``SourceEstimator`` runs a Bayesian inverse solver.
5. ``UncertaintyEstimator`` converts posterior summaries into calibration-ready
   uncertainty objects.
6. ``UncertaintyCalibrator`` evaluates pre-calibration curves and, if
   requested, fits and applies isotonic recalibration.

In larger studies, workflow scripts repeat these steps across runs and
conditions so that reconstruction accuracy, posterior uncertainty magnitude,
and calibration can be compared separately.

Implemented scope
-----------------

The current codebase implements:

- source models

  - fixed orientation
  - reduced free-orientation MEG
  - free-orientation EEG

- solver families

  - ``gamma_map_sflex``
  - ``gamma_lambda_map_sflex``
  - ``BMN``
  - ``BMN_joint``

- noise-variance strategies

  - ``oracle``
  - ``baseline``
  - ``adaptive_joint_learning``

- uncertainty summaries

  - fixed-orientation scalar marginal intervals
  - free-orientation ``marginal`` componentwise intervals
  - free-orientation ``full_cov`` local ellipsoidal diagnostics

- calibration outputs

  - empirical coverage curves
  - MAD, MSD, MUD, and MOD summary metrics
  - isotonic post-hoc recalibration under multiple split designs

Interpretation
--------------

The main scientific distinction in CaliBrain is that uncertainty magnitude and
uncertainty calibration are different objects.

- posterior mean and posterior covariance determine reconstruction error and
  uncertainty size;
- calibration evaluates whether nominal posterior credibility is empirically
  reliable;
- post-hoc recalibration can improve the nominal interpretation of the same
  posterior summaries without changing the underlying reconstruction.

This is the conceptual role of CaliBrain: it provides a unified framework for
studying reconstruction, posterior uncertainty, and empirical calibration
together, rather than treating uncertainty as an informal by-product of source
estimation.
