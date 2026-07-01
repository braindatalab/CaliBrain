Conceptual Overview
===================

CaliBrain is a simulation-based toolbox for studying uncertainty calibration in
EEG/MEG inverse source imaging. The toolbox is intended for settings in
which source activity is known by construction, sensor measurements are
generated through a specified forward model, and inverse solvers return both a
posterior mean and a posterior uncertainty summary. The central question is not
only whether an inverse method reconstructs the source signal accurately, but
whether its reported uncertainty has valid empirical coverage under controlled
experimental conditions.

Scientific background
---------------------

In EEG/MEG source imaging, a sensor measurement matrix
:math:`Y \in \mathbb{R}^{M \times T}` is modeled as

.. math::

   Y = L X + E,


where :math:`L` is the leadfield, :math:`X` is source activity, :math:`E` is
sensor noise, :math:`M` is the number of sensors, and :math:`T` is the number
of time samples. The inverse problem is ill-posed: distinct source
configurations can induce similar sensor patterns, and uncertainty is therefore
an intrinsic part of the inference problem. In CaliBrain, inverse methods are
evaluated as posterior procedures. Each method produces a point estimate
:math:`\hat{X}` together with a posterior covariance or a reduced uncertainty
representation derived from it.

CaliBrain focuses on coverage calibration. For a nominal coverage level
:math:`c`,
the package constructs credible intervals or ellipsoids and estimates empirical
coverage,

.. math::

   \hat{g}(c) =
   \frac{1}{N}
   \sum_{i=1}^{N}
   \mathbf{1}\left[x_i^{\mathrm{true}} \in C_i(c)\right].


A calibrated uncertainty model satisfies :math:`\hat{g}(c) \approx c` over the
nominal coverage grid.

In practice, CaliBrain evaluates both pre-calibration and post-calibration
behavior. Pre-calibration curves quantify how the raw posterior uncertainty
behaves. Post-calibration curves quantify how that behavior changes after
learning a monotone recalibration map, typically by isotonic regression on a
training split and evaluating it on a held-out split.

Workflow
--------

The current CaliBrain workflow is organized around the following stages:

1. ``SourceSimulator`` generates source-level ground truth.
2. ``LeadfieldBuilder`` loads or constructs a leadfield.
3. ``SensorSimulator`` projects sources to sensors and adds noise.
4. ``SourceEstimator`` applies an active inverse solver:

   - ``gamma_map_sflex``
   - ``gamma_lambda_map_sflex``
   - ``BMN``
   - ``BMN_joint``

5. ``UncertaintyEstimator`` converts posterior summaries into calibration-ready
   intervals or ellipsoids.
6. ``UncertaintyCalibrator`` fits and evaluates isotonic recalibration maps.
7. Workflow scripts batch these operations across runs, then aggregate and
   calibrate on disk.

This separation is deliberate. Data generation produces simulation outputs and
posterior summaries. Aggregation reduces those outputs into calibration-ready
representations. Calibration then operates on the aggregated summaries rather
than rerunning the inverse solvers. This design supports controlled benchmark
studies across source models, noise regimes, and calibration strategies.

Implemented methods
-------------------

- Source models

  - fixed orientation
  - free-orientation EEG
  - reduced free-orientation MEG

- Inverse solvers

  - ``gamma_map_sflex``
  - ``gamma_lambda_map_sflex``
  - ``BMN``
  - ``BMN_joint``

- Noise-variance strategies

  - ``oracle``
  - ``baseline``
  - ``adaptive_joint_learning``

- Uncertainty representations

  - fixed-orientation marginal variances
  - free-orientation ``marginal`` intervals
  - free-orientation ``full_cov`` ellipsoids

- Calibration methods

  - Recalibration model

    - isotonic regression for monotone recalibration of nominal coverage

  - Evaluation modes

    - ``precal``: evaluate raw empirical coverage without fitting a recalibration map
    - ``post_oracle``: fit on a matched training split and evaluate on a matched evaluation split
    - ``post_pooled``: fit on pooled training data and evaluate on a target evaluation split
    - ``post_pooled_mismatch``: fit on pooled but intentionally mismatched training conditions and evaluate on the target split
    - ``post_fixed``: fit one recalibration map at a reference condition and reuse it across a sweep of evaluation conditions

Research use
------------

CaliBrain is intended for controlled workflow studies and benchmark-style
evaluation rather than routine source analysis. Typical uses include:

- comparing inverse solvers under controlled source sparsity and signal-to-noise
  conditions;
- studying whether posterior uncertainty is under-confident or over-confident;
- evaluating whether recalibration learned under one condition transfers to
  another;
- comparing fixed and free-orientation uncertainty representations;
- producing calibration figures and benchmark summaries from simulation
  experiments.
