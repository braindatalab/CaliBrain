Package Overview
================


CaliBrain studies how uncertainty estimates behave in simulation-controlled
EEG/MEG inverse source imaging. The package is designed for experiments where
the latent neural source activity is known, the sensor measurements are
simulated through a forward model, and inverse solvers are evaluated by asking
whether their posterior uncertainty is empirically calibrated.

Scientific background
---------------------


In EEG/MEG source imaging, a sensor measurement matrix \(Y \in \mathbb{R}^{M
\times T}\) is modeled as

.. math::

   Y = L X + E,


where \(L\) is the leadfield, \(X\) is source activity, \(E\) is sensor noise,
\(M\) is the number of sensors, and \(T\) is the number of time samples. The
inverse problem is ill-posed because many source configurations can explain the
same sensor data. A useful inverse method should therefore report not only a
point estimate \(\hat{X}\), but also an uncertainty estimate that can be checked
against simulated ground truth.

CaliBrain focuses on coverage calibration. For a nominal coverage level \(c\),
the package constructs credible intervals or ellipsoids and estimates empirical
coverage,

.. math::

   \hat{g}(c) =
   \frac{1}{N}
   \sum_{i=1}^{N}
   \mathbf{1}\left[x_i^{\mathrm{true}} \in C_i(c)\right].


A calibrated uncertainty model satisfies \(\hat{g}(c) \approx c\) over the
nominal coverage grid.

Main features
-------------


- Controlled source simulation with fixed and free orientations.
- Sensor simulation through fixed or free-orientation leadfields.
- Inverse solvers for Gamma-MAP, sFLEX Gamma-MAP, eLORETA, Bayesian minimum
  norm, and joint noise-learning variants.
- Posterior summary storage with a manifest-based workflow for reproducible
  aggregation.
- Experiment-level calibration with isotonic nominal-coverage recalibration.
- Calibration metrics, EMD-based source-space metrics, and paper-style
  calibration figures.

Typical use cases
-----------------


- Benchmarking inverse solvers under controlled signal-to-noise ratios.
- Comparing fixed and free-orientation source models.
- Testing calibration transfer across subjects, source sparsity levels, and
  noise settings.
- Producing reproducible calibration curves for simulation studies.
- Inspecting storage and uncertainty representations used by calibration.

Design principles
-----------------


CaliBrain separates expensive simulation from downstream analysis. The
data-generation stage writes posterior summaries and a manifest. Aggregation
reduces these summaries into calibration-ready datasets. Calibration consumes
only those aggregated datasets and writes JSON summaries. This structure keeps
large numerical artifacts explicit, makes split definitions inspectable, and
allows calibration analyses to be rerun without regenerating simulated data.
