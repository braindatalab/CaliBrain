Glossary
========

This glossary defines CaliBrain-specific vocabulary together with general
inverse-imaging and neuroimaging terms used throughout the documentation.

.. glossary::
   :sorted:

   aggregated calibration
      Calibration performed on source summaries that have been reduced across
      time rather than on the full source-by-time posterior output.

   adaptive_joint_learning
      A noise-variance workflow mode in which no fixed sensor-noise variance is
      supplied to the inverse solver. Instead, a joint-learning solver
      estimates the noise level during fitting.

   baseline noise variance
      Noise-variance estimate obtained from the pre-stimulus segment of the
      simulated sensor data.

   BMN
      Bayesian minimum norm. In CaliBrain, this refers to a minimum-norm-style
      Bayesian inverse solver that returns posterior means and covariance
      summaries under a common source-variance model.

   BMN_joint
      A BMN variant that can learn a common sensor-noise variance jointly with
      the source hyperparameter.

   calibration
      The process of comparing nominal uncertainty levels with empirical
      coverage and, when needed, learning a recalibration map.

   calibration curve
      A curve showing empirical coverage as a function of nominal coverage.

   coil type
      MNE/FIFF metadata describing the sensor hardware type, for example EEG
      electrodes or MEG magnetometers.

   credible interval
      An interval derived from a posterior distribution. In CaliBrain, scalar
      credible intervals are used for fixed-orientation uncertainty evaluation.

   credible set
      A posterior uncertainty region, such as an interval or ellipsoid,
      associated with a nominal coverage level.

   empirical coverage
      The observed fraction of cases in which the true source value or vector
      lies inside the nominal posterior credible set.

   EMD
      Earth mover's distance. In CaliBrain, this is used as a source-space
      distributional metric for comparing estimated and true source activity.

   EEG
      Electroencephalography. In CaliBrain, EEG can be treated in fixed
      orientation or in free orientation with three local source components.

   forward model
      The mapping from source activity to sensor measurements, represented by
      the leadfield.

   free orientation
      A source model in which each source location has multiple orientation
      components rather than a single fixed scalar coefficient.

   full_cov
      CaliBrain's name for the free-orientation uncertainty representation that
      uses local posterior covariance blocks to define multivariate ellipsoidal
      credible sets.

   gamma_map_sflex
      A sparse Bayesian inverse solver in CaliBrain based on Gamma-MAP and an
      sFLEX basis construction.

   gamma_lambda_map_sflex
      An sFLEX Gamma-MAP variant that jointly learns a noise-related
      regularization parameter.

   head
      Informal workflow term for one subject-specific or geometry-specific
      simulation context used when pooling or splitting calibration data.

   inverse problem
      The problem of recovering latent neural source activity from EEG/MEG
      sensor measurements.

   isotonic regression
      A monotone regression method used in CaliBrain to recalibrate nominal
      coverage levels while preserving ordering.

   leadfield
      The matrix or tensor that maps source amplitudes to sensor measurements.

   marginal
      CaliBrain's name for the free-orientation uncertainty representation that
      calibrates component-wise intervals using marginal variances only.

   MEG
      Magnetoencephalography. In CaliBrain, MEG can be represented in fixed
      orientation or in a reduced free-orientation form with tangential
      components.

   nominal coverage
      The target coverage level attached to a credible interval or credible
      set, for example 0.9 for a nominal 90% credible set.

   noise-variance strategy
      The rule used to provide or estimate the sensor-noise variance for source
      reconstruction. In CaliBrain, the main strategies are ``oracle``,
      ``baseline``, and ``adaptive_joint_learning``.

   oracle noise variance
      The true sensor-noise variance computed from the injected simulation
      noise.

   post_fixed
      A calibration workflow mode in which one recalibration map is fit at a
      reference condition and then reused across a sweep of evaluation
      conditions.

   post_oracle
      A calibration workflow mode in which recalibration is fit and evaluated
      on matched train and evaluation conditions.

   post_pooled
      A calibration workflow mode in which recalibration is fit on pooled
      training data and evaluated on a target condition.

   post_pooled_mismatch
      A calibration workflow mode in which recalibration is fit on pooled but
      intentionally mismatched training conditions and evaluated on the target
      condition.

   posterior covariance
      The covariance matrix returned by an inverse solver to quantify posterior
      uncertainty in source space or coefficient space.

   posterior mean
      The mean of the posterior distribution returned by an inverse solver and
      used as the point estimate of source activity.

   posterior summary
      The stored solver output used downstream in CaliBrain, typically
      including posterior mean, posterior covariance, and associated metadata.

   precal
      A workflow mode that evaluates raw empirical coverage without fitting a
      recalibration map.

   recalibration
      The post-hoc correction of nominal coverage levels using a fitted mapping
      such as isotonic regression.

   reduced free-orientation MEG
      A free-orientation MEG representation with two tangential components per
      source location.

   run manifest
      A tabular index of generated runs used to locate posterior summaries and
      their metadata in downstream workflow stages.

   sensor noise
      Additive noise at the sensor level, simulated in CaliBrain before source
      reconstruction.

   source activity
      Latent neural current amplitudes at source locations over time.

   source space
      The discrete set of candidate source locations used for inverse source
      imaging.

   source_estimator
      The high-level CaliBrain class that wraps an inverse solver and applies
      it to a leadfield and sensor data.

   uncertainty representation
      The geometric object used for calibration, such as a scalar interval,
      component-wise interval family, or local ellipsoid.
