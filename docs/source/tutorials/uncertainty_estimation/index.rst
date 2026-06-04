Uncertainty Estimation
======================


These tutorials focus on intervals and uncertainty representations before
calibration maps are fitted.


Estimating Predictive Uncertainty
---------------------------------


- **Description**: Compute interval coverage inputs from posterior means and
  variances.
- **Target audience**: Users learning ``UncertaintyEstimator``.
- **Difficulty**: Beginner to intermediate.
- **Estimated duration**: 30 minutes.
- **Prerequisites**: Running a Basic Source Reconstruction Workflow.
- **Main APIs introduced**: ``UncertaintyEstimator``.
- **Scientific concepts**: Predictive uncertainty, posterior standard
  deviation, credible intervals.
- **Expected outputs**: Interval bounds and empirical coverage arrays.
- **Figures or plots**: Coverage curve and interval examples.
- **Gallery executable**: Yes.

Working with Confidence and Credible Intervals
----------------------------------------------


- **Description**: Compare scalar fixed-orientation intervals with free
  orientation ``full_cov`` and ``marginal`` interval types.
- **Target audience**: Users analyzing fixed/free orientation experiments.
- **Difficulty**: Intermediate.
- **Estimated duration**: 35 minutes.
- **Prerequisites**: Handling Trial-Level and Source-Level Outputs.
- **Main APIs introduced**: ``UncertaintyEstimator``,
  ``UncertaintyCalibrator``.
- **Scientific concepts**: Marginal intervals, local covariance ellipsoids,
  componentwise coverage.
- **Expected outputs**: Interval coverage summaries for fixed and free
  orientation.
- **Figures or plots**: Interval and covariance-block visualizations.
- **Gallery executable**: Yes for synthetic data.

Comparing Uncertainty Estimation Methods
----------------------------------------


- **Description**: Compare uncertainty representations produced by different
  solvers or interval assumptions before recalibration.
- **Target audience**: Users evaluating solver uncertainty quality.
- **Difficulty**: Intermediate.
- **Estimated duration**: 45 minutes.
- **Prerequisites**: Comparing Inverse Solvers.
- **Main APIs introduced**: ``MetricEvaluator``,
  ``UncertaintyEstimator``, solver functions.
- **Scientific concepts**: Mean posterior standard deviation, raw calibration,
  uncertainty sharpness versus reliability.
- **Expected outputs**: Method comparison table and pre-calibration curves.
- **Figures or plots**: Pre-calibration curves per method.
- **Gallery executable**: Yes for small synthetic comparisons.

Interpreting Uncertainty Maps
-----------------------------


- **Description**: Visualize spatial patterns of posterior variance or
  covariance-derived uncertainty.
- **Target audience**: Users interpreting source-space uncertainty.
- **Difficulty**: Intermediate.
- **Estimated duration**: 30 minutes.
- **Prerequisites**: Estimating Predictive Uncertainty.
- **Main APIs introduced**: ``Visualizer``, ``MetricEvaluator.emd``.
- **Scientific concepts**: Spatial uncertainty concentration, source-space
  localization error, EMD.
- **Expected outputs**: Source-level uncertainty arrays and optional spatial
  metrics.
- **Figures or plots**: Source uncertainty maps or sorted variance plots.
- **Gallery executable**: Yes if using lightweight source coordinates.
