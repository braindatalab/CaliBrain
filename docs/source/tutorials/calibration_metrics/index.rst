Calibration and Metrics
=======================


These tutorials explain empirical coverage, recalibration, and quantitative
evaluation.


Computing Calibration Curves
----------------------------


- **Description**: Compute pre-calibration empirical coverage over a nominal
  coverage grid.
- **Target audience**: Users interpreting calibration output.
- **Difficulty**: Beginner.
- **Estimated duration**: 25 minutes.
- **Prerequisites**: Estimating Predictive Uncertainty.
- **Main APIs introduced**: ``UncertaintyEstimator``,
  ``MetricEvaluator.calibration_curve``.
- **Scientific concepts**: Nominal coverage, empirical coverage, perfect
  calibration.
- **Expected outputs**: Calibration-curve arrays.
- **Figures or plots**: Calibration curve with diagonal reference.
- **Gallery executable**: Yes.

Expected Calibration Error and Related Metrics
----------------------------------------------


- **Description**: Summarize deviations between nominal and empirical coverage
  using the metrics implemented in the current evaluator.
- **Target audience**: Users comparing methods quantitatively.
- **Difficulty**: Intermediate.
- **Estimated duration**: 30 minutes.
- **Prerequisites**: Computing Calibration Curves.
- **Main APIs introduced**: ``MetricEvaluator.calibration_metrics_4``,
  ``MetricEvaluator.evaluate_all``.
- **Scientific concepts**: Mean signed deviation, mean absolute deviation,
  maximum underconfidence, maximum overconfidence.
- **Expected outputs**: Metric dictionary and comparison table.
- **Figures or plots**: Bar plot of calibration metrics.
- **Gallery executable**: Yes.

Coverage-Based Evaluation
-------------------------


- **Description**: Fit an isotonic recalibration map and evaluate coverage on a
  held-out split.
- **Target audience**: Workflow users.
- **Difficulty**: Intermediate.
- **Estimated duration**: 45 minutes.
- **Prerequisites**: Computing Calibration Curves.
- **Main APIs introduced**: ``UncertaintyCalibrator.calibrate``,
  ``UncertaintyCalibrator.fit_mapping``,
  ``UncertaintyCalibrator.evaluate_with_mapping``.
- **Scientific concepts**: Train/eval split, isotonic regression, recalibrated
  nominal coverages.
- **Expected outputs**: Pre- and post-calibration blocks.
- **Figures or plots**: Pre/post calibration curves.
- **Gallery executable**: Yes for a small synthetic split.

Spatial Calibration Metrics
---------------------------


- **Description**: Compute spatial error metrics for aggregated evaluation
  datasets when source coordinates are available.
- **Target audience**: Users evaluating localization and calibration jointly.
- **Difficulty**: Advanced.
- **Estimated duration**: 40 minutes.
- **Prerequisites**: Interpreting Uncertainty Maps.
- **Main APIs introduced**: ``compute_dataset_emd``,
  ``MetricEvaluator.emd``, ``get_subset_source_rr``.
- **Scientific concepts**: Earth Mover's Distance, source-space localization,
  reduced versus lifted source representations.
- **Expected outputs**: EMD value per evaluation dataset.
- **Figures or plots**: Optional spatial mass comparison.
- **Gallery executable**: Conditional; requires lightweight coordinates.

Comparing Calibration Across Methods
------------------------------------


- **Description**: Compare ``precal``, ``post_oracle``, ``post_pooled``,
  ``post_pooled_mismatch``, and ``post_fixed``.
- **Target audience**: Users reproducing paper comparisons.
- **Difficulty**: Advanced.
- **Estimated duration**: 60 minutes for a reduced benchmark.
- **Prerequisites**: Coverage-Based Evaluation.
- **Main APIs introduced**: ``run_calibration``,
  ``stack_empirical_curves``, ``iter_calibration_records``.
- **Scientific concepts**: Matched calibration, pooled calibration,
  cross-subject mismatch, fixed-setting generalization.
- **Expected outputs**: Calibration JSON groups and summary statistics.
- **Figures or plots**: Mean ± standard-deviation calibration curves.
- **Gallery executable**: No for paper-scale; yes for a reduced fixture.
