Advanced Topics
===============


Advanced tutorials focus on extension and scale while preserving the current
workflow contracts.


Custom Uncertainty Estimators
-----------------------------


- **Description**: Add uncertainty-estimation logic that produces interval
  inputs compatible with calibration.
- **Target audience**: Method developers.
- **Difficulty**: Advanced.
- **Estimated duration**: 60 minutes.
- **Prerequisites**: Estimating Predictive Uncertainty.
- **Main APIs introduced**: ``UncertaintyEstimator``,
  ``UncertaintyCalibrator``.
- **Scientific concepts**: Alternative uncertainty summaries, interval
  construction, calibration compatibility.
- **Expected outputs**: Custom uncertainty arrays and calibration curves.
- **Figures or plots**: Comparison against default intervals.
- **Gallery executable**: Yes for a small synthetic method.

Custom Calibration Metrics
--------------------------


- **Description**: Add new metrics while keeping calibration JSON outputs
  interpretable and reproducible.
- **Target audience**: Researchers extending evaluation.
- **Difficulty**: Advanced.
- **Estimated duration**: 60 minutes.
- **Prerequisites**: Expected Calibration Error and Related Metrics.
- **Main APIs introduced**: ``MetricEvaluator``, ``calibration_storage``.
- **Scientific concepts**: Metric definitions, aggregation across runs,
  schema stability.
- **Expected outputs**: Extended metric dictionary and JSON-compatible records.
- **Figures or plots**: Metric comparison plot.
- **Gallery executable**: Yes.

Custom Simulators
-----------------


- **Description**: Extend simulation while preserving data-generation metadata
  and posterior-summary storage.
- **Target audience**: Method developers and simulation researchers.
- **Difficulty**: Advanced.
- **Estimated duration**: 60--90 minutes.
- **Prerequisites**: Creating Benchmark Datasets.
- **Main APIs introduced**: ``SourceSimulator``, ``SensorSimulator``,
  ``DataGenerator``.
- **Scientific concepts**: Simulation assumptions, source sparsity,
  noise-generation design.
- **Expected outputs**: Custom simulated sources/sensors and manifest rows.
- **Figures or plots**: Simulation diagnostic plots.
- **Gallery executable**: Conditional; should remain lightweight.

Scaling to Large Experiments
----------------------------


- **Description**: Plan storage, runtime, and workflow execution for large
  paper-style simulation grids.
- **Target audience**: Users running production benchmarks.
- **Difficulty**: Advanced.
- **Estimated duration**: 45 minutes plus execution time.
- **Prerequisites**: Reproducing Paper Experiments.
- **Main APIs introduced**: Workflow config files, ``generation_n_jobs``,
  aggregation split selectors.
- **Scientific concepts**: Disk budget, raw versus reduced artifacts,
  reproducibility under parallel execution.
- **Expected outputs**: Execution plan, expected artifact counts, storage
  estimate.
- **Figures or plots**: Optional runtime/storage summary.
- **Gallery executable**: No.

Developer-Level Extensions
--------------------------


- **Description**: Add workflow-compatible solvers, metrics, or plotting tools
  without breaking existing pipelines.
- **Target audience**: Package contributors.
- **Difficulty**: Advanced.
- **Estimated duration**: 90 minutes.
- **Prerequisites**: Custom Uncertainty Estimators and Custom Calibration
  Metrics.
- **Main APIs introduced**: Solver registry in
  ``calibrain.workflows.data_generation``, public API exports,
  ``plot_paper_calibration_figures``.
- **Scientific concepts**: API contracts, schema evolution, reproducible
  extension points.
- **Expected outputs**: New extension with tests or smoke examples.
- **Figures or plots**: Only if extending visualization.
- **Gallery executable**: No; developer tutorial.
