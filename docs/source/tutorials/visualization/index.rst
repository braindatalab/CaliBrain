Visualization
=============


Visualization tutorials focus on interpretation and figure generation from
current data structures.


Plotting Source and Sensor Signals
----------------------------------


- **Description**: Plot simulated source traces and sensor-level observations.
- **Target audience**: Users validating simulation outputs.
- **Difficulty**: Beginner.
- **Estimated duration**: 20 minutes.
- **Prerequisites**: Simulating Source Activity.
- **Main APIs introduced**: ``Visualizer.plot_source_signals``,
  ``Visualizer.plot_sensor_signals``.
- **Scientific concepts**: ERP timing, active sources, sensor projection.
- **Expected outputs**: Source and sensor signal figures.
- **Figures or plots**: Time-series plots.
- **Gallery executable**: Yes.

Plotting Uncertainty Maps
-------------------------


- **Description**: Plot source-level uncertainty summaries from posterior
  variance or covariance blocks.
- **Target audience**: Users interpreting posterior uncertainty.
- **Difficulty**: Intermediate.
- **Estimated duration**: 30 minutes.
- **Prerequisites**: Interpreting Uncertainty Maps.
- **Main APIs introduced**: ``Visualizer``, ``load_posterior_summary``.
- **Scientific concepts**: Posterior variance maps, active-source uncertainty.
- **Expected outputs**: Uncertainty summary arrays and figures.
- **Figures or plots**: Variance or interval-width maps.
- **Gallery executable**: Yes with lightweight coordinates.

Visualizing Calibration Curves
------------------------------


- **Description**: Load calibration JSON and plot pre- and post-calibration
  curves.
- **Target audience**: Users reading workflow outputs.
- **Difficulty**: Beginner to intermediate.
- **Estimated duration**: 25 minutes.
- **Prerequisites**: Computing Calibration Curves.
- **Main APIs introduced**: ``load_calibration_record``,
  ``Visualizer.plot_pre_post_calibration_curves``.
- **Scientific concepts**: Calibration curve interpretation, repeated
  recalibrated nominal levels, boundary clamping.
- **Expected outputs**: Calibration curve figure.
- **Figures or plots**: Single-run pre/post curve.
- **Gallery executable**: Yes.

Visualizing Benchmark Results
-----------------------------


- **Description**: Aggregate calibration JSON files across runs and visualize
  method-level mean and dispersion.
- **Target audience**: Users comparing benchmark conditions.
- **Difficulty**: Advanced.
- **Estimated duration**: 45 minutes.
- **Prerequisites**: Comparing Calibration Across Methods.
- **Main APIs introduced**: ``stack_empirical_curves``,
  ``plot_paper_calibration_figures.main``.
- **Scientific concepts**: Across-run aggregation, paired fixed/free
  comparisons, uncertainty in empirical curves.
- **Expected outputs**: Grouped calibration statistics.
- **Figures or plots**: Mean ± standard deviation curves.
- **Gallery executable**: No for full paper outputs; yes for reduced JSON
  fixtures.

Creating Publication-Ready Figures
----------------------------------


- **Description**: Produce paper-style calibration panels from organized JSON
  outputs.
- **Target audience**: Paper authors and contributors.
- **Difficulty**: Advanced.
- **Estimated duration**: 60 minutes.
- **Prerequisites**: Visualizing Benchmark Results.
- **Main APIs introduced**: ``calibrain.workflows.plot_paper_calibration_figures``.
- **Scientific concepts**: Consistent styling, fixed/free pairing, grouped
  legends, reproducible figure paths.
- **Expected outputs**: Publication-style PNG figures.
- **Figures or plots**: Multi-panel calibration figures.
- **Gallery executable**: No; workflow/documentation tutorial.
