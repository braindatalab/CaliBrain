Inverse Modeling
================


These tutorials explain how source reconstruction outputs are produced and
prepared for calibration.


Running a Basic Source Reconstruction Workflow
----------------------------------------------


- **Description**: Run one solver on simulated sensor data and inspect posterior
  mean and covariance outputs.
- **Target audience**: Users learning solver outputs.
- **Difficulty**: Beginner to intermediate.
- **Estimated duration**: 30 minutes.
- **Prerequisites**: Generating Sensor-Level Data.
- **Main APIs introduced**: ``gamma_map``, ``SourceEstimator``.
- **Scientific concepts**: Posterior mean, posterior covariance, active source
  indices.
- **Expected outputs**: ``posterior_mean``, ``posterior_cov``, active-source
  metadata.
- **Figures or plots**: True versus reconstructed source traces.
- **Gallery executable**: Yes.

Comparing Inverse Solvers
-------------------------


- **Description**: Compare supported solvers on the same simulated data and
  inspect reconstruction and uncertainty differences.
- **Target audience**: Users selecting reconstruction methods.
- **Difficulty**: Intermediate.
- **Estimated duration**: 45 minutes.
- **Prerequisites**: Running a Basic Source Reconstruction Workflow.
- **Main APIs introduced**: ``gamma_map``, ``gamma_map_sflex``,
  ``gamma_lambda_map_sflex``, ``eloreta``, ``BMN``, ``BMN_joint``.
- **Scientific concepts**: Sparse Bayesian estimation, eLORETA-style
  estimation, BMN normalization, solver-dependent uncertainty.
- **Expected outputs**: Solver comparison table and posterior summaries.
- **Figures or plots**: Reconstruction error and uncertainty comparison.
- **Gallery executable**: Yes for a small fixed-orientation synthetic case.

Handling Trial-Level and Source-Level Outputs
---------------------------------------------


- **Description**: Explain array shapes for source time courses, posterior
  means, covariance matrices, and reduced free-orientation components.
- **Target audience**: Users debugging workflow artifacts.
- **Difficulty**: Intermediate.
- **Estimated duration**: 25 minutes.
- **Prerequisites**: Understanding CaliBrain Data Structures.
- **Main APIs introduced**: ``load_posterior_summary``,
  ``lift_reduced_sources_to_3d``.
- **Scientific concepts**: Source-level aggregation, time averaging, reduced
  MEG tangent bases.
- **Expected outputs**: Shape summary table for fixed and free orientation.
- **Figures or plots**: Optional shape/axis schematic.
- **Gallery executable**: Yes.

Preparing Predictions for Calibration
-------------------------------------


- **Description**: Convert raw solver outputs into the uncertainty fields
  consumed by calibration.
- **Target audience**: Users integrating custom solvers.
- **Difficulty**: Intermediate.
- **Estimated duration**: 30 minutes.
- **Prerequisites**: Handling Trial-Level and Source-Level Outputs.
- **Main APIs introduced**: ``aggregate_posteriors``,
  ``concatenate_summaries``, ``filter_summaries_by_metadata``.
- **Scientific concepts**: Raw covariance versus reduced calibration
  representation, manifest filtering.
- **Expected outputs**: Aggregated NPZ with ``posterior_var`` or
  ``posterior_cov_blocks``.
- **Figures or plots**: None required; schema tables preferred.
- **Gallery executable**: Yes with tiny generated summaries.
