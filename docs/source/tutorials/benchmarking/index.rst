Benchmarking
============


Benchmark tutorials scale the lightweight concepts to reproducible experiment
grids.


Running a Standard Benchmark
----------------------------


- **Description**: Run a reduced benchmark through data generation,
  aggregation, and calibration.
- **Target audience**: Users validating the full pipeline.
- **Difficulty**: Intermediate.
- **Estimated duration**: 60--90 minutes for reduced settings.
- **Prerequisites**: First End-to-End Calibration Workflow.
- **Main APIs introduced**: ``run_data_generation``, ``aggregate_posteriors``,
  ``run_calibration``.
- **Scientific concepts**: Benchmark split design, reproducibility, storage
  budgeting.
- **Expected outputs**: H5 summaries, manifest CSV, aggregated NPZ files,
  calibration JSON files.
- **Figures or plots**: Reduced benchmark calibration curves.
- **Gallery executable**: No by default; reduced smoke version can be executable.

Comparing Solvers Across Noise Levels
-------------------------------------


- **Description**: Compare solver/noise combinations such as BMN oracle,
  baseline, adaptive noise, and sFLEX variants.
- **Target audience**: Users analyzing solver robustness.
- **Difficulty**: Advanced.
- **Estimated duration**: 1--2 hours for reduced settings.
- **Prerequisites**: Comparing Inverse Solvers.
- **Main APIs introduced**: Solver registry in ``run_data_generation``,
  ``MetricEvaluator``, ``run_calibration``.
- **Scientific concepts**: Noise mismatch, adaptive noise learning,
  calibration robustness.
- **Expected outputs**: Per-solver calibration JSON directories.
- **Figures or plots**: Solver-by-noise calibration comparison.
- **Gallery executable**: No for full grid; yes for a tiny comparison fixture.

Evaluating Multiple Random Seeds
--------------------------------


- **Description**: Run and summarize multiple seeds/runs to estimate variability
  of calibration curves.
- **Target audience**: Users designing reliable simulation studies.
- **Difficulty**: Intermediate.
- **Estimated duration**: 45--90 minutes for reduced grids.
- **Prerequisites**: Running a Standard Benchmark.
- **Main APIs introduced**: ``DataGenerator.run``, manifest metadata,
  ``stack_empirical_curves``.
- **Scientific concepts**: Monte Carlo variability, run IDs, seed control.
- **Expected outputs**: Multiple posterior summaries and calibration JSONs.
- **Figures or plots**: Mean ± standard deviation curves across seeds.
- **Gallery executable**: Conditional; small seed counts only.

Aggregating Results
-------------------


- **Description**: Aggregate posterior summaries into calibration-ready NPZ
  datasets and aggregate calibration JSONs into summary curves.
- **Target audience**: Users managing benchmark artifacts.
- **Difficulty**: Intermediate.
- **Estimated duration**: 30 minutes after generation.
- **Prerequisites**: Understanding CaliBrain Data Structures.
- **Main APIs introduced**: ``aggregate_posteriors``,
  ``load_manifest_csv``, ``stack_empirical_curves``.
- **Scientific concepts**: Manifest filtering, split definitions, reduced
  uncertainty storage.
- **Expected outputs**: Aggregated NPZ files, JSON sidecars, stacked empirical
  coverage arrays.
- **Figures or plots**: Split-size summary and aggregated curves.
- **Gallery executable**: Yes with tiny fixtures.

Reproducing Paper Experiments
-----------------------------


- **Description**: Use the default config files to reproduce paper-style
  fixed/free calibration experiments.
- **Target audience**: Authors, reviewers, advanced contributors.
- **Difficulty**: Advanced.
- **Estimated duration**: Hours to days depending on grid and hardware.
- **Prerequisites**: Running a Standard Benchmark and access to full data.
- **Main APIs introduced**: ``configs/data_generation_default.py``,
  ``configs/aggregate_default.py``, ``configs/calibration_default.py``,
  ``plot_paper_calibration_figures.main``.
- **Scientific concepts**: Default SNR/NNZ setting, SNR and NNZ sweeps,
  matched and mismatched calibration, fixed/free comparisons.
- **Expected outputs**: Full result tree under the configured results root and
  publication-style figures.
- **Figures or plots**: Figure 2, Figure S1, Figure S2 style outputs.
- **Gallery executable**: No; large workflow tutorial.
