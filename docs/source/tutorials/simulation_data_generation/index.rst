Simulation and Data Generation
==============================


These tutorials explain how synthetic source activity and sensor observations
are produced and how benchmark-ready posterior summaries are written.


Simulating Source Activity
--------------------------


- **Description**: Generate sparse fixed-orientation source time courses with
  controlled ERP timing and amplitude.
- **Target audience**: Users building lightweight calibration examples.
- **Difficulty**: Beginner.
- **Estimated duration**: 20 minutes.
- **Prerequisites**: Core Concepts in Uncertainty Calibration.
- **Main APIs introduced**: ``SourceSimulator``.
- **Scientific concepts**: Sparse source support, ERP waveform parameters,
  fixed versus free orientation.
- **Expected outputs**: ``x_true``, active source indices, saved source traces.
- **Figures or plots**: Active source time courses.
- **Gallery executable**: Yes. See
  :doc:`/auto_tutorials/02_source_simulation`.

Generating Sensor-Level Data
----------------------------


- **Description**: Project source activity through a leadfield and inspect clean
  and noisy sensor signals.
- **Target audience**: Users connecting simulation to inverse estimation.
- **Difficulty**: Beginner.
- **Estimated duration**: 20 minutes.
- **Prerequisites**: Simulating Source Activity.
- **Main APIs introduced**: ``SensorSimulator``, ``LeadfieldBuilder``,
  ``Visualizer.plot_sensor_signals``.
- **Scientific concepts**: Leadfield projection, sensor-space observations,
  SNR.
- **Expected outputs**: ``y_clean``, ``y_noisy``, noise arrays.
- **Figures or plots**: Sensor traces before and after noise.
- **Gallery executable**: Yes, using a small synthetic leadfield.

Building and Loading Leadfields
-------------------------------


- **Description**: Demonstrate ``LeadfieldBuilder`` retrieval modes:
  ``random``, ``simulate``, and ``load``.
- **Target audience**: Users preparing source-to-sensor simulations.
- **Difficulty**: Beginner.
- **Estimated duration**: 25 minutes.
- **Prerequisites**: Simulating Source Activity.
- **Main APIs introduced**: ``LeadfieldBuilder``, ``get_data_path``.
- **Scientific concepts**: Leadfield shape, fixed/free orientation,
  source-to-sensor units, local dataset layout.
- **Expected outputs**: Leadfield arrays and metadata.
- **Figures or plots**: Leadfield column-norm quality-control plot.
- **Gallery executable**: Yes. See
  :doc:`/auto_tutorials/03_leadfield_building`.

Adding Noise and Uncertainty
----------------------------


- **Description**: Compare oracle, baseline, and adaptive noise settings as
  inputs to inverse solvers and downstream calibration.
- **Target audience**: Users interpreting solver/noise combinations.
- **Difficulty**: Intermediate.
- **Estimated duration**: 30 minutes.
- **Prerequisites**: Generating Sensor-Level Data.
- **Main APIs introduced**: ``SensorSimulator``, ``gamma_map``, ``BMN``,
  ``BMN_joint``, ``gamma_lambda_map_sflex``.
- **Scientific concepts**: Noise variance, SNR, uncertainty propagation,
  adaptive noise learning.
- **Expected outputs**: Sensor data and solver posterior summaries under
  different noise settings.
- **Figures or plots**: Noise-level comparison and posterior variance summary.
- **Gallery executable**: Yes, if kept synthetic and small.

Creating Benchmark Datasets
---------------------------


- **Description**: Use the data-generation workflow to create H5 posterior
  summaries and a manifest CSV for controlled benchmark splits.
- **Target audience**: Users preparing paper-style experiments.
- **Difficulty**: Intermediate.
- **Estimated duration**: 45--90 minutes for small grids.
- **Prerequisites**: Understanding CaliBrain Data Structures.
- **Main APIs introduced**: ``DataGenerator``, ``run_data_generation``.
- **Scientific concepts**: Parameter grids, random seeds, metadata filtering,
  reproducible simulation design.
- **Expected outputs**: Posterior H5 files, manifest CSV, logs.
- **Figures or plots**: Optional run-count and grid-summary plots.
- **Gallery executable**: No for paper-scale grids; yes for a small smoke grid.
