Results
================================

The ``results/`` directory organizes all outputs generated during data simulation, forward modeling, benchmarking, and uncertainty analysis. The structure is as follows:

1. ``benchmark_results/``
--------------------------

- Stores the benchmark performance results as ``.csv`` files.

- **Files:**
  - Multiple CSV files (e.g., ``benchmark_results_20250428_185434.csv``) capturing evaluation metrics from different benchmarking runs.

2. ``figures/``
---------------

- Contains generated figures from simulations and analyses.

- **Subfolders:**

  - ``uncertainty_analysis_figures/``: Visualizations related to uncertainty estimation.

    - **Sub-structure:**
      - Organized hierarchically based on:
        - Estimator type (e.g., ``gamma_map``)
        - Gamma regularization value (e.g., ``gammas=0.001``)
        - Noise type (e.g., ``noise_type=oracle``)
        - Alpha signal-to-noise ratio (e.g., ``alpha_snr=0.9``)
        - Number of time points (e.g., ``n_times=2``)
        - Number of non-zero sources (e.g., ``nnz=5``)
        - Source orientation type (either ``fixed`` or ``free``)

    - **Contents inside each orientation type:**
      - ``sorted_covariances.png``, ``sorted_variances.png``: Sorted values of posterior variances and covariances.
      - ``CI/``: Confidence interval plots at various confidence levels (``clvl``) for each active source (``t0``, ``t1``).
      - ``posterior_covariance_matrix.png``: Visualization of the posterior covariance matrix.
      - ``proportion_of_hits.png``: How often ground truth sources fall within estimated confidence intervals.
      - ``active_sources_single_time_step_0.png``: Activation patterns at a specific time step.

  - ``data_sim/``: Figures related to synthetic data generation.

    - **Files:**
      - ``pre_post_stimulus_active_sources_subplots.png``: Active sources before and after a stimulus.
      - ``pre_post_stimulus_specific_sensor_signals.png``: Sensor signal examples around stimulus onset.
      - ``data_sim.png``: Overall simulation overview.
      - ``leadfield_topomap.png``: Topography visualization of the leadfield.
      - ``leadfield.png``: Leadfield matrix visualization.


3. ``forward/``
---------------

- Stores precomputed forward models and leadfields for source localization.

- **Files:**
  - ``fsaverage-fixed-fwd.fif``, ``fsaverage-free-fwd.fif``: Forward solutions for fixed and free orientations.
  - ``fsaverage-leadfield-fixed.npz``, ``fsaverage-leadfield-free.npz``: Saved leadfield matrices.
  - ``fsaverage-bem.fif``: Boundary Element Method (BEM) model.
  - ``fsaverage-src.fif``: Source space definition.
  - ``fsaverage-montage.fif``: Electrode montage.
  - ``fsaverage-info.fif``: Measurement info.


General Notes
=============

- The deep hierarchical organization under ``figures/uncertainty_analysis_figures/`` reflects parameter sweep experiments across different estimation settings (e.g., gamma, noise type).

- Figures and data are organized to facilitate easy retrieval and comparison between different modeling and analysis settings.