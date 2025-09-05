.. _results:

Results
================================

The ``results/`` directory organizes all outputs generated during data simulation, forward modeling, benchmarking, and uncertainty analysis. The structure is as follows:

1. ``benchmark_results/``
--------------------------

Stores the benchmark performance results as ``.csv`` files.
- ``benchmark_results/benchmark_log_TIMESTAMP.txt``: Log file for a specific benchmarking run.

2. ``logs/``
------------

- ``logs/benchmark_log_TIMESTAMP.txt``: Log file for a specific benchmarking run.

3. ``figures/``
---------------

- Contains generated figures from simulations and analyses.

- **Subfolders:**

  - ``uncertainty_analysis_figures/``: Visualizations related to uncertainty estimation.

    - **Sub-structure:**
      - Organized hierarchically based on:
      - Subject (e.g., ``CC120166``)
      - Solver type (e.g., ``gamma_map``)
      - Source orientation type (either ``fixed`` or ``free``)
      - Alpha signal-to-noise ratio (e.g., ``alpha_snr=0.9``)
      - Noise type (e.g., ``noise_type=oracle``)
      - Number of non-zero sources (e.g., ``nnz=5``)
      - Seed values (e.g., ``seed=42``)

    - **Contents inside each seed value:**
      - ``data_simulation/``: Figures related to synthetic data generation.
      - ``uncertainty_analysis/``: Figures related to uncertainty estimation.

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