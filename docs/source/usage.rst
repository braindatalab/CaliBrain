Usage
=====

This section explains how to use **CaliBrain** for simulating data, running experiments, and analyzing uncertainty calibration.

Leadfield Simulation
---------------------

To simulate or load a leadfield matrix:

.. code-block:: bash

   python examples/leadfield_simulation_example.py --config configs/leadfield_sim_cfg.yml --log-level INFO

**Arguments:**
- ``--config``: Path to the YAML configuration file (required).
- ``--log-level``: Logging level (optional: DEBUG, INFO, WARNING, etc.).

For the default configuration, simply run:

.. code-block:: bash

   python examples/leadfield_simulation_example.py

**Configuration Options (`configs/leadfield_sim_cfg.yml`):**
- **Data paths**: Set paths for data storage and subject directories.
- **Source Space**: Define source space (default: ``ico4`` spacing, ``white`` surface).
- **BEM Model**: Specify three-layer head model (brain, skull, scalp).
- **Montage**: Set EEG sensor positions (e.g., ``easycap-M43`` or custom .fif).
- **Info**: Define EEG/MEG measurement info (e.g., ``sfreq=100 Hz``, EEG modality).
- **Forward Solution**: Define dipole orientations (fixed or free).
- **Leadfield Matrix**: Extract and save for efficient simulation.

**Generated Files:**
- ``{subject}-src.fif`` (Source Space)
- ``{subject}-bem.fif`` (BEM Model)
- ``{subject}-montage.fif`` (Montage)
- ``{subject}-info.fif`` (Measurement Info)
- ``{subject}-fixed-fwd.fif`` or ``{subject}-free-fwd.fif`` (Forward Solution)
- ``{subject}-leadfield-fixed.npz`` or ``{subject}-leadfield-free.npz`` (Leadfield Matrix)

---

Running Experiments
--------------------

To benchmark uncertainty estimation and calibration:

.. code-block:: bash

   python examples/run_experiments.py --config configs/experiment_cfg.yml --log-level INFO

**Workflow Overview:**
- **Step 1: Leadfield Setup**: Simulate leadfield matrix if not available.
- **Step 2: Data Simulation**: Generate synthetic brain activity, project to sensors, add noise.
- **Step 3: Source Estimation**: Solve inverse problem (Gamma-MAP, eLORETA).
- **Step 4: Calibration Evaluation**: Assess confidence intervals or activation probabilities.
- **Step 5: Save Results**: Store figures, calibration plots, and metrics in ``results/`` directory.

**Experiment Configuration Example:**

.. code-block:: yaml

   data_param_grid:
     n_times: [2]
     nnz: [5]
     orientation_type: ["fixed"]
     alpha_snr: [0.9]

   gamma_map_params:
     gammas: [0.001]
     noise_type: ["oracle"]

**Customization Options**:
- Modify data simulation parameters
- Adjust inverse solver settings
- Configure noise structure
- Set benchmark repetitions

---

API Overview
-------------

Core components of **CaliBrain**:
- **LeadfieldSimulator**: Setup and simulate leadfields.
- **DataSimulator**: Simulate synthetic brain activity with noise control.
- **SourceEstimator**: Solve inverse problems using various solvers.
- **UncertaintyEstimator**: Analyze confidence intervals and calibration.
- **Benchmark**: Automate experiment running and results collection.

**Tip**: All modules support YAML-based configuration for reproducibility.

---

Examples
--------

Examples are located in the ``examples/`` folder:
- ``leadfield_simulation_example.py``: Simulate or load leadfields.
- ``run_experiments.py``: Run benchmark experiments for calibration.

Example: Leadfield Simulation

.. code-block:: python

   from calibrain import LeadfieldSimulator
   from calibrain.utils import load_config

   config = load_config("configs/leadfield_sim_cfg.yml")
   simulator = LeadfieldSimulator(config=config)
   leadfield = simulator.simulate()

Example: Benchmark Experiments

.. code-block:: python

   from calibrain import DataSimulator, Benchmark, gamma_map

   data_simulator = DataSimulator(leadfield_mode="simulate", leadfield_config_path="configs/leadfield_sim_cfg.yml")
   benchmark = Benchmark(solver=gamma_map, solver_param_grid=..., data_param_grid=..., data_simulator=data_simulator, metrics=[...])
   results = benchmark.run(nruns=1)

---

Summary
-------

After installing **CaliBrain**:

1. Simulate or load a leadfield matrix.
2. Simulate synthetic EEG/MEG measurements.
3. Solve the inverse problem to reconstruct sources.
4. Quantify uncertainty and visualize calibration results.