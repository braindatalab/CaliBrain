Usage
=====

This section explains how to use **CaliBrain** for simulating data, running experiments, and analyzing uncertainty calibration.

---

What can I do with **CaliBrain**?
---------------------------------

1. Simulate or load a leadfield matrix.
2. Simulate synthetic dipole activity.
3. Simulate sensor-level measurements.
4. Solve the inverse problem to reconstruct sources.
5. Quantify uncertainty and visualize calibration results.

---

Running Experiments
--------------------

To run the entire benchmark with default settings:

.. code-block:: bash

   python examples/run_experiments.py


**Experiment Configuration Example:**

.. code-block:: python

    data_param_grid_meg = {
        "subject": ["CC120166"], # "CC120166", "CC120264", "CC120309", "CC120313",
        "nnz": [1, 10, 100],
        "orientation_type": ["fixed"], # "fixed", "free"
        "alpha_SNR": [0.0, 0.3, 0.5, 0.7, 0.99],
    }
    
    data_param_grid_eeg = {
        "subject": ["fsaverage"], # "caliBrain_fsaverage", "fsaverage",
        "nnz": [1, 10, 100],
        "orientation_type": ["fixed"], # "fixed", "free"
        "alpha_SNR": [0.0, 0.3, 0.5, 0.7, 0.99],
    }
        
    gamma_map_params = {
        "init_gamma": [0.001], #  0.001, 1.0, or tuple for random values (0.001, 0.1)   
        "noise_type": ["oracle"], # "baseline", "oracle", "joint_learning", "CV"
    }
    
    eloreta_params = {
        "noise_type": ["oracle"],
    }
    
    estimators = [
        (gamma_map, gamma_map_params, data_param_grid_meg),
        (eloreta, eloreta_params, data_param_grid_meg),
        (gamma_map, gamma_map_params, data_param_grid_eeg),
        (eloreta, eloreta_params, data_param_grid_eeg),
    ]

---

API Overview
-------------

Core components of **CaliBrain**:
- **SourceSimulator**: Simulate source activity.
- **SensorSimulator**: Simulate sensor-level measurements.
- **LeadfieldBuilder**: Setup and simulate leadfields.
- **SourceEstimator**: Solve inverse problems (e.g., Gamma-MAP, eLORETA).
- **UncertaintyEstimator**: Analyze confidence intervals and calibration.
- **MetricEvaluator**: Compute evaluation metrics for model performance.
- **Visualizer**: Plot calibration results, source estimates, and uncertainty.
- **Benchmark**: Automate experiment running and results collection.
---

Examples
--------

Examples are located in the ``examples/`` folder:
- ``leadfield_builder_example.py``: Simulate or load leadfields.
- ``run_experiments.py``: Run benchmark experiments for calibration.


Leadfield Simulation
---------------------

To simulate or load a leadfield matrix:

.. code-block:: bash

   python examples/run_experiments.py --config configs/leadfield_sim_cfg.yml --log-level INFO

**Arguments:**
- ``--config``: Path to the YAML configuration file (required).
- ``--log-level``: Logging level (optional: DEBUG, INFO, WARNING, etc.).

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

Example: Leadfield Simulation

.. code-block:: python

   from calibrain import LeadfieldBuilder
   from calibrain.utils import load_config

   config = load_config("configs/leadfield_sim_cfg.yml")
   simulator = LeadfieldBuilder(config=config)
   leadfield = simulator.simulate()
