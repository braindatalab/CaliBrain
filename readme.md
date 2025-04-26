# CaliBrain

## Uncertainty Calibration in Brain Source Imaging

<img src="/docs/images/caliBrain.jpeg" alt="CaliBrain Logo" width="25%">

---

### Overview

CaliBrain is a framework for uncertainty estimation and calibration in EEG/MEG inverse source imaging. It evaluates how well estimated source activations reflect the true uncertainty in reconstructed neural currents. The framework supports both regression (continuous current estimates) and classification (binary activation detection) tasks.

The pipeline includes:
- Simulating sparse source time courses.
- Setting up the source space and creating forward solutions.
- Solving the inverse problem to reconstruct the source time courses.

The methods implemented are applied to source imaging solutions obtained using:
- Gamma-MAP
- eLORETA
- Bayesian Minimum Norm inverse methods

The goal is to ensure that confidence intervals (CIs) and probabilistic activation estimates are well-calibrated, meaning they accurately reflect the true underlying uncertainty in source reconstructions.

---

### Problem Statement

Inverse source imaging methods provide estimates of neural activity, but these estimates come with uncertainty. Properly quantifying and calibrating this uncertainty is crucial for reliable interpretation.

#### 1. Regression Case: Confidence Interval Calibration
- The goal is to check how often the true simulated source current is contained within the estimated confidence intervals (CIs).
- Posterior mean and variance are computed from the inverse method, and the fraction of cases where the true current falls within different CI levels (e.g., 10%, 20%, …, 90%) is assessed.
- The calibration curve plots the expected vs. observed coverage of true currents within CIs. A well-calibrated model should follow the diagonal.

#### 2. Classification Case: Probabilistic Activation Calibration
- The goal is to assess how well the estimated probability of activation reflects the true activation of sources.
- Each voxel is classified as active (nonzero current) or inactive (zero current) using ground truth.
- Voxels are binned based on their estimated probability of activation (e.g., 0–10%, 10–20%, …), and the actual activation frequency is compared.
- The calibration curve compares the estimated probability of activation with the actual observed activation. A well-calibrated model should align with the diagonal.

---

### Parameters

The following parameters are configurable in the framework:

- **Estimator Name**: Gamma-MAP, eLORETA, Bayesian Minimum Norm
- **Orientation Type**: Fixed, Free
- **Covariance Type**: Scaled Identity, Cross Validation, Joint Learning
- **Alpha SNR**: Signal-to-noise ratio parameter for regularization
- **Number of Simulated Non-Zero Sources**: Number of active sources in the simulated data
- **Noise Type**: Type of noise model used in the simulations (e.g., Scaled Identity, Cross Validation, Joint Learning)

---

### Outcome

The framework produces:
1. **Regression Calibration Curve**:
   - Shows how well estimated confidence intervals capture the true current.
   - A well-calibrated model should produce a curve that closely matches the diagonal line.

2. **Classification Calibration Curve**:
   - Indicates how well the estimated probability of activation aligns with the actual activation.
   - A well-calibrated model should align with the diagonal.

3. **Quantitative Metrics**:
   - Summarize the calibration quality for both regression and classification tasks.

---

### Installation

#### 1. Install via `pyproject.toml`
If you prefer to use `pyproject.toml` for dependency management, you can install the package using `pip`:

1. Install the package:
    ```bash
    pip install .
    ```

2. Or install in editable mode for development:
    ```bash
    pip install -e .
    ```

#### 2.  Optional: Install via Conda
CaliBrain can be installed using Conda for dependency management. Follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/braindatalab/CaliBrain.git
    ```

2. Navigate to the project directory:
    ```bash
    cd CaliBrain
    ```

3. Create a Conda environment:
    ```bash
    conda create -n calibrain python=3.9 -y
    conda activate calibrain
    ```

4. Install dependencies:
    ```bash
    conda install --file requirements.txt
    ```

5. Install the package:
    ```bash
    pip install .
    ```

6. Alternatively, install in editable mode for development:
    ```bash
    pip install -e .
    ```
---

### Usage
The `examples` folder contains scripts to demonstrate how to use the CaliBrain framework for various tasks. 

Below are the key scripts and their usage:
#### 1. Leadfield Simulation
The `leadfield_simulation_example.py` script generates a leadfield matrix for free or fixed orientation based on a configuration file.

**Command**
```bash
python examples/leadfield_simulation_example.py --config <path_to_config_file> --log-level <log_level>
```

**Arguments:**

`--config`: Path to the YAML configuration file for leadfield simulation (required).

`--log-level`: Logging level (optional). Options: `NOTSET`, `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.

**Example:**
From the Command Line (Default Behavior)

```bash
python examples/run_experiments.py --config configs/experiment_cfg.yml --log-level INFO
```

Or programmatically (Inside the Script)
You can call the main function with a list of arguments directly:

```python
if __name__ == "__main__":
    main([
        "--config", "examples/dipole_sim_cfg.yml",
        "--log-level", "DEBUG" # or INFO
    ])
```

**Configuration Files**
The configuration file `leadfield_sim_cfg.yml` define the parameters for the simulation or loading all the forward operator steps:
- handling the source space
- handling the bem model
- handling the forward solution
- handling the leadfield

#### 2. Running Experiments
The `run_experiments.py` script runs benchmark experiments for uncertainty estimation and calibration.
Note: the script automatically detects if a leadfield matrix has already been generated. If not, it will generate the leadfield matrix based on the same configuration file.

### How to Run the Code


---

### Contributing

We welcome contributions! Please read our [contributing guidelines](CONTRIBUTING.md) for more information.

---

### Citations

If you use this project, please cite relevant works on EEG/MEG inverse modeling and uncertainty quantification.

---

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
