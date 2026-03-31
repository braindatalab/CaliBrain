<!-- TODO: Convert this to readme.rst format: -->
# CaliBrain

## Uncertainty Calibration in Brain Source Imaging

<img src="/docs/source/_static/caliBrain.png" alt="CaliBrain Logo" width="25%">

<!-- [![Latest Release](https://img.shields.io/github/v/release/braindatalab/CaliBrain)](https://github.com/braindatalab/CaliBrain/releases/latest) -->
<!-- [![PyPI version](https://img.shields.io/pypi/v/calibrain.svg)](https://pypi.org/project/calibrain/) -->
<!-- [![Python Version](https://img.shields.io/pypi/pyversions/calibrain.svg)](https://pypi.org/project/calibrain/) -->

[![commits](https://badgen.net/github/commits/braindatalab/CaliBrain/main)](https://github.com/braindatalab/CaliBrain/commits/main?icon=github&color=green)
[![Documentation (latest)](https://readthedocs.org/projects/calibrain/badge/?version=latest)](https://calibrain.readthedocs.io/en/latest/?badge=latest)
<!-- [![Documentation (stable)](https://readthedocs.org/projects/calibrain/badge/?version=stable)](https://calibrain.readthedocs.io/en/stable/?badge=stable) -->


---

## Overview

**CaliBrain** is a Python framework for uncertainty estimation and calibration in EEG/MEG inverse source imaging.

It supports both:
- **Regression** (continuous source estimates)
- **Classification** (binary activation detection)

**Key Features**:
- Setup of source space, BEM model, forward solution, and leadfield matrices.
- Simulation of source activity and sensor-level measurements with controllable noise and source orientation (fixed or free).
- Solving the inverse problem and reconstructing source time courses.
- Estimation and visualization of confidence intervals.
- Calibration analysis by comparing expected vs. observed confidence levels.

### Supported Inverse Methods
- Gamma-MAP
- eLORETA
- Bayesian Minimum Norm

---

## Calibration Tasks

### 1. Regression (Confidence Interval Calibration)
- Check if true simulated source currents fall within predicted confidence intervals.
- Plot calibration curve (Expected vs. Observed coverage).
- Well-calibrated models should follow the diagonal.

### 2. Classification (Activation Calibration)
- Assess if estimated activation probabilities match true activation frequencies.
- Plot calibration curve for activation detection.
- Ideal calibration follows the diagonal.

---

## Main Parameters

- **Estimator**: Gamma-MAP, eLORETA, Bayesian Minimum Norm
- **Orientation**: Fixed or Free
- **Noise Type**: Oracle, Baseline, Cross-Validation, Joint Learning
- **SNR Level (α)**: Control regularization strength
- **Active Sources (nnz)**: Number of nonzero sources

<img src="/docs/images/un-ca-param.jpg" alt="un-ca-param" width="75%">

---

## Outcomes

- **Regression Calibration Curves** (confidence intervals)
- **Classification Calibration Curves** (activation probabilities)
- **Quantitative Calibration Metrics**

---

## Installation

Please see the [Installation Guide](docs/source/installation.rst).

---

## Usage

Please see the [Usage Guide](docs/source/usage.rst).

---

## Experiment-Level Calibration Workflow

To recalibrate uncertainty estimates using whole experiments instead of per-source cross-validation:

1. **Pick a workflow-specific entry point**  
   - `python calibrain/workflows/data_generation.py --config configs/data_generation_default.py`  
   - `python calibrain/workflows/aggregation.py --config configs/aggregate_default.py`  
   - `python calibrain/workflows/calibration.py --config configs/calibration_default.py`

2. **Edit the Python configs**  
   Each config file defines a `CONFIG` dictionary describing the inputs for its workflow (solver grids, metadata filters, dataset paths, etc.). Copy or modify the `.py` files under `configs/` to capture different experiment setups; you can add comments or share common structures inside the module.

3. **Typical sequence**
   - Run the data-generation module to produce simulations (each run stores a single `posterior_summary_*.h5` file containing arrays + metadata).
   - Adjust the aggregation config module (e.g., `configs/aggregate_default.py`) to define metadata filters, then run the aggregation module to produce `_train.npz` / `_test.npz`.
   - Update the calibration config module (e.g., `configs/calibration_default.py`) with the aggregated dataset paths and run the calibration module to get isotonic calibration metrics/plots.

### Metadata keys available for filtering

- `solver`
- `noise_type`
- `subject`
- `orientation_type`
- `nnz`
- `alpha_SNR`
- `seed`
- `run_id` / `global_run_id`
- `nruns`

Combine these to form arbitrary splits (e.g., leave-one-subject-out, different noise regimes, etc.) before running the new post-calibration pipeline.

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/source/contributing.rst).

---

## License

This project is licensed under the GNU Affero General Public License v3.0. See [LICENSE](LICENSE).

---

## Citation

If you use CaliBrain, please cite relevant works in EEG/MEG source imaging and uncertainty quantification.
