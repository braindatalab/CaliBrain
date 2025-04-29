Convert this to readme.rst format:
# CaliBrain

## Uncertainty Calibration in Brain Source Imaging

<img src="/docs/source/_static/caliBrain.png" alt="CaliBrain Logo" width="25%">

---

## Overview

**CaliBrain** is a Python framework for uncertainty estimation and calibration in EEG/MEG inverse source imaging.

It supports both:
- **Regression tasks** (continuous source estimates)
- **Classification tasks** (binary activation detection)

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

Please see the [Installation Guide](docs/installation.rst).

---

## Usage

Please see the [Usage Guide](docs/source/usage.rst).

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/source/contributing.rst).

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

---

## Citation

If you use CaliBrain, please cite relevant works in EEG/MEG source imaging and uncertainty quantification.