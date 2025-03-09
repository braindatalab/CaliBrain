# CaliBrain

## Uncertainty Calibration in Brain Source Imaging

<img src="/docs/images/caliBrain.jpeg" alt="CaliBrain Logo" width="25%">

### Overview
This framework focuses on uncertainty estimation and calibration in EEG/MEG inverse source imaging. It evaluates how well estimated source activations reflect the true uncertainty in reconstructed neural currents. We assess uncertainty calibration for both regression (continuous current estimates) and classification (binary activation detection) tasks.

The pipeline includes simulating Sparse Source Time Courses, setting up the source space using the fsaverage template, creating forward solutions, and solving the inverse problem to reconstruct the source time courses. The methods implemented are applied to source imaging solutions obtained using the Gamma-MAP, eLORETA, and Bayesian Minimum Norm inverse methods.

The goal is to ensure that confidence intervals (CIs) and probabilistic activation estimates are well-calibrated, meaning they accurately reflect the true underlying uncertainty in source reconstructions.

---

### Problem Statement

Inverse source imaging methods provide estimates of neural activity, but these estimates come with uncertainty. Properly quantifying and calibrating this uncertainty is crucial for reliable interpretation.

#### 1. Regression Case: Confidence Interval Calibration
- The goal is to check how often the true simulated source current is contained within the estimated confidence intervals (CIs).
- We compute posterior mean and variance from the inverse method and assess the fraction of cases where the true current falls within different CI levels (e.g., 10%, 20%, …, 90%).
- The calibration curve plots the expected vs. observed coverage of true currents within CIs. A well-calibrated model should follow the diagonal.

#### 2. Classification Case: Probabilistic Activation Calibration
- The goal is to assess how well the estimated probability of activation reflects the true activation of sources.
- Each voxel is classified as active (nonzero current) or inactive (zero current) using ground truth.
- We bin voxels based on their estimated probability of activation (e.g., 0–10%, 10–20%, …) and check how often they are actually active in the ground truth.
- The calibration curve compares the estimated probability of activation with the actual observed activation. A well-calibrated model should align with the diagonal.

---

#### Parameters

- **Estimator Name:** Gamma-MAP, eLORETA, Bayesian Minimum Norm
- **Orientation Type:** Fixed, Free
- **Covariance Type:** Scaled Identity, Cross Validation, Joint Learning
- **Alpha SNR:** Signal-to-noise ratio parameter for regularization.
- **Number of Simulated Non-Zero Sources:** Number of active sources in the simulated data.
- **Noise Type:** Type of noise model used in the simulations. Options: Scaled Identity, Cross Validation, Joint Learning.

<img src="/docs/images/un-ca-param.jpg" alt="un-ca-param" width="75%">

---

### Outcome

- A calibration curve for regression, showing how well estimated confidence intervals capture the true current.
- A calibration curve for classification, indicating how well the estimated probability of activation aligns with the actual activation.
- A quantitative metric summarizing calibration quality.

A well-calibrated model should produce curves that closely match the diagonal line, indicating that estimated uncertainty matches observed uncertainty.

---


### Installation

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

### Usage

To use the BSI Zoo tools, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/braindatalab/CaliBrain.git
    ```
2. Navigate to the project directory:
    ```bash
    cd CaliBrain
    ```
3. Run the main script:
    ```bash
    python main.py
    ```

---

### Contributing

We welcome contributions! Please read our [contributing guidelines](CONTRIBUTING.md) for more information.

---

### Citations

If you use this project, please cite relevant works on EEG/MEG inverse modeling and uncertainty quantification.

---

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
