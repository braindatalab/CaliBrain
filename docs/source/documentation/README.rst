.. _documentation:

Documentation
=============

.. toctree::
   :maxdepth: 2
   :hidden:

   usage
   results
   ../auto_examples/index
   ../auto_tutorials/index


Overview
--------

**CaliBrain** is a Python framework for uncertainty estimation and calibration in EEG/MEG inverse source imaging. It supports both:

- **Regression tasks** (continuous source estimates)
- **Classification tasks** (binary activation detection)

**Key Features**:

- Setup of source space, BEM model, forward solution, and leadfield matrices.
- Simulation of source activity and sensor-level measurements with controllable noise and source orientation (fixed or free).
- Inverse problem solving and reconstruction of source time courses.
- Estimation and visualization of confidence intervals and calibration analysis (expected vs. observed coverage).

Supported Inverse Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Gamma-MAP
- eLORETA
- Bayesian Minimum Norm

Learning Resources
~~~~~~~~~~~~~~~~~~

📚 **New to CaliBrain?** Start with our comprehensive :doc:`tutorials <../auto_tutorials/index>` that cover the complete workflow from forward modeling to uncertainty quantification.

🔬 **Ready to explore?** Check out practical :doc:`examples <../auto_examples/index>` showing real-world applications and complete analysis pipelines.

The tutorials provide step-by-step guidance, while examples demonstrate how to adapt CaliBrain for your specific research needs.

Calibration Tasks
-----------------

1. **Regression Calibration**: 
   - Checks if simulated source currents fall within predicted confidence intervals.
   - Ideal: Coverage follows the diagonal (Expected vs. Observed).
   
2. **Classification Calibration**: 
   - Assesses if activation probabilities match true activation frequencies.
   - Ideal: Calibration follows the diagonal.

Main Parameters
---------------

- **Estimator**: Gamma-MAP, eLORETA, Bayesian Minimum Norm
- **Orientation**: Fixed or Free
- **Noise Type**: Oracle, Baseline, Cross-Validation, Joint Learning
- **SNR Level (α)**: Regularization strength control
- **Active Sources (nnz)**: Non-zero sources

Outcomes
--------

- **Regression Calibration Curves** (confidence intervals)
- **Classification Calibration Curves** (activation probabilities)
- **Quantitative Calibration Metrics**