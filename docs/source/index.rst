.. CaliBrain documentation master file ...

CaliBrain documentation
=======================

**CaliBrain** is a Python library designed for simulating realistic Magnetoencephalography (MEG) and Electroencephalography (EEG) data and evaluating the calibration and uncertainty quantification of brain source estimation algorithms.

It provides tools to:

*   Simulate source activity, leadfield matrices, and sensor-level measurements with controllable noise levels and source configurations (fixed or free orientation).
*   Estimate and visualize confidence intervals for source estimates based on posterior covariance information.
*   Analyze the calibration of source estimation methods by comparing confidence levels to the actual proportion of ground truth values captured.

This documentation provides installation instructions, usage examples, and a detailed API reference.

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   installation
   usage
   api
   examples/index
   readme
   results
   contributing
   changelog