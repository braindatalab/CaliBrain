.. CaliBrain documentation master file ...

CaliBrain documentation
=======================

**CaliBrain** is a Python library designed for simulating Magnetoencephalography (MEG) and Electroencephalography (EEG) data and evaluating the calibration and uncertainty quantification of brain source estimation algorithms.

It provides tools to:

*   Simulate source activity, leadfield matrices, and sensor-level measurements with controllable noise levels and source configurations (fixed or free orientation).
*   Estimate and visualize confidence intervals for source estimates based on posterior covariance information.
*   Analyze the calibration of source estimation methods by comparing confidence levels to the actual proportion of ground truth values captured.

This documentation provides installation instructions, usage examples, and a detailed API reference.

.. toctree::
   :maxdepth: 2
   :caption: Installation
   :hidden:

   installation/index

.. toctree::
   :maxdepth: 2
   :caption: Documentation
   :hidden:

   documentation/index

.. toctree::
   :maxdepth: 2
   :caption: Development
   :hidden:

   development/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/index

Introduction
============

CaliBrain is designed to help researchers evaluate the quality of uncertainty estimates in brain source localization methods. The package provides comprehensive tools for simulation, analysis, and visualization.

Key Features
------------

- **Source Simulation**: Generate realistic brain source activity patterns
- **Forward Modeling**: Simulate sensor measurements with controllable noise
- **Inverse Solutions**: Multiple source estimation algorithms
- **Uncertainty Quantification**: Confidence interval estimation
- **Calibration Analysis**: Evaluate uncertainty estimate quality
- **Visualization**: Comprehensive plotting and analysis tools

Quick Start
-----------

.. code-block:: python

   from calibrain import Benchmark
   
   # Configure ERP parameters
   ERP_config = {
       "tmin": -0.5,
       "tmax": 0.5,
       "sfreq": 250,
       "amplitude": 50.0
   }
   
   # Run benchmark
   benchmark = Benchmark(ERP_config=ERP_config)
   results = benchmark.run()

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`