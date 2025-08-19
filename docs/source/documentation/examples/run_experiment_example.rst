Running Experiments Example
===========================

This example runs benchmark experiments for uncertainty calibration in source imaging.

Command Line Usage
------------------

To start the experiment:

.. code-block:: bash

   python examples/run_experiments.py

Key Actions:
- Generate synthetic dipoles activity.
- Simulate EEG/MEG measurements.
- Solve the inverse problem using Gamma-MAP (or other solvers).
- Evaluate calibration using regression and classification curves.
- Generate figures for uncertainty analysis.
- Save benchmark results automatically to the ``results/`` directory.