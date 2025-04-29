Running Experiments Example
===========================

This example runs benchmark experiments for uncertainty calibration in source imaging.

Command Line Usage
------------------

To start the experiment:

.. code-block:: bash

   python examples/run_experiments.py --config configs/experiment_cfg.yml --log-level INFO

Key Actions:
- Generate synthetic brain activity data.
- Solve the inverse problem using Gamma-MAP (or other solvers).
- Evaluate calibration using regression and classification curves.
- Save benchmark results automatically to the ``results/`` directory.