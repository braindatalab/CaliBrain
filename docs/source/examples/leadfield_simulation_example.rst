Leadfield Simulation Example
============================

This example demonstrates how to create or load a leadfield matrix.

Command Line Usage
------------------

To run the example with a specified configuration:

.. code-block:: bash

   python examples/leadfield_simulation_example.py --config configs/leadfield_sim_cfg.yml --log-level INFO

For default configuration, simply:

.. code-block:: bash

   python examples/leadfield_simulation_example.py

Main Steps:
- Set up the source space.
- Build the BEM model.
- Create or load the forward solution.
- Generate the leadfield matrix.