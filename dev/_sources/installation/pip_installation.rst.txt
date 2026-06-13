Pip Installation
================

Install from a local checkout:

.. code-block:: bash

   git clone https://github.com/braindatalab/CaliBrain.git
   cd CaliBrain
   python -m pip install -e .

Install documentation dependencies:

.. code-block:: bash

   python -m pip install -e ".[docs]"

Install development dependencies:

.. code-block:: bash

   python -m pip install -e ".[dev]"

The workflow scripts may require precomputed forward solutions and leadfields.
Set ``CALIBRAIN_DATA`` if these files live outside the repository ``data/``
directory.
