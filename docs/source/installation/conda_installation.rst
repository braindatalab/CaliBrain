Conda Installation
==================

Use this method if you want to create the environment defined by the repository
``environment.yml`` file.

Clone the repository
--------------------

Download the source code:

.. code-block:: bash

   git clone https://github.com/braindatalab/CaliBrain.git
   cd CaliBrain

Create the conda environment
----------------------------

Create the environment from ``environment.yml``:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate calibrain

Install CaliBrain into the environment
--------------------------------------

Install the package from the checkout:

.. code-block:: bash

   python -m pip install -e .

Optional extras
---------------

For documentation work:

.. code-block:: bash

   python -m pip install -e ".[docs]"

For development work:

.. code-block:: bash

   python -m pip install -e ".[dev]"

Check that the installation worked
----------------------------------

.. code-block:: bash

   python -c "import calibrain; print(calibrain.__version__)"
