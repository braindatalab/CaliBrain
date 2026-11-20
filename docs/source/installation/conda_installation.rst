.. meta::
   :description: Install CaliBrain into a conda environment, either from PyPI or from a repository checkout.

Conda Installation
==================

CaliBrain is distributed on PyPI. With conda, you create an environment and
install CaliBrain into it with ``pip``. There are two ways to do this.

Install the PyPI package into a conda environment
-------------------------------------------------

Use this if you want the released package. Create a new environment with a
supported Python version, activate it, and install with ``pip``:

.. code-block:: bash

   conda create -n calibrain python=3.11 pip
   conda activate calibrain
   python -m pip install calibrain

To use an existing environment instead, activate it and run only the last
command. For 3D plotting, install the optional extra:

.. code-block:: bash

   python -m pip install "calibrain[viz]"

Use the repository environment file
-----------------------------------

Use this if you work from a source checkout. The repository ``environment.yml``
only defines Python and ``pip``. The package dependencies are defined in
``pyproject.toml`` and installed in the second step.

Clone the repository:

.. code-block:: bash

   git clone https://github.com/braindatalab/CaliBrain.git
   cd CaliBrain

Create the environment:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate calibrain

Install the package from the checkout:

.. code-block:: bash

   python -m pip install -e .

Optional extras
~~~~~~~~~~~~~~~

3D plotting, documentation, and development dependencies:

.. code-block:: bash

   python -m pip install -e ".[viz]"
   python -m pip install -e ".[docs]"
   python -m pip install -e ".[dev]"

Check that the installation worked
----------------------------------

.. code-block:: bash

   python -c "import calibrain; print(calibrain.__version__)"
