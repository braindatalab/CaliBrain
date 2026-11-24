.. meta::
   :description: Install CaliBrain with conda: create an environment with Python and pip, install the released package, or set up a source checkout with the repository environment file.

Install with conda
==================

Use this if you manage your Python with conda or Anaconda. CaliBrain is installed from
PyPI with ``pip`` inside a conda environment. A conda environment is a folder with its own
Python and packages (see :ref:`install-environment`).

You need conda (install Miniconda from https://docs.conda.io/en/latest/miniconda.html if
``conda --version`` prints an error) and a terminal.

Install the released package
----------------------------

Create an environment called ``calibrain`` with Python 3.11 and ``pip``, activate it, and
install CaliBrain:

.. code-block:: bash

   conda create -n calibrain python=3.11 pip
   conda activate calibrain
   python -m pip install calibrain

Then follow :ref:`install-check`. When you open a new terminal, run
``conda activate calibrain`` again before you use CaliBrain.

For 3D plots or other extras, see :ref:`install-extras`. To use an existing environment,
activate it and run only the last command.

Set up a source checkout
------------------------

Use this to change the code. The file ``environment.yml`` in the repository contains only
Python and ``pip``. The packages CaliBrain needs are installed in the last step, from
``pyproject.toml``.

.. code-block:: bash

   git clone https://github.com/braindatalab/CaliBrain.git
   cd CaliBrain
   conda env create -f environment.yml
   conda activate calibrain
   python -m pip install -e ".[dev]"

The options of the last command are explained in :doc:`pip_installation`. Then follow
:ref:`install-check`.

Remove the environment
----------------------

.. code-block:: bash

   conda deactivate
   conda env remove -n calibrain
