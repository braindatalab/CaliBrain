Installation
============

CaliBrain can be installed from source for development or installed into a
standard Python environment for running workflows. The package metadata currently
declares Python ``>=3.10``; in practice, the scientific Python dependencies used
by the workflows are best managed with a recent conda or virtualenv environment.

Dependencies
------------

Core dependencies are declared in ``pyproject.toml`` and include:

* ``numpy``
* ``pandas``
* ``scipy``
* ``scikit-learn``
* ``mne``
* ``POT``
* ``nibabel``
* ``pyyaml``
* ``matplotlib`` and visualization dependencies

Documentation dependencies are declared in the optional ``docs`` extra.

Source installation
-------------------

From a local checkout:

.. code-block:: bash

   git clone https://github.com/braindatalab/CaliBrain.git
   cd CaliBrain
   python -m pip install -e .

For documentation builds:

.. code-block:: bash

   python -m pip install -e ".[docs]"

Conda environment
-----------------

The repository includes ``environment.yml``. Use it when you want conda to
create the scientific Python environment:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate calibrain
   python -m pip install -e .

Pip requirements
----------------

For a pip-only setup:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python -m pip install -e .

Data directory
--------------

Workflow examples expect precomputed forward solutions and leadfields. By
default, CaliBrain uses the repository ``data/`` directory. To use another data
location, set:

.. code-block:: bash

   export CALIBRAIN_DATA=/path/to/calibrain/data

The workflow configs also contain explicit paths for results, posterior
summaries, run manifests, aggregation outputs, and calibration outputs. Inspect
``configs/*.py`` before running large experiments.

Building the documentation
--------------------------

From the repository root:

.. code-block:: bash

   cd docs
   make html

The rendered site is written to ``docs/build/html/index.html``.

.. toctree::
   :maxdepth: 1
   :caption: Installation topics

   pip_installation
   conda_installation
