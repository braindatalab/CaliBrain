.. meta::
   :description: Install CaliBrain with pip from a local repository checkout for development or testing.

Pip Installation
================

Use this method if you want to install CaliBrain from a local repository
checkout.

Clone the repository
--------------------

Download the source code and move into the project directory:

.. code-block:: bash

   git clone https://github.com/braindatalab/CaliBrain.git
   cd CaliBrain

Create an isolated Python environment
-------------------------------------

It is recommended to install into a virtual environment.

On Linux and macOS:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip

On Windows (PowerShell):

.. code-block:: powershell

   python -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip

Install CaliBrain from the checkout
-----------------------------------

Install the package in editable mode:

.. code-block:: bash

   python -m pip install -e .

Editable mode means that changes in the local source tree are available without
reinstalling the package.

Optional extras
---------------

Install the 3D plotting packages (``PyQt5``, ``pyvistaqt``, ``ipywidgets``):

.. code-block:: bash

   python -m pip install -e ".[viz]"

Install documentation dependencies:

.. code-block:: bash

   python -m pip install -e ".[docs]"

Install development dependencies (``pytest``, ``pytest-cov``, ``ruff``):

.. code-block:: bash

   python -m pip install -e ".[dev]"

Extras can be combined, for example ``".[viz,dev,docs]"``.

Check that the installation worked
----------------------------------

.. code-block:: bash

   python -c "import calibrain; print(calibrain.__version__)"

With the ``dev`` extra installed, you can also run the test suite:

.. code-block:: bash

   python -m pytest tests/
