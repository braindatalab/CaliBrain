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

It is recommended to install into a virtual environment:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
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

Install documentation dependencies:

.. code-block:: bash

   python -m pip install -e ".[docs]"

Install development dependencies:

.. code-block:: bash

   python -m pip install -e ".[dev]"

Check that the installation worked
----------------------------------

.. code-block:: bash

   python -c "import calibrain; print(calibrain.__version__)"
