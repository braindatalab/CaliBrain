.. meta::
   :description: Install CaliBrain from a clone of the repository in editable mode, to change the code, run the tests, or build the documentation.

Install from source
===================

Use this if you want to change CaliBrain, run its tests, or build this documentation. To
only use CaliBrain, :doc:`pypi_installation` is simpler.

You need Python 3.10 or newer, a terminal, and Git (``git --version`` should print a
version; otherwise install Git from https://git-scm.com/downloads).

Step 1: Get the code
--------------------

.. code-block:: bash

   git clone https://github.com/braindatalab/CaliBrain.git
   cd CaliBrain

All the following commands are run in this folder (``CaliBrain``).

Step 2: Create and activate an environment
------------------------------------------

Follow :ref:`install-environment`. If you use conda, follow :doc:`conda_installation`
instead of steps 2 and 3.

Step 3: Install in editable mode
--------------------------------

.. code-block:: bash

   python -m pip install -e ".[dev]"

The option ``-e`` (*editable*) links the installation to the folder, so that changes you
make in the code take effect without installing again. ``[dev]`` adds ``pytest`` and
``ruff``. To also build the documentation, use ``".[dev,docs]"``. For all extras, see
:ref:`install-extras`.

Step 4: Check the installation
------------------------------

Follow :ref:`install-check`. Then run the tests, which take about half a minute:

.. code-block:: bash

   python -m pytest tests/

Tests that need the example data are skipped until you have downloaded it
(:doc:`../datasets/index`).

Next steps
----------

How to make a change and submit it is described in :doc:`../development/contributing`.
