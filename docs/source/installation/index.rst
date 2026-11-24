.. meta::
   :description: How to install CaliBrain: step-by-step instructions for pip, conda, a source checkout, and a container, with a check that the installation works and fixes for common errors.

Installation
============

This page tells you what you need and which way of installing to choose. The pages for
the individual ways are linked in the table below. At the end of every way, you check
the installation with the same short test (see :ref:`install-check`).

**In short:** if you have Python 3.10 or newer, open a terminal and run

.. code-block:: bash

   python -m pip install calibrain

then continue with :ref:`install-check`. If this fails or you are unsure, read on.

Requirements
------------

- **Python 3.10 or newer.** CaliBrain is tested with Python 3.10 to 3.14.
- **A terminal.** On Windows, open *PowerShell* from the Start menu. On macOS, open
  *Terminal* (in Applications, Utilities). On Linux, open your terminal program.

To see which Python you have, type:

.. code-block:: bash

   python --version

If it prints ``Python 3.10.x`` or higher, you are ready. If it prints an older version, or
an error such as ``command not found`` or ``Python was not found``, install Python from
https://www.python.org/downloads/ or use conda (see :doc:`conda_installation`). On Linux
and macOS the command may be called ``python3``. Then use ``python3`` wherever this guide
says ``python``.

Choose how to install
---------------------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - I want to ...
     - Use
   * - use CaliBrain in my own scripts, and I have Python
     - :doc:`pypi_installation`
   * - use conda or Anaconda to manage my Python
     - :doc:`conda_installation`
   * - change the code, run the tests, or build the documentation
     - :doc:`pip_installation`
   * - run CaliBrain on a cluster or server, or get exactly the same software every time
     - :doc:`container_installation`

If you are unsure, take the first row.

.. _install-environment:

Use a separate environment
--------------------------

An *environment* is a folder with its own copy of Python and its own packages. Installing
CaliBrain into one keeps it apart from your other projects and avoids version conflicts.
We recommend it for every way of installing, except the container. Create it once, and
activate it every time you open a new terminal.

With ``venv`` (included in Python). Create the environment in the folder ``.venv``:

.. code-block:: bash

   python -m venv .venv

Activate it. On Linux and macOS:

.. code-block:: bash

   source .venv/bin/activate

On Windows (PowerShell):

.. code-block:: powershell

   .venv\Scripts\Activate.ps1

When it is active, the prompt starts with ``(.venv)``. Type ``deactivate`` to leave it. If
Windows reports that running scripts is disabled, run
``Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`` once and activate again.

With conda, see :doc:`conda_installation`.

.. _install-extras:

Optional extras
---------------

CaliBrain installs everything it needs for simulation, inverse solvers, calibration, and
the 2D plots. Some tasks need more packages, which you add with an *extra* in square
brackets. Quote it, so that the terminal does not misread the brackets:

.. list-table::
   :header-rows: 1
   :widths: 14 56 30

   * - Extra
     - What it adds
     - Example
   * - ``viz``
     - 3D brain plots (``PyQt5``, ``pyvistaqt``, ``ipywidgets``). They need a screen.
     - ``pip install "calibrain[viz]"``
   * - ``dev``
     - ``pytest`` and ``ruff``, to run the tests.
     - ``pip install -e ".[dev]"``
   * - ``docs``
     - The tools that build this documentation.
     - ``pip install -e ".[docs]"``

Combine extras with commas: ``".[viz,dev,docs]"``. The ``-e`` in the last two examples is
explained in :doc:`pip_installation`.

.. _install-check:

Check the installation
----------------------

In the terminal where the environment is active, run:

.. code-block:: bash

   python -c "import calibrain; print(calibrain.__version__)"

It should print a version number such as ``1.1.0``. If it prints an error, see the next
section.

If something goes wrong
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - What you see
     - What to do
   * - ``pip: command not found``
     - Always write ``python -m pip`` instead of ``pip``, as in this guide.
   * - ``Requires-Python >=3.10`` or ``No matching distribution found``
     - Your Python is too old. Check it with ``python --version`` and install a newer one.
   * - ``error: externally-managed-environment``
     - Your system protects its own Python. Create an environment as described
       :ref:`above <install-environment>` and install there.
   * - ``ModuleNotFoundError: No module named 'calibrain'``
     - You are in a different environment than the one you installed into. Activate it, or
       install again.
   * - Windows: ``running scripts is disabled``
     - See the note under :ref:`install-environment`.
   * - Something else
     - Open an issue at https://github.com/braindatalab/CaliBrain/issues with the complete
       error message, your operating system, and the output of ``python --version``.

Next steps
----------

CaliBrain works without data for simulations. The real head models (example data) are
downloaded in a separate step, which is described in :doc:`../datasets/index`, and the
tutorials start with :doc:`../tutorials/index`.

.. toctree::
   :hidden:

   pypi_installation
   conda_installation
   pip_installation
   container_installation
