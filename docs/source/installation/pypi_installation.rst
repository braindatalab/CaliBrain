.. meta::
   :description: Install the latest or a specific released version of CaliBrain from the Python Package Index.

PyPI Installation
=================

Use this method if you want the released CaliBrain package and do not need a
local source checkout.

Install the latest release
--------------------------

Install CaliBrain from PyPI:

.. code-block:: bash

   python -m pip install calibrain

Upgrade to a newer release
--------------------------

.. code-block:: bash

   python -m pip install --upgrade calibrain

Install the optional 3D plotting packages
-----------------------------------------

3D brain plotting needs a GUI toolkit (``PyQt5``, ``pyvistaqt``, and
``ipywidgets``). They are optional, so install them with the ``viz`` extra:

.. code-block:: bash

   python -m pip install "calibrain[viz]"

Install a specific version
--------------------------

If you need a specific release, pin the version explicitly:

.. code-block:: bash

   python -m pip install "calibrain==1.0.3"

Check that the installation worked
----------------------------------

Verify that Python can import the package:

.. code-block:: bash

   python -c "import calibrain; print(calibrain.__version__)"

If this prints a version number, the installation is usable.

Optional: use a virtual environment
-----------------------------------

If you do not want to install into the system Python environment, create and
activate a virtual environment first.

On Linux and macOS:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install calibrain

On Windows (PowerShell):

.. code-block:: powershell

   python -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install calibrain

Next steps
----------

The example forward-model and leadfield files are not part of the package.
Download the forward solution, reduce it to 1284 sources, then create the leadfield
files from it:

.. code-block:: python

   from calibrain import extract_leadfields, fetch_example_data, reduce_forward_solutions

   fetch_example_data()
   reduce_forward_solutions()
   extract_leadfields()

See :doc:`../datasets/index` for where they are stored and how to choose another folder.
