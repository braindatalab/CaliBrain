PyPI Installation
=================

Use this method if you want the released CaliBrain package and do not need a
local source checkout.

Install the latest release
--------------------------

Install CaliBrain from PyPI:

.. code-block:: bash

   python -m pip install calibrain

Install a specific version
--------------------------

If you need a specific release, pin the version explicitly:

.. code-block:: bash

   python -m pip install "calibrain==1.0.2"

Check that the installation worked
----------------------------------

Verify that Python can import the package:

.. code-block:: bash

   python -c "import calibrain; print(calibrain.__version__)"

If this prints a version number, the installation is usable.

Optional: use a virtual environment
-----------------------------------

If you do not want to install into the system Python environment, create and
activate a virtual environment first:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install calibrain
