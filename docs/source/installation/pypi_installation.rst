.. meta::
   :description: Install CaliBrain with pip from the Python Package Index: the latest release, a specific version, an upgrade, and how to remove it.

Install with pip (PyPI)
=======================

This is the usual way. It installs the released package from the Python Package Index
(PyPI). Use it to run CaliBrain in your own scripts and notebooks. To change the code,
see :doc:`pip_installation` instead.

You need Python 3.10 or newer and a terminal (see :doc:`index`).

Step 1: Create and activate an environment
------------------------------------------

Follow :ref:`install-environment`. If you skip this step, CaliBrain is installed into your
default Python.

Step 2: Install CaliBrain
-------------------------

.. code-block:: bash

   python -m pip install calibrain

This also installs the packages CaliBrain needs (NumPy, SciPy, MNE-Python, and others),
which can take a few minutes.

Step 3: Check the installation
------------------------------

Follow :ref:`install-check`. It should print the version number.

Other things you may need
-------------------------

**Install with 3D plotting.** Add the ``viz`` extra (see :ref:`install-extras`):

.. code-block:: bash

   python -m pip install "calibrain[viz]"

**Install a specific version**, for example to reproduce an earlier result:

.. code-block:: bash

   python -m pip install "calibrain==1.1.0"

**Upgrade** to the newest release:

.. code-block:: bash

   python -m pip install --upgrade calibrain

**Uninstall:**

.. code-block:: bash

   python -m pip uninstall calibrain

Next steps
----------

Continue with :doc:`../datasets/index` to get the example data, or with the
:doc:`tutorials <../tutorials/index>`.
