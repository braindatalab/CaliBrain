.. meta::
   :description: Install CaliBrain from PyPI, a local source checkout, or into a conda environment.

Installation
============

Choose one installation method and follow it from start to finish.

Requirements
------------

Before installing CaliBrain, make sure that:

- you have Python ``3.10`` or newer (CaliBrain is tested with Python 3.10 to 3.14);
- you have ``pip`` if you want to install from PyPI or from a local checkout;
- you have ``conda`` if you want to use a conda environment.

Which method should I use?
--------------------------

- :doc:`PyPI <pypi_installation>` if you want the latest released package and
  do not need to modify the source code.
- :doc:`pip <pip_installation>` if you want to install CaliBrain from a local
  repository checkout.
- :doc:`conda <conda_installation>` if you want to install into a conda
  environment, either from PyPI or from a source checkout.

Installation methods
--------------------

CaliBrain is documented through three installation paths:

- :doc:`PyPI <pypi_installation>` for installing a published release into an existing Python environment;
- :doc:`pip <pip_installation>` for installing directly from a local source checkout;
- :doc:`conda <conda_installation>` for installing into a conda environment, from PyPI or from a source checkout.

Optional packages
-----------------

3D brain plotting needs ``PyQt5``, ``pyvistaqt``, and ``ipywidgets``, which are
not installed by default. Add the ``viz`` extra to any of the methods above, for
example ``python -m pip install "calibrain[viz]"``.

Data files
----------

The example forward-model and leadfield files are not part of the package. Download
the forward solution with ``calibrain.datasets.fetch_example_data()`` and create the
leadfield files from it with ``calibrain.extract_leadfield.extract_leadfields()``. The
``CALIBRAIN_DATA`` environment variable chooses the folder they are stored in. See
:doc:`../datasets/index` for details and the expected layout.

.. toctree::
   :hidden:

   pypi_installation
   pip_installation
   conda_installation
