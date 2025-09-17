Conda Installation
-------------------

If you prefer to manage dependencies with **Conda**, you can create an isolated Conda environment using the provided `environment.yml` file.

First, clone the repository:

.. code-block:: bash

    git clone https://github.com/braindatalab/CaliBrain.git
    cd CaliBrain

Then create the environment:

.. code-block:: bash

    conda env create -f environment.yml

Activate the environment:

.. code-block:: bash

    conda activate calibrain

Finally, install CaliBrain into the activated environment:

.. code-block:: bash

    pip install .

This ensures that all Conda and pip dependencies are properly installed.

----


Optional Setup for Development
-------------------------------

If you plan to contribute to CaliBrain or run experiments:

.. code-block:: bash

    pip install -e .[dev]