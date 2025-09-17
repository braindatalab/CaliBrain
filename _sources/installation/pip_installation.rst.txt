PIP Installation
-------------------------------

The recommended way to install CaliBrain is using `pip` with the `pyproject.toml` file.

First, clone the repository:

.. code-block:: bash

    git clone https://github.com/braindatalab/CaliBrain.git
    cd CaliBrain

Then install the package:

.. code-block:: bash

    pip install .

If you are actively developing the code and want automatic updates when you edit files:

.. code-block:: bash

    pip install -e .

This will install all dependencies defined inside `pyproject.toml`.

----

Simple Pip Installation via requirements.txt
---------------------------------------------

Alternatively, you can install CaliBrain using a traditional `requirements.txt`:

.. code-block:: bash

    pip install -r requirements.txt

Note:
    - This method is simpler but does **not** capture full metadata (e.g., Python version compatibility).
    - Make sure your environment uses a supported Python version (>=3.8).

----

Optional Setup for Development
-------------------------------

If you plan to contribute to CaliBrain or run experiments:

.. code-block:: bash

    pip install -e .[dev]