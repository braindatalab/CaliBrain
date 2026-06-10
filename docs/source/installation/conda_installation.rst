Conda Installation
==================

Create the environment from ``environment.yml``:

.. code-block:: bash

   git clone https://github.com/braindatalab/CaliBrain.git
   cd CaliBrain
   conda env create -f environment.yml
   conda activate calibrain

Install CaliBrain into the environment:

.. code-block:: bash

   python -m pip install -e .

For documentation work:

.. code-block:: bash

   python -m pip install -e ".[docs]"

Inspect ``configs/*.py`` before running workflows, because the default configs
may write results to absolute paths on the local machine where the experiments
were developed.
