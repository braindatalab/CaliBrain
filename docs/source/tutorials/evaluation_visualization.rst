Evaluation and Visualization
============================


Calibration metrics
-------------------


The calibration workflow reports metrics computed from the difference between
nominal and empirical coverage curves. The default calibration metrics include:

- mean signed deviation
- mean absolute deviation
- maximum underconfidence deviation
- maximum overconfidence deviation

These metrics summarize the calibration curve but should be interpreted
alongside the full curve. A single scalar can hide local failures at low or high
coverage.

Source-space evaluation
-----------------------


When source coordinates are available, the calibration workflow can compute EMD
using ``compute_dataset_emd``. For reduced MEG free-orientation models, source
estimates can be lifted through ``Q_basis`` so that metrics are evaluated in a
consistent source-space representation.

Visualization
-------------


The core visualization path for the current paper pipeline is:

.. code-block:: bash

   python calibrain/workflows/plot_paper_calibration_figures.py


The plotting workflow reads calibration JSON files and follows their
``eval_source`` links to recover metadata from the aggregated NPZ datasets. This
is why aggregation writes scalar metadata such as solver, noise type,
orientation, SNR, NNZ, subject, seed, and run ID.

For exploratory analysis, calibration JSON files are simple enough to inspect
with standard Python tools:

.. code-block:: python

   import json
   from pathlib import Path
   
   path = Path("results/calibration_eval/example.json")
   payload = json.loads(path.read_text())
   nominal = payload["pre_calibration"]["nominal_coverages"]
   empirical = payload["pre_calibration"]["empirical_coverages"]
