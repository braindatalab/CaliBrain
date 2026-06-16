How to Cite CaliBrain
=====================

If CaliBrain contributes to a publication, preprint, thesis, or software
project, cite the archived software release used for the analysis.

Recommended citation
--------------------

Cite CaliBrain as software:

.. code-block:: text

   Orabe, M., Huseynov, I., Nagarajan, S., & Haufe, S. (2026).
   CaliBrain: Python framework for uncertainty estimation and calibration in
   EEG/MEG inverse source imaging (v1.0.1). Zenodo.
   https://doi.org/10.5281/zenodo.20703249

If you use another release, replace the version with the exact version used for
the analysis. For development versions, also report the git commit hash:

.. code-block:: bash

   git rev-parse HEAD

BibTeX
------

.. code-block:: bibtex

   @software{calibrain,
     title = {CaliBrain: Python framework for uncertainty estimation and calibration in EEG/MEG inverse source imaging},
     author = {Orabe, Mohammad and Huseynov, Ismail and Nagarajan, Srikantan and Haufe, Stefan},
     version = {1.0.1},
     year = {2026},
     publisher = {Zenodo},
     doi = {10.5281/zenodo.20703249},
     url = {https://doi.org/10.5281/zenodo.20703249}
   }

What to report
--------------

For scientific reproducibility, report:

- CaliBrain version or git commit hash.
- Python version and key dependency versions.
- Workflow configs used for data generation, aggregation, and calibration.
- Dataset or leadfield source.
- Calibration mode, solver, orientation type, noise type, SNR, NNZ, and split
  definition.
