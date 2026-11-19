.. meta::
   :description: Citation guidance and BibTeX metadata for referencing CaliBrain software releases in academic work.

How to Cite CaliBrain
=====================

If CaliBrain contributes to a publication, preprint, thesis, or software
project, cite the archived software release used for the analysis.

Recommended citation
--------------------

.. code-block:: bibtex

   Orabe, Mohammad, Nagarajan, Srikantan, Haufe, Stefan, & Huseynov, Ismail T. (2026).
   CaliBrain: A Python toolbox for uncertainty quantification, calibration, and
   benchmarking in Bayesian M/EEG inverse source imaging. Zenodo.
   https://doi.org/10.5281/zenodo.20683761

This DOI always points to the latest version of the archive. To make an analysis
reproducible, also report the exact version (``calibrain.__version__``), and cite the
DOI of that release, which the Zenodo record lists for every version. For development
versions, report the git commit hash:

.. code-block:: bash

   git rev-parse HEAD

BibTeX
------

.. code-block:: bibtex

   @software{calibrain,
     title = {CaliBrain: A Python toolbox for uncertainty quantification, calibration, and benchmarking in Bayesian M/EEG inverse source imaging},
     author = {Orabe, Mohammad and Nagarajan, Srikantan and Haufe, Stefan and Huseynov, Ismail T.},
     year = {2026},
     publisher = {Zenodo},
     doi = {10.5281/zenodo.20683761},
     url = {https://doi.org/10.5281/zenodo.20683761}
   }
