.. meta::
   :description: Citation guidance and BibTeX metadata for referencing CaliBrain software releases in academic work.

How to Cite CaliBrain
=====================

If CaliBrain contributes to a publication, preprint, thesis, or software
project, cite the archived software release used for the analysis.

Recommended citation
--------------------

.. code-block:: bibtex

   @software{calibrain,
     title = {CaliBrain: A Python toolbox for uncertainty quantification, calibration, and benchmarking in Bayesian M/EEG inverse source imaging},
     author = {Orabe, Mohammad and Nagarajan, Srikantan and Haufe, Stefan and Huseynov, Ismail},
     year = {2026},
     publisher = {Zenodo},
     doi = {10.5281/zenodo.20683761},
     url = {https://doi.org/10.5281/zenodo.20683761}
   }

This DOI always points to the latest version of the archive. To make an analysis
reproducible, also report the exact version (``calibrain.__version__``), and cite the
DOI of that release, which the Zenodo record lists for every version. For development
versions, report the git commit hash:

.. code-block:: bash

   git rev-parse HEAD

The software paper
------------------

CaliBrain is described in a software paper, available as a preprint that has not yet
been peer reviewed. The DOI below always points to the latest version of the preprint.
When you describe the toolbox itself, cite the paper in addition to the software:

.. code-block:: bibtex

   @misc{orabe2026calibrain,
     title = {CaliBrain: A Python toolbox for uncertainty quantification, calibration, and benchmarking in Bayesian M/EEG inverse source imaging},
     author = {Orabe, Mohammad and Nagarajan, Srikantan and Haufe, Stefan and Huseynov, Ismail},
     year = {2026},
     publisher = {Zenodo},
     note = {Preprint},
     doi = {10.5281/zenodo.24544960},
     url = {https://doi.org/10.5281/zenodo.24544960}
   }

The method paper
----------------

The methods that CaliBrain implements, and the experiments it was built for, are
described in the preprint `Uncertainty calibration in Bayesian M/EEG source imaging
<https://arxiv.org/abs/2611.06979>`_ (arXiv:2611.06979). If your work builds on the
method and not only on the software, cite the preprint in addition to the software:

.. code-block:: bibtex

   @misc{huseynov2026uncertainty,
     title = {Uncertainty calibration in Bayesian M/EEG source imaging},
     author = {Huseynov, Ismail and Orabe, Mohammad and Nagarajan, Srikantan and Haufe, Stefan},
     year = {2026},
     eprint = {2611.06979},
     archivePrefix = {arXiv},
     url = {https://arxiv.org/abs/2611.06979}
   }
