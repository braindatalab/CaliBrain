How to Cite CaliBrain
=====================

If CaliBrain contributes to a publication, preprint, thesis, or software
project, cite the software version and repository used for the analysis.

Recommended citation
--------------------

Until a formal software paper or archived DOI is available, cite CaliBrain as
software:

.. code-block:: text

   Orabe, M., Huseynov, I., Nagarajan, S., & Haufe, S. CaliBrain:
   simulation-based uncertainty calibration for M/EEG inverse source imaging.
   Software, version <version>, https://github.com/braindatalab/CaliBrain

Replace ``<version>`` with the version or git commit hash used for the analysis.
For reproducibility, include the commit hash whenever possible:

.. code-block:: bash

   git rev-parse HEAD

BibTeX
------

.. code-block:: bibtex

   @software{calibrain,
     title = {CaliBrain: simulation-based uncertainty calibration for M/EEG inverse source imaging},
     author = {Orabe, Mohammad and Huseynov, Ismail and Nagarajan, Srikantan and Haufe, Stefan},
     url = {https://github.com/braindatalab/CaliBrain},
     version = {<version-or-commit>},
     year = {<year>}
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

When a DOI or software paper becomes available, this page should be updated and
the BibTeX entry should cite the archived release.
