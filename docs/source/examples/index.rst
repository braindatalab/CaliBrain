Examples Gallery
================


The examples gallery contains small, runnable Sphinx-Gallery scripts that
demonstrate one concept at a time. Example pages, thumbnails, downloadable
Python files, and notebooks are generated during ``make html`` from scripts in
the repository-level ``examples`` directory.

.. toctree::
   :maxdepth: 2
   :caption: Executable examples

   01. Executable Examples Gallery <../auto_examples/index>


Beginner examples
-----------------


- Minimal fixed-orientation simulation and calibration.
- Reading a calibration JSON summary.
- Plotting a single calibration curve from saved JSON.

Intermediate examples
---------------------


- Aggregating manifest rows for one solver and noise condition.
- Comparing ``full_cov`` and ``marginal`` free-orientation calibration.
- Computing EMD for an aggregated evaluation dataset.

Advanced examples
-----------------


- Running a full SNR sweep with ``post_fixed``.
- Comparing fixed and free-orientation calibration curves.
- Reproducing paper-style multi-panel figures.

Each future example should include the objective, complete code, expected
outputs, and a short interpretation section.
