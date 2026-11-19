Tutorial Scripts
================

These tutorials are executable Sphinx-Gallery scripts. They are stored in the
repository-level ``tutorials`` directory and rendered into the documentation
under ``docs/source/auto_tutorials`` during ``make html``.

Each tutorial should be deterministic, lightweight, and runnable without local
datasets. A tutorial may download the small example data on demand with
``fetch_example_data()`` (about 36 MB for the ``fsaverage`` template), and says so at
the top; it must not need the CamCAN data or any other large dataset unless it is
explicitly marked as a large workflow tutorial.
Use numeric filename prefixes, for example ``03_quick_start.py``, so the
gallery presents tutorials in the intended reading order.
