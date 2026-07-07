Tutorial Scripts
================

These tutorials are executable Sphinx-Gallery scripts. They are stored in the
repository-level ``tutorials`` directory and rendered into the documentation
under ``docs/source/auto_tutorials`` during ``make html``.

Each tutorial should be deterministic, lightweight, and runnable without large
local datasets unless it is explicitly marked as a large workflow tutorial.
Use numeric filename prefixes, for example ``03_quick_start.py``, so the
gallery presents tutorials in the intended reading order.
