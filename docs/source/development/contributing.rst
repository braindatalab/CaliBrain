.. meta::
   :description: Learn how to contribute code, tests, documentation, bug reports, and workflow improvements to CaliBrain.

Contributing
============

This page describes how to contribute to CaliBrain development. For the
repository-level entry point used by GitHub and package reviewers, see
``CONTRIBUTING.md`` in the repository root.

Ways to contribute
------------------

Contributions are welcome in several forms:

- bug reports and bug fixes;
- documentation improvements;
- tests for core package functionality;
- workflow and API improvements;
- packaging and release infrastructure improvements.

Before you start
----------------

Before opening a pull request:

- check whether a related issue already exists;
- open an issue first for substantial changes;
- keep changes focused on one problem or one feature;
- update documentation when user-facing behavior changes.

Development setup
-----------------

Clone the repository:

.. code-block:: bash

   git clone https://github.com/braindatalab/CaliBrain.git
   cd CaliBrain

Create and activate an isolated environment. Either a virtual environment or a
conda environment is fine. One minimal ``venv``-based setup is (on Windows,
activate with ``.venv\Scripts\Activate.ps1``):

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip

Install CaliBrain in editable mode with development and documentation extras:

.. code-block:: bash

   python -m pip install -e ".[dev,docs]"

Create a working branch
-----------------------

Create a branch for your change:

.. code-block:: bash

   git checkout -b feature/short-description

Use a separate branch for each independent change.

Coding expectations
-------------------

Contributions should follow the current package conventions:

- keep changes small and targeted;
- preserve the current public workflow unless the change is explicitly intended
  to modify it;
- use clear names for variables, functions, and classes;
- write NumPy-style docstrings for public functions and classes;
- add type hints when they improve clarity;
- prefer deterministic examples and fixed random seeds in tutorials and tests.

Testing and linting
-------------------

Run the relevant tests for the code you changed. Prefer running the smallest
relevant subset first and then the whole suite:

.. code-block:: bash

   python -m pytest tests/
   python -m pytest tests/ --cov=calibrain
   ruff check .

Tests that need the example forward-model files (see :doc:`../datasets/index`)
are skipped when the files are not available. If you add or change public
behavior, add or update tests.

Documentation
-------------

If you change public APIs, workflows, or tutorials, update the documentation in
the same pull request.

Build the documentation locally with:

.. code-block:: bash

   cd docs
   make html

If Sphinx-Gallery examples are affected, ensure that the relevant tutorial
scripts still execute successfully.

Submitting a pull request
-------------------------

Open pull requests against the ``dev`` branch. Before opening a pull request:

- make sure your branch is up to date with ``dev``;
- write a clear commit history;
- summarize what changed and why;
- mention any limitations, follow-up work, or known issues.

In the pull request description, include:

- the problem being addressed;
- the approach taken;
- any user-facing changes;
- any documentation or test updates.

Code of conduct
---------------

By participating in this project, you agree to follow the repository
``CODE_OF_CONDUCT.md``.

Releasing (maintainers)
-----------------------

A release is permanent: the GitHub release is archived by Zenodo and uploaded to
PyPI, and neither can be changed afterwards. Go through this checklist before
publishing:

1. Check that the :doc:`changelog <changelog>` lists every change since the last
   release, and replace ``(unreleased)`` with the release date.
2. Update the version in ``calibrain/__init__.py`` (the only place the package
   reads it from), in ``CITATION.cff`` (``version`` and ``date-released``), and
   add the new version to ``docs/source/_static/switcher.json``.
3. Run ``python -m pytest tests/`` and ``ruff check .``, and build the
   documentation.
4. Check what will be archived: look through ``git ls-files`` for unpublished,
   private, or large files. Anything tracked in the tagged commit ends up in the
   Zenodo archive.
5. Tag the release commit ``vX.Y.Z`` and publish a GitHub release. Publishing
   runs the tests and uploads the package to PyPI, and Zenodo archives the
   release automatically.
6. When Zenodo has created the new version, update the DOI in ``CITATION.cff``,
   ``README.md``, ``docs/source/cite/index.rst``, and ``docs/source/index.rst``.
7. Verify the release: install ``calibrain==X.Y.Z`` into a fresh environment, and
   check the Read the Docs ``stable`` build and the Zenodo record.

Publishing new example data (maintainers)
-----------------------------------------

The example data live in the
`CaliBrain-data <https://github.com/braindatalab/CaliBrain-data>`_ repository as
release assets, never in git. A release is never changed afterwards: new data
means a new release, because the checksums in ``calibrain/datasets.py`` must
stay valid for older versions of CaliBrain.

1. Build the archives in the same Python environment each time (the checksum
   depends on the MNE and NumPy versions):

   .. code-block:: bash

      python scripts/build_example_data_archives.py --output ~/example_data_upload

   The archives contain only the forward solutions, because the leadfield files
   depend on the version of CaliBrain and are created locally with
   ``python -m calibrain.extract_leadfield``. The script removes local paths,
   refuses to continue if a personal path or a numerical difference remains, and
   writes ``manifest.json`` with the SHA-256 checksums. Only add
   ``--include-camcan`` if the CamCAN terms allow sharing derived data.
2. In CaliBrain-data, create a release with a new tag (``example-data-2``, ...) and
   attach the zip files and ``manifest.json``.
3. Download an asset again and check that its SHA-256 equals the one in
   ``manifest.json``.
4. In ``calibrain/datasets.py``, set ``DATA_VERSION`` to the new tag and update the
   archive names and checksums, then update the changelog.
