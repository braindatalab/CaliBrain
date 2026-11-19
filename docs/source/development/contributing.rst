.. meta::
   :description: How to report bugs, set up a development environment, make a change, and submit a pull request to CaliBrain, with the maintainer checklists for releases and example data.

Contributing
============

Changes are made on a branch and submitted as a pull request against the ``dev``
branch. This page lists the steps. ``CONTRIBUTING.md`` in the repository root is the
short version for GitHub. How the code and the tests are organized is described in the
:doc:`Development Guide <index>`.

Reporting a bug or proposing a change
-------------------------------------

Open an issue at https://github.com/braindatalab/CaliBrain/issues.

For a bug, include:

- the CaliBrain version (``python -c "import calibrain; print(calibrain.__version__)"``),
  the Python version, and the operating system;
- a short script that reproduces the problem, and the complete traceback.

Open an issue before you start on a larger change, for example a new solver, a
different signature of a public function, or a change of a file format. Then the design
can be discussed before you write the code.

Setting up
----------

CaliBrain needs Python 3.10 or newer. Clone the repository and install it in editable
mode with the development and documentation extras. With ``venv`` (on Windows, activate
with ``.venv\Scripts\Activate.ps1``):

.. code-block:: bash

   git clone https://github.com/braindatalab/CaliBrain.git
   cd CaliBrain
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -e ".[dev,docs]"

With conda, ``environment.yml`` creates an environment with Python 3.11 and ``pip``:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate calibrain
   python -m pip install -e ".[dev,docs]"

Most tests run without data. Some tests and the tutorials 05, 06, and 15 use the example
data. To prepare it, see :doc:`../datasets/index`.

Branches and commits
--------------------

Start from an up-to-date ``dev`` and use one branch per change:

.. code-block:: bash

   git switch dev
   git pull
   git switch -c feature/short-description

Name branches ``feature/...``, ``fix/...``, ``documentation/...``, or ``refactor/...``. Start
every commit message with its type, as in the history of the repository: ``feat:``,
``fix:``, ``docs:``, ``refactor:``, ``test:``, ``ci:``, or ``chore:``. Then write a short
summary in the imperative, for example ``fix: keep the source order in the posterior
summary``. If the reason for a change is not obvious, explain it in the body of the message.

Checks before you open a pull request
-------------------------------------

.. code-block:: bash

   python -m pytest tests/
   ruff check .
   cd docs && make html SPHINXOPTS="-W --keep-going"

- ``pytest`` runs the test suite. Add or update tests for every change of behavior. The
  rules for tests are in the :doc:`Development Guide <index>`.
- ``ruff`` finds syntax errors and undefined names.
- The documentation build with ``-W`` turns warnings into errors, as on Read the Docs.
  It also runs the tutorials, so a tutorial that fails fails the build. Build the
  documentation whenever you change a public function, a workflow, or a tutorial, and
  update the pages that describe it in the same pull request.

The same checks run on GitHub for every pull request (``tests.yml`` and
``docs-check.yml``; see the :doc:`Development Guide <index>`), on Python 3.10 to 3.14 and
with the lowest allowed dependency versions.

Changelog entry
---------------

Add one line to the top section of the :doc:`changelog <changelog>` (the one marked
``unreleased``). Begin it with a tag and describe what changes for the user, not how you
implemented it. Put names of functions, classes, and files in double backticks.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Tag
     - Use for
   * - ``[FEATURE]``
     - New public functionality.
   * - ``[ENHANCEMENT]``
     - An improvement of existing behavior that needs no change in user code.
   * - ``[BUGFIX]``
     - A fix of wrong behavior.
   * - ``[BREAKING]``
     - A change after which user code or existing data files must be changed.
   * - ``[DOCS]``, ``[TEST]``, ``[CI]``
     - Changes of the documentation, the tests, or the continuous integration.

Pull request
------------

Open the pull request against ``dev``, from a branch that contains the current ``dev``
(merge it or rebase on it). Keep it to one problem or one feature. The description says:

- what problem the change solves and how it does it;
- what changes for users, and which changelog line describes it;
- which tests you added or ran, and anything that is not finished or known to be
  limited.

The pull request can be merged when the checks on GitHub have passed.

By taking part in this project you agree to follow ``CODE_OF_CONDUCT.md``.

Releasing (maintainers)
-----------------------

A release is permanent: the GitHub release is archived by Zenodo and uploaded to
PyPI, and neither can be changed afterwards. Go through this checklist before
publishing:

1. Check that the :doc:`changelog <changelog>` lists every change since the last
   release, and replace ``(unreleased)`` with the release date.
2. Update the version in ``calibrain/__init__.py`` (the only place the package
   reads it from), in ``CITATION.cff`` (``version`` and ``date-released``), and in
   ``docs/source/_static/switcher.json``: add an entry for the new version, and set
   the ``version`` of the ``"stable"`` entry to the new release number.
3. Run ``python -m pytest tests/`` and ``ruff check .``, build the documentation with
   ``make html SPHINXOPTS="-W --keep-going"``, and run
   ``python -m pytest --network tests/test_datasets.py`` to check the real downloads of
   the example data.
4. Check what will be archived: look through ``git ls-files`` for unpublished,
   private, or large files. Anything tracked in the tagged commit ends up in the
   Zenodo archive.
5. Tag the release commit ``vX.Y.Z`` and publish a GitHub release. Publishing
   runs the tests and uploads the package to PyPI, and Zenodo archives the
   release automatically.
6. Update the documentation on Read the Docs. The ``stable`` version is built from
   the git branch ``stable``, not from the newest tag, so a tag alone does not
   update it: move the branch to the release commit (for example
   ``git push origin <release commit>:stable``). New tags are created as inactive
   versions, so also activate the version ``vX.Y.Z`` in the Read the Docs project
   (Versions), which builds it.
7. When Zenodo has created the new version, update the DOI in ``CITATION.cff``,
   ``README.md``, ``docs/source/cite/index.rst``, and ``docs/source/index.rst``.
8. Verify the release: install ``calibrain==X.Y.Z`` into a fresh environment, and
   check the Read the Docs ``stable`` build and the Zenodo record.
9. Run ``python scripts/check_version_switcher.py``. It checks the version menu of
   every documented version against the live site (see below).

.. _doc-version-switcher:

The documentation version switcher
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every version of the documentation loads its list of versions from
``https://calibrain.readthedocs.io/stable/_static/switcher.json``. That address is
stored in the HTML of every build and cannot be changed afterwards, so the address
and the file must never move: add a new entry for every release, and keep the
addresses of the form ``https://calibrain.readthedocs.io/<version>/`` (without
``/en/``, and never GitHub Pages).

The ``"stable"`` entry is marked ``"preferred": true``, and its ``version`` is the
current release number. The theme compares it with the version of each page and
shows a banner on older pages ("This is documentation for an old version") and on
development builds that are ahead of it. Exactly one entry may be preferred, and a
test checks that it matches ``calibrain.__version__``. Because the banner is part of
every build, old versions show it as soon as they can load this list.

Releases up to v1.0.3 were built before this was fixed: v0.1.2 and v1.0.0 load their
own bundled list (which is malformed or links to the old GitHub Pages site), and
v1.0.1 to v1.0.3 load ``/en/latest/_static/switcher.json``, which does not exist on
this project. They are fixed with redirects in the Read the Docs project (Admin,
Redirects), in this order, all of type "Exact Redirect":

1. ``/v0.1.2/_static/switcher.json`` to ``/stable/_static/switcher.json`` (force redirect)
2. ``/v1.0.0/_static/switcher.json`` to ``/stable/_static/switcher.json`` (force redirect)
3. ``/en/latest/_static/switcher.json`` to ``/stable/_static/switcher.json``
4. ``/en/*`` to ``/:splat``

Rule 4 also sends any other old address with an ``/en/`` prefix to the right page.
The first matching rule is applied, so keep the order. After changing the
redirects, ``python scripts/check_version_switcher.py`` must report no problems.

Publishing new example data (maintainers)
-----------------------------------------

The example data live in the
`CaliBrain-data <https://github.com/braindatalab/CaliBrain-data>`_ repository as
release assets, never in git. Every dataset (``fsaverage``, ``camcan``) has its own
release, and a release is never changed afterwards: new data means a new release,
because the checksums in ``calibrain/datasets.py`` must stay valid for older
versions of CaliBrain.

The archives contain the forward solutions with the full source space (the files in
``<data>/fwd``, stored as ``fwd/<subject>-fwd.fif``). The reduced forward solutions and
the leadfield files depend on the version of CaliBrain, so they are not published; users
create them with ``reduce_forward_solutions`` and ``extract_leadfields``.

1. Build one archive per dataset, in the same Python environment each time (the
   checksum depends on the MNE and NumPy versions). A separate output folder per dataset
   gives each dataset its own ``manifest.json``:

   .. code-block:: bash

      python scripts/build_example_data_archives.py --output ~/archives-fsaverage --datasets fsaverage
      python scripts/build_example_data_archives.py --output ~/archives-camcan --datasets camcan

   The script removes local paths from the files, stops if a personal path or a
   numerical difference remains, and writes the size and SHA-256 checksum of every
   archive to ``manifest.json``. Only publish ``camcan`` if the CamCAN terms allow
   sharing derived data (see ``DATA_NOTICE.rst``).
2. In CaliBrain-data, create a release for each dataset with a new tag
   (``example-data-fsaverage-3``, ``example-data-camcan-3``, ...: the number after the
   last existing one). Put the checksum and the file list in the description, and drag
   the zip file (and ``manifest.json``) into the assets box, not into the description.
3. Download each asset again and check that its SHA-256 equals the one in
   ``manifest.json``.
4. In ``calibrain/datasets.py``, update ``version``, ``sha256``, and, if they changed,
   ``archive`` and ``files`` of the dataset in the registry. Run
   ``python -m pytest --network tests/test_datasets.py``, which downloads every dataset
   and checks it.
5. Update the README of CaliBrain-data, the sizes in the documentation
   (:doc:`../datasets/index`), and the changelog.
