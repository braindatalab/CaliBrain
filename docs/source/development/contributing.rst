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
conda environment is fine. One minimal ``venv``-based setup is:

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

Testing
-------

Run the relevant tests for the code you changed. As the test suite expands,
prefer running the smallest relevant subset first and then the broader suite.

When a ``tests/`` suite is present, typical commands are:

.. code-block:: bash

   pytest tests/
   pytest tests/ --cov=calibrain

If you add or change public behavior, add or update tests when practical.

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

Before opening a pull request:

- make sure your branch is up to date with the target branch;
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
