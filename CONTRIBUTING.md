# Contributing to CaliBrain

Thank you for contributing to CaliBrain.

This file is the repository-level contribution entry point. The full
development guide is available in the rendered documentation and in
`docs/source/development/contributing.rst`.

## Before you start

- Check whether a related issue already exists.
- Open an issue first for substantial changes.
- Keep each pull request focused on one problem or one feature.
- Update documentation when user-facing behavior changes.

## Development setup

Clone the repository:

```bash
git clone https://github.com/braindatalab/CaliBrain.git
cd CaliBrain
```

Create and activate an isolated environment (on Windows, activate with
`.venv\Scripts\Activate.ps1`):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install CaliBrain with development and documentation extras:

```bash
python -m pip install -e ".[dev,docs]"
```

## Working on a change

Start from an up-to-date `dev` and create a branch for your work:

```bash
git switch dev
git pull
git switch -c feature/short-description
```

Name branches `feature/...`, `fix/...`, `docs/...`, or `refactor/...`, and start each
commit message with its type (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `ci:`,
or `chore:`).

Expected contribution practices:

- keep changes small and targeted;
- use NumPy-style docstrings for public APIs;
- update the relevant documentation;
- add or update tests for every change of behavior;
- add one line to the top section of the changelog
  (`docs/source/development/changelog.rst`), starting with a tag such as `[FEATURE]`,
  `[ENHANCEMENT]`, `[BUGFIX]`, or `[BREAKING]`.

## Tests and linting

Run the tests and the linter before opening a pull request:

```bash
python -m pytest tests/
python -m pytest tests/ --cov=calibrain
ruff check .
```

Tests that need the example data are skipped when it is not available. Tests that
download the real data only run with `python -m pytest --network`.

## Documentation

Build the documentation locally, with warnings as errors as on Read the Docs:

```bash
cd docs
make html SPHINXOPTS="-W --keep-going"
```

The build runs the tutorials, so a failing tutorial fails the build.

## Pull requests

Open pull requests against the `dev` branch. In your pull request, describe:

- what changed;
- why it changed;
- any user-facing impact;
- any test or documentation updates.

## Code of conduct

By participating in this project, you agree to follow the repository
`CODE_OF_CONDUCT.md`.
