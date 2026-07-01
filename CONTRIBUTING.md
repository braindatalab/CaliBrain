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

Create and activate an isolated environment:

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

Create a branch for your work:

```bash
git checkout -b feature/short-description
```

Expected contribution practices:

- keep changes small and targeted;
- use clear names and NumPy-style docstrings for public APIs;
- update relevant documentation;
- add or update tests when practical.

## Tests

When the test suite is present, run the relevant tests before opening a pull
request:

```bash
pytest tests/
pytest tests/ --cov=calibrain
```

## Documentation

Build the documentation locally with:

```bash
cd docs
make html
```

## Pull requests

In your pull request, describe:

- what changed;
- why it changed;
- any user-facing impact;
- any test or documentation updates.

## Code of conduct

By participating in this project, you agree to follow the repository
`CODE_OF_CONDUCT.md`.
