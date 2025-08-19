Installation
============

You can install **CaliBrain** using either **pip** or **Conda**, depending on your preferences.

We provide the following installation options:

- `pyproject.toml`: Standard modern pip-based installation (recommended).

- `requirements.txt`: Simple pip-based installation (optional).

- `environment.yml`: Conda-based environment installation (optional).

Choose the method that best fits your workflow.


Which method should I use?
---------------------------

- **Recommended**: Use pip with `pyproject.toml` for clean dependency management (`pip install .`).
- **If you prefer Conda**: Use `environment.yml` to create a Conda environment first.
- **If you just want quick pip install**: Use `requirements.txt`.

All methods lead to the same installed package — just choose the method that matches your ecosystem (pip-only or Conda).

----

Minimum Requirements
---------------------

- Python >= 3.8
- Tested on Python 3.8, 3.9, 3.10
- Operating systems: Linux, macOS, Windows (WSL recommended for full compatibility)

----

.. toctree::
   :maxdepth: 2
   :hidden:

   pip_installation
   conda_installation