Contributing
============

CaliBrain is maintained by a community of scientists and research labs.
We welcome contributions in many forms, including:

- Bug reports and fixes
- Feature requests and additions
- Code improvements
- Documentation enhancements

Getting Started
---------------

- Open an **Issue** on GitHub to propose a change or report a bug.
- For general usage questions, use **GitHub Discussions**.
- Please follow our **Code of Conduct**.

How to Contribute
-----------------

1. **Fork the Repository**
   
   Go to the `CaliBrain` GitHub page and click **Fork** to create your own copy.

2. **Set up your Development Environment**
   
   Clone your fork:
   
   .. code-block:: bash
   
      git clone https://github.com/your-username/CaliBrain.git
      cd CaliBrain

   (Optional) Create and activate a virtual environment:
   
   .. code-block:: bash
   
      conda create -n calibrain-dev python=3.9 -y
      conda activate calibrain-dev

   Install the package in editable mode:
   
   .. code-block:: bash
   
      pip install -e .[dev]

3. **Create a New Branch**
   
   Create a branch for your feature or bug fix:
   
   .. code-block:: bash
   
      git checkout -b feature/your-feature-name

4. **Make Your Changes**
   
   - Implement your feature or bug fix.
   - Follow the existing code style (PEP8 and NumPy-style docstrings).
   - Add or update documentation and tests where necessary.

5. **Test Your Changes**
   
   - Make sure the code runs correctly.
   - Add unit tests if appropriate.

6. **Commit and Push**
   
   Write clear and descriptive commit messages.
   Push your branch to your GitHub fork:
   
   .. code-block:: bash
   
      git push origin feature/your-feature-name

7. **Submit a Pull Request**
   
   - Go to the original repository.
   - Open a Pull Request (PR) from your branch.
   - Fill in the PR template and describe your changes clearly.

Code Style and Quality
----------------------

- Follow `PEP8` coding standards.
- Use meaningful variable and function names.
- Write docstrings for all public functions and classes using **NumPy docstring format**.
- Add type hints where appropriate.

Development Setup
-----------------

For a complete development setup:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/braindatalab/CaliBrain.git
   cd CaliBrain
   
   # Create development environment
   conda create -n calibrain-dev python=3.9 -y
   conda activate calibrain-dev
   
   # Install in development mode with all dependencies
   pip install -e ".[dev,docs]"
   
   # Install pre-commit hooks (optional)
   pre-commit install

Running Tests
-------------

.. code-block:: bash

   # Run all tests
   pytest tests/
   
   # Run with coverage
   pytest tests/ --cov=calibrain

Building Documentation
----------------------

.. code-block:: bash

   # Build documentation
   cd docs
   make clean
   make html
   
   # Or use the build script
   ./build_docs.sh

Code Review Guidelines
----------------------

When reviewing pull requests, we look for:

- **Functionality**: Does the code work as intended?
- **Code Quality**: Is the code readable and well-structured?
- **Documentation**: Are docstrings and comments adequate?
- **Tests**: Are there appropriate tests for new functionality?
- **Compatibility**: Does it work with supported Python versions?

Thank You!
----------

Thank you for contributing to CaliBrain! Your efforts help make this tool better for the entire neuroimaging community.