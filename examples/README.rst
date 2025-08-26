.. _examples:

==========================
Examples
==========================

This gallery contains practical examples demonstrating real-world applications 
of CaliBrain for brain source localization and uncertainty quantification.

These examples show complete workflows that you can adapt for your own research 
projects. Each example is a standalone script that can be run independently.

**Quick Start**: If you're new to CaliBrain, start with the benchmarking experiment 
to see how different algorithms compare on simulated data.

🔬 **Interactive Gallery**: All examples are automatically processed by Sphinx Gallery and converted to HTML with embedded plots and downloadable notebooks.

Available Examples
==================

All examples in this directory are automatically processed and available in multiple formats:

**Benchmarking Experiments** (``run_experiments.py``)
   Comprehensive comparison of different inverse methods:
   
   - Multiple algorithms (Gamma-MAP, eLORETA, Bayesian Minimum Norm)
   - Various noise conditions and source configurations
   - Performance metrics and calibration analysis
   - Automated result visualization

**Leadfield Simulation** (``leadfield_simulation_example.py``)
   Step-by-step leadfield matrix computation:
   
   - Forward model setup and configuration
   - Source space and BEM model creation
   - Leadfield matrix extraction and validation

Example Features
================

Each example includes:

- **Python script** (`.py`) - Complete working example in this directory
- **HTML documentation** - Generated automatically with embedded plots
- **Jupyter notebook** (`.ipynb`) - Available for download and interactive exploration
- **Download options** - Get the source code and modify for your needs

Running Examples
================

Each example can be executed directly from the command line::

    cd examples/
    python run_experiments.py

All scripts in this directory are automatically converted to beautiful HTML documentation with embedded plots and made available as downloadable Jupyter notebooks through Sphinx Gallery.

Prerequisites
=============

Make sure CaliBrain is installed before running examples::

    pip install calibrain

See the installation guide for detailed setup instructions.