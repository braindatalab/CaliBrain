.. CaliBrain documentation master file

=========
CaliBrain
=========

.. image:: https://img.shields.io/pypi/v/calibrain.svg
   :target: https://pypi.org/project/calibrain/
   :alt: PyPI version

.. image:: https://readthedocs.org/projects/calibrain/badge/?version=latest
   :target: https://calibrain.readthedocs.io/en/latest/
   :alt: Documentation status

.. image:: https://img.shields.io/pypi/pyversions/calibrain.svg
   :target: https://pypi.org/project/calibrain/
   :alt: Supported Python versions

.. .. image:: https://static.pepy.tech/badge/calibrain
..    :target: https://pepy.tech/projects/calibrain
..    :alt: Total downloads

.. image:: https://img.shields.io/github/license/braindatalab/CaliBrain
   :target: https://github.com/braindatalab/CaliBrain/blob/main/LICENSE
   :alt: License

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.20703249.svg
   :target: https://doi.org/10.5281/zenodo.20703249
   :alt: DOI

A Python framework for uncertainty estimation and calibration in EEG/MEG inverse source imaging.

Overview
--------

CaliBrain addresses a specific reliability problem in EEG/MEG inverse source
imaging: a posterior estimate is only useful if its uncertainty is
well-calibrated. The package provides a simulation-based workflow for
generating source activity, propagating it through forward models, reconstructing
posterior source estimates, quantifying empirical coverage, and learning
recalibration maps from controlled experiments.

Documentation
-------------

The documentation is hosted on Read the Docs:
https://calibrain.readthedocs.io/

For runnable end-to-end examples, see the tutorials and workflow
documentation on Read the Docs.

Citation
--------

If you use CaliBrain in academic work, please cite the software archive:

``Orabe, Mohammad, Huseynov, Ismail T., Nagarajan, Srikantan, & Haufe, Stefan. (2026). CaliBrain: Python framework for uncertainty estimation and calibration in EEG/MEG inverse source imaging (v1.0.1). Zenodo. https://doi.org/10.5281/zenodo.20703249``

Workflow
--------

The package follows this workflow:

1. generate source-level ground truth under controlled sparsity and amplitude assumptions;
2. project sources to sensors through a leadfield and add noise at defined SNR;
3. reconstruct posterior means and uncertainty summaries with inverse solvers;
4. convert uncertainty summaries into intervals, ellipses, or ellipsoids;
5. compare empirical against nominal coverage;
6. fit isotonic recalibration functions on training splits and evaluate them on held-out splits.

CaliBrain currently supports fixed and free-orientation source models for inverse source imaging methods:

* ``gamma_map_sflex`` for Gamma-MAP reconstruction with sparse basis field expansions;
* ``gamma_lambda_map_sflex`` for the S-FLEX Gamma-MAP variant with joint sparsity and lambda regularization;
* ``BMN`` as a Bayesian minimum norm baseline;
* ``BMN_joint`` as a Bayesian minimum norm variant with joint gamma/lambda learning.

Installation
------------

From PyPI:

.. code-block:: bash

   python -m pip install calibrain

From a local checkout:

.. code-block:: bash

   git clone https://github.com/braindatalab/CaliBrain.git
   cd CaliBrain
   python -m pip install -e .

License
-------

CaliBrain is distributed under the BSD 3-Clause License. See ``LICENSE``.

.. toctree::
   :hidden:
   :maxdepth: 1

   Installation <installation/index>
   Documentation <documentation/index>
   API Reference <api_reference/index>
   Development <development/index>
