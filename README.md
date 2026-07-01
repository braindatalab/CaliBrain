# CaliBrain

A Python toolbox for uncertainty estimation and calibration workflows in EEG/MEG inverse source imaging.

## Overview

CaliBrain addresses a specific reliability problem in EEG/MEG inverse source
imaging: a posterior estimate is only useful if its uncertainty is
well-calibrated. The toolbox provides simulation-based workflows for generating
source activity, propagating it through forward models, reconstructing
posterior source estimates, quantifying empirical coverage, and evaluating
recalibration maps under controlled experimental conditions.

## Documentation

[![Tests](https://github.com/braindatalab/CaliBrain/actions/workflows/tests.yml/badge.svg)](https://github.com/braindatalab/CaliBrain/actions/workflows/tests.yml)

The documentation is hosted on Read the Docs:
https://calibrain.readthedocs.io/

For runnable end-to-end examples, see the tutorials and workflow
documentation on Read the Docs.

## Contributing

Contribution guidelines are available in `CONTRIBUTING.md`.
The full development guide is also available in the documentation.

## Citation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20721580.svg)](https://doi.org/10.5281/zenodo.20721580)

If you use CaliBrain in academic work, please cite the software archive:

`Orabe, Mohammad, Huseynov, Ismail T., Nagarajan, Srikantan, & Haufe, Stefan. (2026). CaliBrain: Python toolbox for uncertainty estimation and calibration workflows in EEG/MEG inverse source imaging (v1.0.2). Zenodo. https://doi.org/10.5281/zenodo.20721580`

## Workflow

The package follows this workflow:

1. generate source-level ground truth under controlled sparsity and amplitude assumptions;
2. project sources to sensors through a leadfield and add noise at defined SNR;
3. reconstruct posterior means and uncertainty summaries with inverse solvers;
4. convert uncertainty summaries into intervals, ellipses, or ellipsoids;
5. compare empirical against nominal coverage;
6. fit isotonic recalibration functions on training splits and evaluate them on held-out splits.

CaliBrain currently supports fixed and free-orientation source models for inverse source imaging methods:
- `gamma_map_sflex` for Gamma-MAP reconstruction with sparse basis field expansions;
- `gamma_lambda_map_sflex` for the S-FLEX Gamma-MAP variant with joint sparsity and lambda regularization;
- `BMN` as a Bayesian minimum norm baseline;
- `BMN_joint` as a Bayesian minimum norm variant with joint gamma/lambda learning.

## Relationship to related software

CaliBrain complements broader neurophysiology analysis libraries, general
uncertainty-calibration toolkits, and standard inverse-solver workflows rather
than replacing them.

Its scope is narrower and more specific: CaliBrain focuses on simulation-based
uncertainty estimation and calibration for EEG/MEG inverse source imaging,
including source-level intervals, local covariance-based ellipsoids, empirical
coverage analysis, and recalibration across controlled evaluation conditions.

## Installation

[![PyPI version](https://img.shields.io/pypi/v/calibrain.svg)](https://pypi.org/project/calibrain/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/calibrain.svg)](https://pypi.org/project/calibrain/)
[![Latest GitHub release](https://img.shields.io/github/v/release/braindatalab/CaliBrain)](https://github.com/braindatalab/CaliBrain/releases)

From PyPI:

```bash
python -m pip install calibrain
```

From a local checkout:

```bash
git clone https://github.com/braindatalab/CaliBrain.git
cd CaliBrain
python -m pip install -e .
```

## License

[![License](https://img.shields.io/github/license/braindatalab/CaliBrain)](https://github.com/braindatalab/CaliBrain/blob/main/LICENSE)

CaliBrain is distributed under the BSD 3-Clause License. See `LICENSE`.
