Data Model and Storage
======================


The current pipeline uses four main artifact types. These artifacts are designed
to keep simulation, aggregation, calibration, and plotting reproducible.


Posterior summaries
-------------------


Posterior summaries are HDF5 files written by the data-generation workflow.
Each file corresponds to one simulated run and stores:

- ``x_true``: simulated source activity.
- ``x_hat``: posterior mean estimate.
- ``posterior_cov``: full posterior covariance produced by the solver.
- ``Q_basis``: optional local basis for reduced MEG free-orientation models.
- ``metadata_json``: HDF5 attribute containing run metadata.

The full covariance is retained at this raw stage so the generation output is a
complete record of the solver result.

Manifest CSV
------------


The run manifest is a CSV file written by ``calibrain/workflows/data_generation.py``.
It records one row per run and includes the path to the posterior summary plus
metadata used for downstream filtering:

- ``global_run_id``, ``run_id``, ``seed``
- ``subject``, ``orientation_type``, ``coil_type``
- ``solver``, ``noise_type``
- ``alpha_SNR``, ``nnz``
- ``n_sources``, ``n_times``
- ``posterior_summary``

Aggregation reads from this manifest. The workflow intentionally avoids
filesystem scanning so that the set of eligible runs is explicit and auditable.

Aggregated calibration datasets
-------------------------------


Aggregation reads posterior summaries from the manifest and writes one
calibration dataset per run as compressed NPZ plus a JSON sidecar. The
aggregation stage reduces uncertainty storage:

- fixed orientation: stores ``posterior_var``, the diagonal of the covariance.
- free orientation: stores ``posterior_cov_blocks``, the per-source \(K \times K\)
  covariance blocks.

The full covariance is not stored in the aggregated NPZ. Calibration currently
uses marginal variances for fixed orientation and per-source blocks for
free-orientation ellipsoids or marginal componentwise intervals.

Calibration JSON summaries
--------------------------


The calibration workflow consumes aggregated NPZ datasets and writes JSON files
containing:

- ``train_sources``: training datasets used to fit the calibration map.
- ``eval_source``: evaluation dataset.
- ``pre_calibration``: raw empirical coverage curve and metrics.
- ``post_calibration``: recalibrated empirical coverage curve and metrics.
- ``emd``: optional Earth Mover's Distance metric when source coordinates are
  available.

These JSON files are the inputs to the paper-figure plotting workflow.
