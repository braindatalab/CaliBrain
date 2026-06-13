API Reference
=============

This page documents the public API exposed by ``calibrain`` and the workflow
entrypoints used by the current pipeline. Internal helpers are intentionally not
listed here unless they define a stable data contract.

Top-level public API
--------------------

.. currentmodule:: calibrain

.. autosummary::
   :toctree: generated
   :nosignatures:

   LeadfieldBuilder
   SourceSimulator
   SensorSimulator
   SourceEstimator
   UncertaintyEstimator
   UncertaintyCalibrator
   MetricEvaluator
   Visualizer
   DataGenerator
   gamma_map_sflex
   gamma_lambda_map_sflex
   BMN
   BMN_joint

.. toctree::
   :maxdepth: 1
   :caption: Top-level public API

   generated/calibrain.LeadfieldBuilder
   generated/calibrain.SourceSimulator
   generated/calibrain.SensorSimulator
   generated/calibrain.SourceEstimator
   generated/calibrain.UncertaintyEstimator
   generated/calibrain.UncertaintyCalibrator
   generated/calibrain.MetricEvaluator
   generated/calibrain.Visualizer
   generated/calibrain.DataGenerator
   generated/calibrain.gamma_map_sflex
   generated/calibrain.gamma_lambda_map_sflex
   generated/calibrain.BMN
   generated/calibrain.BMN_joint

Workflow entrypoints
--------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   workflows.data_generation.run_data_generation
   workflows.aggregation.aggregate_posteriors
   workflows.calibration.run_calibration
   workflows.calibration.build_uncertainty_components
   workflows.plot_paper_calibration_figures.main

.. toctree::
   :maxdepth: 1
   :caption: Workflow entrypoints

   generated/calibrain.workflows.data_generation.run_data_generation
   generated/calibrain.workflows.aggregation.aggregate_posteriors
   generated/calibrain.workflows.calibration.run_calibration
   generated/calibrain.workflows.calibration.build_uncertainty_components
   generated/calibrain.workflows.plot_paper_calibration_figures.main

Data model helpers
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   run_manifest.ManifestRow
   run_manifest.load_manifest_csv
   run_manifest.summaries_from_manifest
   calibration_dataset.PosteriorSummary
   calibration_dataset.load_posterior_summary
   calibration_dataset.filter_summaries_by_metadata
   calibration_dataset.concatenate_summaries
   calibration_storage.save_calibration_record
   calibration_storage.load_calibration_record
   calibration_storage.iter_calibration_records
   calibration_storage.stack_empirical_curves

.. toctree::
   :maxdepth: 1
   :caption: Data model helpers

   generated/calibrain.run_manifest.ManifestRow
   generated/calibrain.run_manifest.load_manifest_csv
   generated/calibrain.run_manifest.summaries_from_manifest
   generated/calibrain.calibration_dataset.PosteriorSummary
   generated/calibrain.calibration_dataset.load_posterior_summary
   generated/calibrain.calibration_dataset.filter_summaries_by_metadata
   generated/calibrain.calibration_dataset.concatenate_summaries
   generated/calibrain.calibration_storage.save_calibration_record
   generated/calibrain.calibration_storage.load_calibration_record
   generated/calibrain.calibration_storage.iter_calibration_records
   generated/calibrain.calibration_storage.stack_empirical_curves

Metric and utility helpers
--------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   metric_evaluation.get_subset_source_rr
   uncertainty_estimation.lift_reduced_sources_to_3d
   utils.get_data_path
   utils.load_config
   utils.load_python_config

.. toctree::
   :maxdepth: 1
   :caption: Metric and utility helpers

   generated/calibrain.metric_evaluation.get_subset_source_rr
   generated/calibrain.uncertainty_estimation.lift_reduced_sources_to_3d
   generated/calibrain.utils.get_data_path
   generated/calibrain.utils.load_config
   generated/calibrain.utils.load_python_config
