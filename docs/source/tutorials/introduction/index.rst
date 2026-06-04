Introduction
============


Introductory tutorials establish the scientific problem, core terminology, and
artifact model used throughout CaliBrain.


.. toctree::
   :maxdepth: 1
   :caption: Available tutorials
   
   01. What is CaliBrain? <01_what_is_calibrain>


Planned tutorials
-----------------


The remaining introductory tutorials are still specifications and will be
implemented incrementally.

Core Concepts in Uncertainty Calibration
----------------------------------------


- **Description**: Define nominal coverage, empirical coverage, credible
  intervals, calibration curves, and perfect calibration.
- **Target audience**: Users who need to interpret calibration curves.
- **Difficulty**: Beginner.
- **Estimated duration**: 20 minutes.
- **Prerequisites**: What is CaliBrain?
- **Main APIs introduced**: ``UncertaintyEstimator``,
  ``MetricEvaluator.calibration_metrics_4``.
- **Scientific concepts**: Coverage calibration, underconfidence,
  overconfidence, held-out evaluation.
- **Expected outputs**: Small calibration curve from synthetic arrays.
- **Figures or plots**: Nominal-versus-empirical coverage curve with the
  ``perfect calibration`` diagonal.
- **Gallery executable**: Yes.

First End-to-End Calibration Workflow
-------------------------------------


- **Description**: Run a minimal fixed-orientation workflow from data generation
  through aggregation and calibration using lightweight settings.
- **Target audience**: New users who want the first complete run.
- **Difficulty**: Beginner to intermediate.
- **Estimated duration**: 30--60 minutes, depending on leadfield availability.
- **Prerequisites**: Access to configured leadfield data or a documented
  synthetic fallback.
- **Main APIs introduced**: ``run_data_generation``, ``aggregate_posteriors``,
  ``run_calibration``, ``load_manifest_csv``.
- **Scientific concepts**: Train/eval split, isotonic recalibration,
  pre-calibration versus post-calibration.
- **Expected outputs**: H5 posterior summaries, manifest CSV, aggregated NPZ
  files, calibration JSON files, optional calibration figures.
- **Figures or plots**: Pre- and post-calibration curves.
- **Gallery executable**: Yes, if configured with synthetic or small local data;
  otherwise documentation-only with explicit data requirements.

Understanding CaliBrain Data Structures
---------------------------------------


- **Description**: Explain the storage contracts: posterior H5 summary,
  manifest CSV, aggregated NPZ, JSON sidecar, and calibration JSON.
- **Target audience**: Workflow users and contributors.
- **Difficulty**: Beginner.
- **Estimated duration**: 20 minutes.
- **Prerequisites**: First End-to-End Calibration Workflow.
- **Main APIs introduced**: ``ManifestRow``, ``load_manifest_csv``,
  ``PosteriorSummary``, ``load_posterior_summary``,
  ``summaries_from_manifest``.
- **Scientific concepts**: Reproducible metadata, raw versus reduced
  uncertainty representation.
- **Expected outputs**: Printed schema summaries and example metadata rows.
- **Figures or plots**: Artifact-flow diagram.
- **Gallery executable**: Yes.
