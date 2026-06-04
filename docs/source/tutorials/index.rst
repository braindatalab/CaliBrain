Tutorials
=========


The tutorial section is organized as a scientific learning path rather than as
a flat list of scripts. It starts with uncertainty-calibration concepts, then
builds toward simulation, inverse modeling, uncertainty estimation,
calibration, visualization, benchmarking, and extension.

Tutorials should be narrative and reproducible. Each tutorial starts with a
scientific motivation, defines the relevant uncertainty or calibration concept,
uses realistic lightweight data when possible, includes runnable code, explains
the outputs, and avoids obsolete APIs.


.. toctree::
   :maxdepth: 2
   :caption: Tutorial collections
   
   01. Introduction <introduction/index>
   02. Scientific Background <scientific_background>
   03. Data Model and Storage <data_model>
   04. Workflows <workflows>
   05. Uncertainty Calibration <uncertainty_calibration>
   06. Calibration Methods <calibration_methods/index>
   07. Evaluation and Visualization <evaluation_visualization>
   08. Simulation and Data Generation <simulation_data_generation/index>
   09. Inverse Modeling <inverse_modeling/index>
   10. Uncertainty Estimation <uncertainty_estimation/index>
   11. Calibration and Metrics <calibration_metrics/index>
   12. Visualization <visualization/index>
   13. Benchmarking <benchmarking/index>
   14. Advanced Topics <advanced_topics/index>
   15. Executable Tutorial Gallery <../auto_tutorials/index>


Design principles
-----------------


- **Progressive order**: concepts and data structures precede full workflows.
- **Executable where useful**: small tutorials should run as gallery examples;
  paper-scale tutorials should provide reproducible configs and expected
  artifacts instead of silently launching large jobs.
- **Scientific interpretation**: every tutorial should explain what the
  uncertainty or calibration result means, not only how to call the API.
- **Artifact transparency**: workflow tutorials must state input files, output
  files, storage formats, and downstream consumers.
- **Current APIs only**: tutorials should use ``SourceSimulator``,
  ``SensorSimulator``, solver functions, ``UncertaintyEstimator``,
  ``UncertaintyCalibrator``, ``MetricEvaluator``, workflow entrypoints, and current
  manifest/NPZ/JSON storage helpers.
