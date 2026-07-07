Tutorials
=========

This tutorial gallery introduces the main CaliBrain workflows step by step.
Each tutorial combines short explanation with runnable code and focuses on one
part of the toolbox at a time.

The tutorials are grouped by topic and arranged in a recommended reading order.

.. raw:: html

  <h2 class="tutorial-gallery-section">Foundations</h2>

These tutorials introduce the scientific problem, the mathematical framework,
and the main solver and calibration ideas used throughout the package.

.. raw:: html

  <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Provides the theoretical background for the forward model, posterior uncertainty, empirical coverage, and isotonic recalibration used throughout the documentation.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_00_motivation_thumb.png
    :alt:

  :doc:`/auto_tutorials/00_motivation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">00. Motivation</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Introduces the orientation-aware forward model, temporally aggregated posterior summaries, local Gaussian marginals, credible-region geometry, empirical coverage, and isotonic recalibration.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_01_theoretical_foundations_thumb.png
    :alt:

  :doc:`/auto_tutorials/01_theoretical_foundations`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">01. Theoretical Foundations</div>
    </div>

    <div class="sphx-glr-thumbcontainer" tooltip="Explains how dense and sparse Bayesian inverse solvers differ in posterior uncertainty behavior, variance collapse, and calibration relevance.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_02_solver_families_and_uncertainty_behavior_thumb.png
    :alt:

  :doc:`/auto_tutorials/02_solver_families_and_uncertainty_behavior`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">02. Solver Families and Uncertainty Behavior</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Defines the uncertainty-calibration problem and constructs the minimal objects needed for an empirical coverage curve.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_03_quick_start_thumb.png
    :alt:

  :doc:`/auto_tutorials/03_quick_start`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">03. Quick Start</div>
    </div>

.. thumbnail-parent-div-close

.. raw:: html

    </div>

.. raw:: html

  <h2 class="tutorial-gallery-section">Simulation and Benchmark Setup</h2>

These tutorials cover the components used to generate repeated synthetic runs
for later solver comparison and calibration analysis.

.. raw:: html

  <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Uses SourceSimulator to generate sparse ERP-like source activity for fixed orientation and free-orientation EEG/MEG settings.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_04_source_simulation_thumb.png
    :alt:

  :doc:`/auto_tutorials/04_source_simulation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">04. Source Simulation</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Uses LeadfieldBuilder in random, simulate, and load modes and explains how source and sensor units relate through the forward model.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_05_leadfield_building_thumb.png
    :alt:

  :doc:`/auto_tutorials/05_leadfield_building`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">05. Leadfield Construction</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Uses SensorSimulator to project sources to sensors, add Gaussian noise, and interpret the oracle, baseline, and adaptive joint-learning noise modes.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_06_sensor_simulation_thumb.png
    :alt:

  :doc:`/auto_tutorials/06_sensor_simulation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">06. Sensor Simulation</div>
    </div>

    <div class="sphx-glr-thumbcontainer" tooltip="Uses DataGenerator as the batch orchestration layer across simulation, leadfield loading, sensor generation, source estimation, and run-wise metadata collection.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_07_data_generator_thumb.png
    :alt:

  :doc:`/auto_tutorials/07_data_generator`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">07. Data Generation</div>
    </div>

.. thumbnail-parent-div-close

.. raw:: html

    </div>

.. raw:: html

  <h2 class="tutorial-gallery-section">Inference and Uncertainty</h2>

These tutorials explain how posterior summaries are produced and how they are
converted into the uncertainty objects that CaliBrain evaluates.

.. raw:: html

  <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Uses SourceEstimator with the currently active inverse solvers: gamma_map_sflex, gamma_lambda_map_sflex, BMN, and BMN_joint.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_08_source_estimation_thumb.png
    :alt:

  :doc:`/auto_tutorials/08_source_estimation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">08. Source Estimation</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Uses UncertaintyEstimator to derive fixed-orientation variances, free-orientation marginal intervals, and free-orientation full_cov ellipsoids.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_09_uncertainty_estimation_thumb.png
    :alt:

  :doc:`/auto_tutorials/09_uncertainty_estimation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">09. Uncertainty Estimation</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compares fixed, reduced free-orientation MEG, and free-orientation EEG uncertainty representations, including marginal versus full_cov calibration diagnostics.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_13_orientation_and_uncertainty_representations_thumb.png
    :alt:

  :doc:`/auto_tutorials/13_orientation_and_uncertainty_representations`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">13. Orientation and Uncertainty Representations</div>
    </div>

.. thumbnail-parent-div-close

.. raw:: html

    </div>

.. raw:: html

  <h2 class="tutorial-gallery-section">Calibration and Evaluation</h2>

These tutorials show how empirical coverage curves are recalibrated and how
calibration behavior is summarized quantitatively.

.. raw:: html

  <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Uses UncertaintyCalibrator to explain precal, post_oracle, post_pooled, post_pooled_mismatch, and post_fixed, with runnable matched-split examples.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_10_uncertainty_calibration_thumb.png
    :alt:

  :doc:`/auto_tutorials/10_uncertainty_calibration`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">10. Calibration Methods</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Uses MetricEvaluator to compute calibration curves, summary calibration metrics, source error summaries, and uncertainty summaries.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_11_metric_evaluation_thumb.png
    :alt:

  :doc:`/auto_tutorials/11_metric_evaluation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">11. Metric Evaluation</div>
    </div>

.. thumbnail-parent-div-close

.. raw:: html

    </div>

.. raw:: html

  <h2 class="tutorial-gallery-section">Integrated Workflow</h2>

This tutorial combines the main stages into one compact runnable workflow.

.. raw:: html

  <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Runs a complete in-memory workflow from SourceSimulator through UncertaintyCalibrator, including pre- and post-calibration evaluation.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_12_end_to_end_workflow_thumb.png
    :alt:

  :doc:`/auto_tutorials/12_end_to_end_workflow`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">12. End-to-End Workflow</div>
    </div>

.. thumbnail-parent-div-close

.. raw:: html

    </div>

.. toctree::
  :hidden:

  /auto_tutorials/00_motivation
  /auto_tutorials/01_theoretical_foundations
  /auto_tutorials/02_solver_families_and_uncertainty_behavior
  /auto_tutorials/03_quick_start
  /auto_tutorials/04_source_simulation
  /auto_tutorials/05_leadfield_building
  /auto_tutorials/06_sensor_simulation
  /auto_tutorials/07_data_generator
  /auto_tutorials/08_source_estimation
  /auto_tutorials/09_uncertainty_estimation
  /auto_tutorials/10_uncertainty_calibration
  /auto_tutorials/11_metric_evaluation
  /auto_tutorials/12_end_to_end_workflow
  /auto_tutorials/13_orientation_and_uncertainty_representations
