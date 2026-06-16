Tutorials
=========

This tutorial gallery introduces the main CaliBrain workflows step by step.
Each tutorial combines short explanation with runnable code and focuses on one
part of the package at a time.

The tutorials are ordered progressively. They begin with the uncertainty
calibration problem itself, then move through simulation, source estimation,
uncertainty estimation, calibration, evaluation, and finally the combined
workflow.

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

    <div class="sphx-glr-thumbcontainer" tooltip="Defines the uncertainty-calibration problem and constructs the minimal objects needed for an empirical coverage curve.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_01_quick_start_thumb.png
    :alt:

  :doc:`/auto_tutorials/01_quick_start`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">01. Quick Start</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Uses SourceSimulator to generate sparse ERP-like source activity for fixed orientation and free-orientation EEG/MEG settings.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_02_source_simulation_thumb.png
    :alt:

  :doc:`/auto_tutorials/02_source_simulation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">02. Source Simulation</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Uses LeadfieldBuilder in random, simulate, and load modes and explains how source and sensor units relate through the forward model.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_03_leadfield_building_thumb.png
    :alt:

  :doc:`/auto_tutorials/03_leadfield_building`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">03. Leadfield Construction</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Uses SensorSimulator to project sources to sensors, add Gaussian noise, and interpret the oracle, baseline, and adaptive joint-learning noise modes.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_04_sensor_simulation_thumb.png
    :alt:

  :doc:`/auto_tutorials/04_sensor_simulation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">04. Sensor Simulation</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Uses DataGenerator as the batch orchestration layer across simulation, leadfield loading, sensor generation, source estimation, and run-wise metadata collection.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_05_data_generator_thumb.png
    :alt:

  :doc:`/auto_tutorials/05_data_generator`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">05. Data Generation</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Uses SourceEstimator with the currently active inverse solvers: gamma_map_sflex, gamma_lambda_map_sflex, BMN, and BMN_joint.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_06_source_estimation_thumb.png
    :alt:

  :doc:`/auto_tutorials/06_source_estimation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">06. Source Estimation</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Uses UncertaintyEstimator to derive fixed-orientation variances, free-orientation marginal intervals, and free-orientation full_cov ellipsoids.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_07_uncertainty_estimation_thumb.png
    :alt:

  :doc:`/auto_tutorials/07_uncertainty_estimation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">07. Uncertainty Estimation</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Uses UncertaintyCalibrator to explain precal, post_oracle, post_pooled, post_pooled_mismatch, and post_fixed, with runnable matched-split examples.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_08_uncertainty_calibration_thumb.png
    :alt:

  :doc:`/auto_tutorials/08_uncertainty_calibration`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">08. Calibration Methods</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Uses MetricEvaluator to compute calibration curves, summary calibration metrics, source error summaries, and uncertainty summaries.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_09_metric_evaluation_thumb.png
    :alt:

  :doc:`/auto_tutorials/09_metric_evaluation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">09. Metric Evaluation</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Runs a complete in-memory workflow from SourceSimulator through UncertaintyCalibrator, including pre- and post-calibration evaluation.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_10_end_to_end_workflow_thumb.png
    :alt:

  :doc:`/auto_tutorials/10_end_to_end_workflow`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">10. End-to-End Workflow</div>
    </div>

.. thumbnail-parent-div-close

.. raw:: html

    </div>

.. toctree::
  :hidden:

  /auto_tutorials/00_motivation
  /auto_tutorials/01_quick_start
  /auto_tutorials/02_source_simulation
  /auto_tutorials/03_leadfield_building
  /auto_tutorials/04_sensor_simulation
  /auto_tutorials/05_data_generator
  /auto_tutorials/06_source_estimation
  /auto_tutorials/07_uncertainty_estimation
  /auto_tutorials/08_uncertainty_calibration
  /auto_tutorials/09_metric_evaluation
  /auto_tutorials/10_end_to_end_workflow
