Tutorials
=========

The tutorials page is the main runnable learning surface of the documentation.
It follows a deliberate progression:

1. scientific motivation and calibration concepts;
2. source, leadfield, and sensor simulation;
3. inverse source estimation;
4. uncertainty estimation and calibration;
5. metric evaluation, end-to-end calibration, and workflow orchestration.

The first tutorials therefore form a coherent path from simulation inputs to
calibration outputs before introducing the higher-level ``DataGenerator``
workflow wrapper.

.. raw:: html

  <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Scientific motivation, calibration concepts, and the minimal objects needed for an empirical coverage curve.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_01_what_is_calibrain_thumb.png
    :alt:

  :doc:`/auto_tutorials/01_what_is_calibrain`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">01. Introduction</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Source-level simulation for fixed orientation, free-orientation EEG, and free-orientation MEG.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_02_source_simulation_thumb.png
    :alt:

  :doc:`/auto_tutorials/02_source_simulation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">02. Source Simulation</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Leadfield retrieval in random, simulate, and load modes, with unit-aware interpretation.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_03_leadfield_building_thumb.png
    :alt:

  :doc:`/auto_tutorials/03_leadfield_building`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">03. Leadfield Construction</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Projection from source activity to noisy sensor measurements and the workflow noise-variance modes used downstream.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_04_sensor_simulation_thumb.png
    :alt:

  :doc:`/auto_tutorials/04_sensor_simulation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">04. Sensor Simulation</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Current active inverse solvers: gamma_map_sflex, gamma_lambda_map_sflex, BMN, and BMN_joint.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_05_source_estimation_thumb.png
    :alt:

  :doc:`/auto_tutorials/05_source_estimation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">05. Source Estimation</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Convert posterior means and covariance summaries into calibration-ready marginal intervals and full-covariance ellipsoids.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_06_uncertainty_estimation_thumb.png
    :alt:

  :doc:`/auto_tutorials/06_uncertainty_estimation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">06. Uncertainty Estimation</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Calibration modes and isotonic recalibration using the high-level UncertaintyCalibrator API.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_07_uncertainty_calibration_thumb.png
    :alt:

  :doc:`/auto_tutorials/07_uncertainty_calibration`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">07. Calibration Methods</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Quantitative evaluation of calibration curves, summary metrics, source error, and posterior uncertainty.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_08_metric_evaluation_thumb.png
    :alt:

  :doc:`/auto_tutorials/08_metric_evaluation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">08. Metric Evaluation</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A full runnable chain from source simulation through leadfield building, sensor simulation, source estimation, uncertainty estimation, and post-calibration evaluation.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_09_end_to_end_workflow_thumb.png
    :alt:

  :doc:`/auto_tutorials/09_end_to_end_workflow`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">09. End-to-End Workflow</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="High-level workflow orchestration across source simulation, leadfield loading, sensor simulation, source estimation, and posterior-summary persistence.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_10_data_generator_thumb.png
    :alt:

  :doc:`/auto_tutorials/10_data_generator`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">10. Data Generation</div>
    </div>

.. thumbnail-parent-div-close

.. raw:: html

    </div>

.. toctree::
  :hidden:

  /auto_tutorials/01_what_is_calibrain
  /auto_tutorials/02_source_simulation
  /auto_tutorials/03_leadfield_building
  /auto_tutorials/04_sensor_simulation
  /auto_tutorials/05_source_estimation
  /auto_tutorials/06_uncertainty_estimation
  /auto_tutorials/07_uncertainty_calibration
  /auto_tutorials/08_metric_evaluation
  /auto_tutorials/09_end_to_end_workflow
  /auto_tutorials/10_data_generator
