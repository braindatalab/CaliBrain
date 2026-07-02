{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autofunction:: {{ objname }}
{% if fullname == "calibrain.gamma_map_sflex" %}

.. rubric:: Examples using ``{{ fullname }}``

- :doc:`/auto_tutorials/08_source_estimation`

{% elif fullname == "calibrain.gamma_lambda_map_sflex" %}

.. rubric:: Examples using ``{{ fullname }}``

- :doc:`/auto_tutorials/08_source_estimation`

{% elif fullname == "calibrain.BMN" %}

.. rubric:: Examples using ``{{ fullname }}``

- :doc:`/auto_tutorials/08_source_estimation`

{% elif fullname == "calibrain.BMN_joint" %}

.. rubric:: Examples using ``{{ fullname }}``

- :doc:`/auto_tutorials/08_source_estimation`

{% elif fullname == "calibrain.workflows.data_generation.run_data_generation" %}

.. rubric:: Examples using ``{{ fullname }}``

- :doc:`/auto_tutorials/07_data_generator`

{% elif fullname == "calibrain.workflows.aggregation.aggregate_posteriors" %}

.. rubric:: Examples using ``{{ fullname }}``

- :doc:`/auto_tutorials/12_end_to_end_workflow`

{% elif fullname == "calibrain.workflows.calibration.run_calibration" %}

.. rubric:: Examples using ``{{ fullname }}``

- :doc:`/auto_tutorials/10_uncertainty_calibration`
- :doc:`/auto_tutorials/12_end_to_end_workflow`

{% elif fullname == "calibrain.workflows.calibration.build_uncertainty_components" %}

.. rubric:: Examples using ``{{ fullname }}``

- :doc:`/auto_tutorials/09_uncertainty_estimation`

{% endif %}
