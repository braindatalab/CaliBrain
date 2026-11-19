{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods if item != "__init__" %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
{% if fullname == "calibrain.SourceSimulator" %}

.. rubric:: Examples using ``{{ fullname }}``

- :doc:`/auto_tutorials/04_source_simulation`
- :doc:`/auto_tutorials/14_end_to_end_workflow`

{% elif fullname == "calibrain.LeadfieldBuilder" %}

.. rubric:: Examples using ``{{ fullname }}``

- :doc:`/auto_tutorials/05_preparing_the_example_data`
- :doc:`/auto_tutorials/06_leadfield_building`
- :doc:`/auto_tutorials/14_end_to_end_workflow`

{% elif fullname == "calibrain.SensorSimulator" %}

.. rubric:: Examples using ``{{ fullname }}``

- :doc:`/auto_tutorials/07_sensor_simulation`
- :doc:`/auto_tutorials/14_end_to_end_workflow`

{% elif fullname == "calibrain.DataGenerator" %}

.. rubric:: Examples using ``{{ fullname }}``

- :doc:`/auto_tutorials/08_data_generator`

{% elif fullname == "calibrain.SourceEstimator" %}

.. rubric:: Examples using ``{{ fullname }}``

- :doc:`/auto_tutorials/09_source_estimation`
- :doc:`/auto_tutorials/14_end_to_end_workflow`

{% elif fullname == "calibrain.UncertaintyEstimator" %}

.. rubric:: Examples using ``{{ fullname }}``

- :doc:`/auto_tutorials/03_quick_start`
- :doc:`/auto_tutorials/10_uncertainty_estimation`
- :doc:`/auto_tutorials/14_end_to_end_workflow`
- :doc:`/auto_tutorials/11_orientation_and_uncertainty_representations`

{% elif fullname == "calibrain.UncertaintyCalibrator" %}

.. rubric:: Examples using ``{{ fullname }}``

- :doc:`/auto_tutorials/12_uncertainty_calibration`
- :doc:`/auto_tutorials/14_end_to_end_workflow`

{% elif fullname == "calibrain.MetricEvaluator" %}

.. rubric:: Examples using ``{{ fullname }}``

- :doc:`/auto_tutorials/13_metric_evaluation`
- :doc:`/auto_tutorials/14_end_to_end_workflow`

{% endif %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
