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

- :doc:`/auto_tutorials/02_source_simulation`
- :doc:`/auto_tutorials/10_end_to_end_workflow`

{% elif fullname == "calibrain.LeadfieldBuilder" %}

.. rubric:: Examples using ``{{ fullname }}``

- :doc:`/auto_tutorials/03_leadfield_building`
- :doc:`/auto_tutorials/10_end_to_end_workflow`

{% elif fullname == "calibrain.SensorSimulator" %}

.. rubric:: Examples using ``{{ fullname }}``

- :doc:`/auto_tutorials/04_sensor_simulation`
- :doc:`/auto_tutorials/10_end_to_end_workflow`

{% elif fullname == "calibrain.DataGenerator" %}

.. rubric:: Examples using ``{{ fullname }}``

- :doc:`/auto_tutorials/05_data_generator`

{% elif fullname == "calibrain.SourceEstimator" %}

.. rubric:: Examples using ``{{ fullname }}``

- :doc:`/auto_tutorials/06_source_estimation`
- :doc:`/auto_tutorials/10_end_to_end_workflow`

{% elif fullname == "calibrain.UncertaintyEstimator" %}

.. rubric:: Examples using ``{{ fullname }}``

- :doc:`/auto_tutorials/01_quick_start`
- :doc:`/auto_tutorials/07_uncertainty_estimation`
- :doc:`/auto_tutorials/10_end_to_end_workflow`

{% elif fullname == "calibrain.UncertaintyCalibrator" %}

.. rubric:: Examples using ``{{ fullname }}``

- :doc:`/auto_tutorials/08_uncertainty_calibration`
- :doc:`/auto_tutorials/10_end_to_end_workflow`

{% elif fullname == "calibrain.MetricEvaluator" %}

.. rubric:: Examples using ``{{ fullname }}``

- :doc:`/auto_tutorials/09_metric_evaluation`
- :doc:`/auto_tutorials/10_end_to_end_workflow`

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
