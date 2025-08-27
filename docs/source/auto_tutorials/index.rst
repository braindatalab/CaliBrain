:orphan:

.. _tutorials:

==========================
Tutorials
==========================

Here you will find step-by-step guides and practical examples to help you get started with CaliBrain. These tutorials are designed for users of all experience levels and cover a range of topics, from basic setup to advanced features.

Explore the tutorials below to enhance your understanding and make the most of CaliBrain.

📚 **Interactive Gallery**: Check out the auto-generated tutorial gallery with all scripts converted to HTML with embedded plots and downloadable notebooks.

Available Tutorials
===================

All tutorials in this directory are automatically processed by Sphinx Gallery and available in multiple formats:

**Overview Tutorial** (``overview.py``)
   Complete introduction to CaliBrain's modular architecture and workflow:
   
   - Building forward models with LeadfieldBuilder
   - Simulating brain source activity and sensor measurements  
   - Solving inverse problems with multiple methods
   - Quantifying uncertainty and evaluating performance
   - Automated benchmarking workflows

Tutorial Features
=================

Each tutorial includes:

- **Python script** (`.py`) - Can be run directly from this directory
- **HTML documentation** - Generated automatically with embedded plots
- **Jupyter notebook** (`.ipynb`) - Available for download and interactive exploration
- **Download options** - Get the source code and data

Running Tutorials
=================

To run a tutorial script directly::

    cd tutorials/
    python overview.py

All scripts in this directory are automatically converted to beautiful HTML documentation with embedded plots and made available as downloadable Jupyter notebooks through Sphinx Gallery.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This tutorial demonstrates how to use the SensorSimulator class to generate synthetic MEG/EEG sensor measurements from brain source activity. The SensorSimulator  projects source-level neural signals to sensor space using forward models and adds realistic noise, creating controlled datasets for testing source localization algorithms, validating analysis  pipelines, and benchmarking uncertainty quantification methods.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_sensor_simulation_tutorial_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_sensor_simulation_tutorial.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Sensor Data Simulation with SensorSimulator</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This tutorial demonstrates how to use the SourceSimulator class to generate synthetic brain activity data for neuroimaging research (e.g., MEG/EEG  source simulation). The SourceSimulator creates event-related potential (ERP)-like  signals that can be used to test source localization algorithms, validate analysis  pipelines, and benchmark uncertainty quantification methods.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_source_simulation_tutorial_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_source_simulation_tutorial.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Source Data Simulation with SourceSimulator</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This tutorial demonstrates how to use the visualization features of CaliBrain.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_visualization_tutorial_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_visualization_tutorial.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Visualization Tutorial</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_tutorials/sensor_simulation_tutorial
   /auto_tutorials/source_simulation_tutorial
   /auto_tutorials/visualization_tutorial


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: auto_tutorials_python.zip </auto_tutorials/auto_tutorials_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: auto_tutorials_jupyter.zip </auto_tutorials/auto_tutorials_jupyter.zip>`
