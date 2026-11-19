.. meta::
   :description: Definitions of the terms used in the CaliBrain documentation and code: data and head models, source models, inverse solvers, uncertainty representations, and calibration modes.

Glossary
========

The terms of CaliBrain and of EEG/MEG source imaging as the documentation and the code
use them. Names in ``code font`` are the names used in function arguments and
configuration files.

.. glossary::
   :sorted:

   adaptive_joint_learning
      A noise-variance strategy (``noise_type="adaptive_joint_learning"``). No noise
      variance is given to the solver; a joint-learning solver (``BMN_joint`` or
      ``gamma_lambda_map_sflex``) estimates it during the fit. See :term:`noise-variance
      strategy`.

   aggregated
      An uncertainty mode (``calibration_curve_intervals_aggregated``) that works on the
      time average :math:`\bar{x}` of the source activity instead of on every time sample.
      The posterior covariance of the average is the covariance divided by the number of
      time samples :math:`T`. The calibration workflow uses this mode. See
      :term:`pointwise`.

   alpha_SNR
      The signal-to-noise mixing parameter of ``SensorSimulator.simulate``, between 0 and
      1. For 1 the sensor data have no noise. Otherwise the noise is scaled by
      :math:`(1 - \alpha) / \alpha` times the ratio of the Frobenius norms of the clean
      data and the noise.

   baseline noise variance
      The noise variance estimated from the segment before the stimulus
      (``noise_type="baseline"``). See :term:`noise-variance strategy`.

   BMN
      Bayesian minimum norm: an inverse solver (``calibrain.BMN``) that returns the
      posterior mean and covariance of the source activity under one common source
      variance. The noise variance is an input.

   BMN_joint
      The variant of :term:`BMN` (``calibrain.BMN_joint``) that learns a common sensor-noise
      variance together with the source variance.

   CamCAN
      Cambridge Centre for Ageing and Neuroscience, a data repository. CaliBrain offers
      the forward solutions of four of its participants (``CC120166``, ``CC120264``,
      ``CC120309``, ``CC120313``) as a separate example dataset, which remains subject to the
      CamCAN terms. See :doc:`../datasets/data_notice`.

   calibration
      Comparing the nominal coverage of credible sets with their empirical coverage and,
      where they differ, learning a mapping that corrects them. See :term:`recalibration`.

   calibration curve
      The empirical coverage plotted against the nominal coverage. The diagonal is perfect
      calibration.

   coil type
      The sensor type of the leadfield. ``LeadfieldData.coil_type`` holds the MNE/FIFF code
      of the sensor hardware; ``extract_leadfields`` takes the argument
      ``coil_name``: ``"eeg"`` (electrodes), ``"mag"`` (magnetometers), or ``"grad"``
      (gradiometers).

   cortical normal
      The unit vector perpendicular to the cortical surface at a source. A free-orientation
      leadfield file stores the normals (``cortical_normals``, shape ``(n_sources, 3)``), and
      the free-orientation simulation places the activity of an active source along its normal.

   credible interval
      The interval around the posterior mean that contains the true value of a scalar source
      with a given probability, for a Gaussian posterior. Fixed orientation uses credible
      intervals.

   credible set
      The region of the posterior that contains the true source value with a given
      probability: an interval (:math:`d = 1`), an ellipse (:math:`d = 2`), or an ellipsoid
      (:math:`d = 3`). :term:`nominal coverage` is that probability.

   data directory
      The folder with the example data and the leadfield files. It has the subfolders
      ``fwd/``, ``1284src_fwd/``, and ``1284src_leadfield/``. ``get_data_path()`` returns it:
      ``<repository>/data`` in a source checkout and a per-user folder for an installed
      package. Functions take a ``data_dir`` argument to use another folder.

   empirical coverage
      The fraction of cases in which the true source value lies inside the credible set of a
      given nominal coverage.

   EEG
      Electroencephalography. EEG sources have fixed orientation or three free components.
      EEG leadfields are in µV/nAm.

   EMD
      Earth mover's distance (``emd`` in ``MetricEvaluator``). It compares the distribution of
      the estimated source activity over the source space with that of the true activity.

   ERP
      Event-related potential. ``SourceSimulator`` draws the waveform of every active
      source as an ERP-like signal, configured with ``ERP_config``.

   example data
      The forward solutions of the ``fsaverage`` template and of four CamCAN participants, which
      are downloaded with ``fetch_example_data`` from the CaliBrain-data repository and are
      not part of the package. See :doc:`../datasets/index`.

   fixed orientation
      A source model with one scalar amplitude per source, along the cortical normal. The
      local source dimension is :math:`d = 1`, and the leadfield has the shape
      ``(n_sensors, n_sources)``.

   forward model
      The mapping from source activity to sensor measurements. It is represented by the
      :term:`leadfield`, which is computed from a :term:`forward solution`.

   forward solution
      The result of an MNE forward computation, stored as a ``<subject>-fwd.fif`` file: the
      source space, the gain matrix, and the sensor information. The :term:`example data`
      are forward solutions. ``reduce_forward_solutions`` reduces them to 1284 sources, and
      ``extract_leadfields`` creates the :term:`leadfield` files from them.

   free orientation
      A source model with several components per source instead of one scalar amplitude: three
      for EEG (:math:`d = 3`) and two tangential ones for MEG (:math:`d = 2`, see
      :term:`reduced free-orientation MEG`). The leadfield has the shape ``(n_sensors,
      n_sources, n_components)``.

   free-orientation EEG
      A free-orientation EEG model with three components per source (:math:`d = 3`).

   fsaverage
      The template head model distributed with FreeSurfer and MNE-Python. The ``fsaverage``
      example dataset has 64 EEG channels and contains no participant data.

   full source space
      The source space of a downloaded :term:`forward solution`, with about 5100 sources (2562
      per hemisphere). CaliBrain works with the :term:`reduced source space`.

   full_cov
      The free-orientation uncertainty representation that uses the local posterior
      covariance block of every source, which gives ellipsoidal credible sets. See
      :term:`marginal`.

   gamma_lambda_map_sflex
      The variant of :term:`gamma_map_sflex` that also learns a noise-related regularization
      parameter (``calibrain.gamma_lambda_map_sflex``).

   gamma_map_sflex
      A sparse Bayesian inverse solver (``calibrain.gamma_map_sflex``): Gamma-MAP with an
      :term:`sFLEX` spatial basis. The noise variance is an input.

   head
      The word used in the workflow configurations (``HEADS``) for one subject's forward model,
      for example ``CC120166``. Calibration data are pooled or split by head.

   inverse problem
      Estimating the source activity from the sensor measurements. The leadfield has fewer
      rows (sensors) than columns (sources), so the problem is underdetermined.

   isotonic regression
      A monotonic regression. ``UncertaintyCalibrator`` uses it to map nominal to empirical
      coverage, which keeps the order of the coverage levels.

   leadfield
      The matrix that maps source amplitudes (in nAm) to sensor measurements: fT/nAm for MEG
      and µV/nAm for EEG. CaliBrain stores it in ``<subject>_fixed_leadfield.npz`` and
      ``<subject>_free_leadfield.npz`` together with the source coordinates, the sensor units,
      and for free orientation the :term:`cortical normal`\ s. ``extract_leadfields`` creates
      these files; ``LeadfieldBuilder`` reads them.

   local source basis
      A :math:`3 \times d` matrix per source (``Q_basis``, shape ``(n_sources, 3, d)``) that
      relates the :math:`d` local components of a source to a common 3D frame: :math:`d = 3`
      for EEG and :math:`d = 2` (the tangential plane) for MEG. It is stored in the free
      leadfield file and used by the free-orientation simulation.

   marginal
      The free-orientation uncertainty representation that calibrates every component with its
      own marginal variance and ignores the covariance between components. See
      :term:`full_cov`.

   MEG
      Magnetoencephalography. MEG sources have fixed orientation or two free tangential
      components (:term:`reduced free-orientation MEG`). Magnetometer leadfields are in
      fT/nAm.

   nnz
      The number of active sources (non-zero rows) in a simulated source activity, the argument
      of ``SourceSimulator.simulate``.

   nominal coverage
      The probability a credible set is built to contain the true value with, for example 0.9
      for a 90% credible set.

   noise-variance strategy
      How the sensor-noise variance reaches the solver (``noise_type``): ``oracle`` (the true
      value), ``baseline`` (estimated before the stimulus), or ``adaptive_joint_learning``
      (learned by the solver).

   oracle noise variance
      The true noise variance, computed from the noise that was added to the simulated data
      (``noise_type="oracle"``).

   pointwise
      An uncertainty mode (``calibration_curve_intervals_pointwise``) that checks the credible
      set of every source at every time sample. See :term:`aggregated`.

   post_fixed
      A calibration mode: one recalibration map is fitted at a reference condition and applied
      to a series of evaluation conditions.

   post_oracle
      A calibration mode: the map is fitted and evaluated on matching training and evaluation
      conditions.

   post_pooled
      A calibration mode: the map is fitted on pooled training data and evaluated on a target
      condition.

   post_pooled_mismatch
      A calibration mode: the map is fitted on pooled training data whose conditions
      deliberately differ from the target condition, which is where it is evaluated.

   posterior covariance
      The covariance matrix of the posterior distribution of the source activity, returned by
      the solver.

   posterior mean
      The mean of the posterior distribution of the source activity: the estimate.

   posterior summary
      The file (HDF5) that ``DataGenerator`` saves for every run: posterior mean and covariance
      with the metadata of the run (orientation, solver, noise type, seed, active sources, and
      so on). The :term:`run manifest` lists these files.

   precal
      A calibration mode that measures the empirical coverage before any recalibration.

   recalibration
      Correcting the nominal coverage with a fitted mapping, in CaliBrain isotonic regression.

   reduced free-orientation MEG
      The free-orientation model for MEG with two tangential components per source
      (:math:`d = 2`). Its uncertainty regions are ellipses.

   reduced source space
      The source space of 1284 sources (642 per hemisphere) that CaliBrain works with,
      chosen at random from the :term:`full source space` with a fixed seed.
      ``reduce_forward_solutions`` creates it, and the folders ``1284src_fwd/`` and
      ``1284src_leadfield/`` are named after it.

   run manifest
      The CSV file (``manifest_path``) that lists the posterior summaries of the runs of the
      data-generation workflow. The aggregation step finds the summaries through it.

   sensor noise
      Gaussian noise that ``SensorSimulator`` adds to the sensor data before the sources are
      estimated.

   sFLEX
      Sparse basis field expansion, the spatial basis of the Gamma-MAP solvers. It is built
      from the source coordinates: two sources within ``threshold_factor`` times ``sigma`` of
      each other get the weight :math:`\exp(-d^2 / 2\sigma^2)`, where :math:`d` is their
      distance, and the matrix is normalized. The solver imposes sparsity on the coefficients
      of this basis and so keeps the source-space posterior covariance defined at inactive
      sources.

   source activity
      The time course of the current dipole moments (in nAm) at the sources. Fixed orientation:
      shape ``(n_sources, n_times)``. Free orientation: ``(n_sources, n_components, n_times)``.

   source space
      The set of locations where sources are modelled. See :term:`full source space` and
      :term:`reduced source space`.

   SourceEstimator
      The class (``calibrain.SourceEstimator``) that applies an inverse solver, for example
      ``gamma_map_sflex`` or ``BMN``, to a leadfield and sensor data and returns the posterior
      mean and covariance.

   subject
      The identifier of a forward solution: the file name without ``-fwd.fif``, for example
      ``fsaverage`` or ``CC120166``. Leadfield files and posterior summaries carry it.

   uncertainty representation
      The object that defines a credible set for a source: a scalar interval (fixed
      orientation), a family of component-wise intervals (:term:`marginal`), or an
      ellipse or ellipsoid (:term:`full_cov`).
