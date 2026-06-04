Quick Start
===========


This example runs a small fixed-orientation simulation without requiring a
precomputed MNE forward solution. It demonstrates the core pipeline: simulate
sources, project them to sensors, solve an inverse problem, compute a
calibration curve, visualize it, and save a compact result file.

.. code-block:: python

   from pathlib import Path
   
   import matplotlib.pyplot as plt
   import numpy as np
   
   from calibrain import SensorSimulator, SourceSimulator, UncertaintyEstimator
   from calibrain import gamma_map
   
   rng = np.random.default_rng(42)
   output_dir = Path("results/quickstart")
   output_dir.mkdir(parents=True, exist_ok=True)
   
   # 1. Simulate sparse source activity.
   source_simulator = SourceSimulator(
       ERP_config={
           "tmin": -0.2,
           "tmax": 0.4,
           "sfreq": 200,
           "fmin": 1,
           "fmax": 8,
           "random_erp_timing": True,
       }
   )
   x_true, active_sources = source_simulator.simulate(
       n_sources=40,
       nnz=3,
       orientation_type="fixed",
       seed=42,
   )
   
   # 2. Use a small synthetic leadfield for a runnable demonstration.
   L = rng.normal(size=(20, x_true.shape[0])) / np.sqrt(x_true.shape[0])
   
   # 3. Project sources to sensors and add controlled white noise.
   sensor_simulator = SensorSimulator()
   y_clean, y_noisy, noise, noise_eta = sensor_simulator.simulate(
       x=x_true,
       L=L,
       alpha_SNR=0.7,
       sensor_white_noise_std=1.0,
       seed=43,
   )
   
   # 4. Estimate sources and posterior covariance.
   noise_var = max(float(np.var(noise)), 1e-12)
   result = gamma_map(
       L=L,
       y=y_noisy,
       noise_var=noise_var,
       n_orient=1,
       max_iter=100,
   )
   x_hat = result["posterior_mean"]
   posterior_var = np.maximum(np.diag(result["posterior_cov"]), 0.0)
   
   # 5. Compute a time-aggregated calibration curve.
   ue = UncertaintyEstimator(nominal_coverages=np.linspace(0.0, 1.0, 11))
   curve = ue.calibration_curve_intervals_aggregated(
       x_true=x_true,
       x_hat=x_hat,
       posterior_var=posterior_var,
   )
   
   # 6. Plot and save the calibration curve.
   fig, ax = plt.subplots(figsize=(5, 5))
   ax.plot([0, 1], [0, 1], "--", color="0.5", label="perfect calibration")
   ax.plot(
       curve["nominal_coverages"],
       curve["empirical_coverages"],
       "o-",
       label="Gamma-MAP",
   )
   ax.set(xlabel="Nominal coverage", ylabel="Empirical coverage", xlim=(0, 1), ylim=(0, 1))
   ax.set_aspect("equal", adjustable="box")
   ax.legend()
   fig.tight_layout()
   fig.savefig(output_dir / "calibration_curve.png", dpi=150)
   
   np.savez_compressed(
       output_dir / "quickstart_result.npz",
       x_true=x_true,
       x_hat=x_hat,
       posterior_var=posterior_var,
       active_sources=active_sources,
       nominal_coverages=curve["nominal_coverages"],
       empirical_coverages=curve["empirical_coverages"],
   )


Expected outputs:

- ``results/quickstart/calibration_curve.png``
- ``results/quickstart/quickstart_result.npz``

The synthetic leadfield in this example is useful for checking the API and
documentation examples. For scientific experiments, use the workflow scripts
and precomputed leadfields described in the user guide.
