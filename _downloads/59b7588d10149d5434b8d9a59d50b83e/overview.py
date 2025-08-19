"""
.. _tut-overview:

=============================================
Overview of brain source localization with CaliBrain
=============================================

This tutorial covers the basic CaliBrain pipeline for brain source localization
and uncertainty quantification: building forward models, simulating brain activity,
estimating sources, and quantifying uncertainty. It introduces the core CaliBrain
data structures and components, covering the essential workflow at a high level.
Subsequent tutorials address each topic in greater detail.

CaliBrain is designed around a modular architecture where each component handles
a specific aspect of the source localization and uncertainty quantification pipeline.

We begin by importing the necessary Python modules:
"""

# Authors: Mohammad Orabe  <m.orabe@icloud.com>
# License: AGPL-3.0 license
# Copyright the CaliBrain contributors.

# %%

import numpy as np
import matplotlib.pyplot as plt

# %%
# The CaliBrain architecture
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# CaliBrain follows a modular design with 8 main components that work together
# to provide a complete pipeline for brain source localization with uncertainty
# quantification:
#
# 1. :class:`~calibrain.LeadfieldBuilder` - Creates the forward model
# 2. :class:`~calibrain.SourceSimulator` - Generates brain source activity  
# 3. :class:`~calibrain.SensorSimulator` - Simulates sensor measurements
# 4. :class:`~calibrain.SourceEstimator` - Solves the inverse problem
# 5. :class:`~calibrain.UncertaintyEstimator` - Quantifies estimate uncertainty
# 6. :class:`~calibrain.MetricEvaluator` - Evaluates performance and calibration
# 7. :class:`~calibrain.Visualizer` - Creates plots and visualizations
# 8. :class:`~calibrain.Benchmark` - Orchestrates complete experiments
#
# These components can be used individually for specific tasks or together 
# through the :class:`~calibrain.Benchmark` class for automated workflows.

from calibrain import (
    LeadfieldBuilder,
    SourceSimulator,
    SensorSimulator,
    SourceEstimator,
    UncertaintyEstimator,
    MetricEvaluator,
    Visualizer,
    Benchmark
)

# %%
# Building the forward model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The :class:`~calibrain.LeadfieldBuilder` creates the forward model that maps
# brain sources to sensor measurements. This is the foundation of all source
# localization analyses, as it defines the relationship between neural activity
# and what we observe at the sensors.
#
# The forward model is represented by the leadfield matrix **L**, where:
# **sensor_data = L × source_activity + noise**
#
# CaliBrain integrates with MNE-Python to provide realistic head models and
# supports both EEG and MEG modalities with various source orientations.

# Configure the forward model
leadfield_config = {
    "subject": "fsaverage",           # Use MNE's template brain
    "spacing": "ico4",                # Source space resolution  
    "modality": "eeg",                # EEG sensors
    "orientation": "fixed",           # Fixed source orientations
    "montage": "standard_1020",       # Standard EEG layout
}

# # Build the leadfield matrix
# leadfield_builder = LeadfieldBuilder(config=leadfield_config)
# leadfield = leadfield_builder.simulate()

# print(f"Leadfield matrix shape: {leadfield.shape}")
# print(f"  Sensors: {leadfield.shape[0]}")  
# print(f"  Sources: {leadfield.shape[1]}")

# # %%
# # The leadfield matrix dimensions tell us about our measurement setup:
# # the number of sensors (EEG electrodes or MEG sensors) and the number
# # of potential source locations in the brain. The condition number of
# # this matrix affects the difficulty of the inverse problem.

# condition_number = np.linalg.cond(leadfield)
# print(f"Condition number: {condition_number:.2e}")

# # %%
# # Simulating brain source activity
# # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# #
# # The :class:`~calibrain.SourceSimulator` generates realistic brain source
# # activity patterns with event-related potential (ERP) waveforms. This allows
# # us to test source localization algorithms with known ground truth.

# # Define the ERP characteristics
# ERP_config = {
#     "tmin": -0.2,          # Pre-stimulus period (seconds)
#     "tmax": 0.5,           # Post-stimulus period (seconds)
#     "sfreq": 250,          # Sampling frequency (Hz)
#     "amplitude": 50.0,     # Peak amplitude (nAm)
# }

# # Initialize the source simulator
# source_sim = SourceSimulator(ERP_config=ERP_config)

# # Simulation parameters
# n_sources = leadfield.shape[1]  # Use all available source locations
# n_active = 3                    # Number of simultaneously active sources
# n_trials = 20                   # Number of trials to simulate

# # Generate source activity
# source_data, active_indices = source_sim.simulate(
#     n_sources=n_sources,
#     n_active=n_active, 
#     n_trials=n_trials
# )

# print(f"Source simulation completed:")
# print(f"  Active sources: {len(active_indices)} out of {n_sources}")
# print(f"  Data shape: {source_data.shape} (sources × time × trials)")
# print(f"  Active indices: {active_indices}")

# # %%
# # Let's visualize the simulated ERP waveforms to understand what we've created.
# # We'll plot the time courses of the active sources and show the distribution
# # of source amplitudes across the brain.

# time = np.linspace(ERP_config['tmin'], ERP_config['tmax'], source_data.shape[1])

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# # Plot ERP time courses
# for i, idx in enumerate(active_indices):
#     ax1.plot(time, source_data[idx, :, 0], label=f'Source {idx}')
# ax1.axvline(0, color='r', linestyle='--', alpha=0.7, label='Stimulus onset')
# ax1.set_xlabel('Time (s)')
# ax1.set_ylabel('Amplitude (nAm)')
# ax1.set_title('Simulated ERP Waveforms')
# ax1.legend()
# ax1.grid(True, alpha=0.3)

# # Show source amplitude distribution
# all_amplitudes = np.max(np.abs(source_data), axis=(1, 2))
# ax2.stem(range(len(all_amplitudes)), all_amplitudes, basefmt=' ')
# ax2.set_xlabel('Source Index')
# ax2.set_ylabel('Peak Amplitude (nAm)')
# ax2.set_title('Source Amplitude Distribution')
# ax2.set_yscale('log')
# ax2.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()

# # %%
# # Forward modeling: simulating sensor measurements
# # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# #
# # The :class:`~calibrain.SensorSimulator` applies the forward model to convert
# # our simulated brain activity into realistic sensor measurements. It also adds
# # configurable levels of measurement noise to simulate realistic recording
# # conditions.

# # Initialize sensor simulator
# sensor_sim = SensorSimulator()

# # Apply forward model with noise
# alpha_SNR = 0.5  # Noise level (0 = no noise, 1 = pure noise)
# sensor_clean, sensor_noisy = sensor_sim.simulate(
#     source_data=source_data,
#     leadfield=leadfield,
#     alpha_SNR=alpha_SNR
# )

# # Calculate the actual signal-to-noise ratio
# signal_power = np.var(sensor_clean)
# noise_power = np.var(sensor_noisy - sensor_clean)
# snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf

# print(f"Sensor data generated:")
# print(f"  Shape: {sensor_clean.shape} (channels × time × trials)")
# print(f"  Signal-to-noise ratio: {snr_db:.1f} dB")

# # %%
# # Solving the inverse problem
# # ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# #
# # The :class:`~calibrain.SourceEstimator` solves the inverse problem to estimate
# # brain source activity from the sensor measurements. CaliBrain implements
# # several source estimation methods with different characteristics.

# # Initialize source estimator
# source_est = SourceEstimator()

# # Test different estimation methods
# methods_to_test = ['eloreta', 'gamma_map']
# source_estimates = {}

# for method in methods_to_test:
#     print(f"\nTesting {method.upper()} method...")
    
#     try:
#         # Estimate sources using the first trial
#         estimated = source_est.estimate(
#             sensor_data=sensor_noisy[:, :, 0],
#             leadfield=leadfield,
#             method=method,
#             alpha=0.01  # Regularization parameter
#         )
        
#         source_estimates[method] = estimated
        
#         # Calculate some basic metrics
#         peak_amplitude = np.max(np.abs(estimated))
#         active_threshold = peak_amplitude * 0.1
#         n_detected = np.sum(np.max(np.abs(estimated), axis=1) > active_threshold)
        
#         print(f"  Peak amplitude: {peak_amplitude:.2e}")
#         print(f"  Sources detected: {n_detected}")
        
#     except Exception as e:
#         print(f"  Error: {e}")
#         source_estimates[method] = None

# # %%
# # Let's visualize the source estimation results to compare the different methods
# # and see how well they recover the true source locations.

# if any(est is not None for est in source_estimates.values()):
#     fig, axes = plt.subplots(2, len(methods_to_test), figsize=(12, 8))
#     if len(methods_to_test) == 1:
#         axes = axes.reshape(-1, 1)
    
#     for i, method in enumerate(methods_to_test):
#         if source_estimates[method] is not None:
#             estimated = source_estimates[method]
            
#             # Time course comparison
#             ax = axes[0, i]
#             for idx in active_indices[:2]:  # Show first 2 active sources
#                 ax.plot(time, estimated[idx, :], '-', label=f'Est. {idx}')
#                 ax.plot(time, source_data[idx, :, 0], '--', alpha=0.7, label=f'True {idx}')
#             ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
#             ax.set_xlabel('Time (s)')
#             ax.set_ylabel('Amplitude')
#             ax.set_title(f'{method.upper()} - Time Courses')
#             ax.legend()
#             ax.grid(True, alpha=0.3)
            
#             # Source detection visualization
#             ax = axes[1, i]
#             est_amplitudes = np.max(np.abs(estimated), axis=1)
#             colors = ['red' if idx in active_indices else 'blue' 
#                      for idx in range(len(est_amplitudes))]
            
#             ax.scatter(range(len(est_amplitudes)), est_amplitudes, 
#                       c=colors, alpha=0.6, s=20)
#             ax.set_xlabel('Source Index')
#             ax.set_ylabel('Peak Amplitude')
#             ax.set_title(f'{method.upper()} - Detection (Red=True Active)')
#             ax.set_yscale('log')
#             ax.grid(True, alpha=0.3)
#         else:
#             for j in range(2):
#                 axes[j, i].text(0.5, 0.5, f'{method.upper()}\nNot Available', 
#                                ha='center', va='center', transform=axes[j, i].transAxes)
    
#     plt.tight_layout()
#     plt.show()

# # %%
# # Quantifying uncertainty
# # ^^^^^^^^^^^^^^^^^^^^^^^
# #
# # The :class:`~calibrain.UncertaintyEstimator` computes confidence intervals
# # for the source estimates, allowing us to assess the reliability of our
# # localization results. This is crucial for understanding which parts of
# # our source estimates we can trust.

# # Select the best available method for uncertainty quantification
# best_method = 'eloreta' if source_estimates.get('eloreta') is not None else 'gamma_map'

# if source_estimates[best_method] is not None:
#     print(f"Computing uncertainty for {best_method.upper()} estimates...")
    
#     # Initialize uncertainty estimator
#     uncertainty_est = UncertaintyEstimator()
    
#     # Compute 95% confidence intervals
#     confidence_level = 0.95
    
#     try:
#         lower_bounds, upper_bounds, point_estimates = uncertainty_est.estimate(
#             sensor_data=sensor_noisy[:, :, :5],  # Use first 5 trials
#             leadfield=leadfield,
#             method=best_method,
#             confidence_level=confidence_level,
#             n_bootstrap=50  # Number of bootstrap samples
#         )
        
#         print(f"✓ Computed {confidence_level:.0%} confidence intervals")
        
#         # Calculate coverage for the active sources
#         true_signal = source_data[:, :, :5]  # First 5 trials to match
#         within_bounds = ((true_signal >= lower_bounds[:, :, np.newaxis]) & 
#                         (true_signal <= upper_bounds[:, :, np.newaxis]))
#         empirical_coverage = np.mean(within_bounds)
        
#         print(f"Empirical coverage: {empirical_coverage:.1%} (target: {confidence_level:.0%})")
        
#     except Exception as e:
#         print(f"Error in uncertainty estimation: {e}")
#         lower_bounds = upper_bounds = point_estimates = None

# # %%
# # Let's visualize the uncertainty estimates for a few of the active sources
# # to see how well our confidence intervals capture the true variability.

# if 'lower_bounds' in locals() and lower_bounds is not None:
#     fig, axes = plt.subplots(1, min(2, len(active_indices)), figsize=(12, 4))
#     if len(active_indices) == 1:
#         axes = [axes]
    
#     for i, source_idx in enumerate(active_indices[:2]):
#         ax = axes[i]
        
#         # Plot confidence intervals
#         ax.fill_between(time, 
#                        lower_bounds[source_idx, :], 
#                        upper_bounds[source_idx, :],
#                        alpha=0.3, color='blue', 
#                        label=f'{confidence_level:.0%} CI')
        
#         # Plot point estimate
#         ax.plot(time, point_estimates[source_idx, :], 
#                 'b-', linewidth=2, label='Point estimate')
        
#         # Plot true signal from first trial
#         ax.plot(time, source_data[source_idx, :, 0], 
#                 'r--', linewidth=2, label='True signal')
        
#         ax.axvline(0, color='gray', linestyle='--', alpha=0.7)
#         ax.set_xlabel('Time (s)')
#         ax.set_ylabel('Amplitude (nAm)')
#         ax.set_title(f'Source {source_idx} Uncertainty')
#         ax.legend()
#         ax.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.show()

# # %%
# # Performance evaluation
# # ^^^^^^^^^^^^^^^^^^^^^^
# #
# # The :class:`~calibrain.MetricEvaluator` assesses the quality of our source
# # estimates and uncertainty quantification. This is essential for validating
# # our methods and comparing different approaches.

# if source_estimates[best_method] is not None:
#     print(f"Evaluating {best_method.upper()} performance...")
    
#     # Initialize metric evaluator
#     metric_eval = MetricEvaluator()
    
#     try:
#         # Compute localization metrics
#         localization_metrics = metric_eval.compute_localization_metrics(
#             true_sources=source_data[:, :, 0],  # Use first trial as reference
#             estimated_sources=source_estimates[best_method],
#             true_active_indices=active_indices
#         )
        
#         print("\n📊 Localization Performance:")
#         for metric, value in localization_metrics.items():
#             print(f"  {metric}: {value:.4f}")
        
#         # Compute calibration metrics if uncertainty is available
#         if 'lower_bounds' in locals() and lower_bounds is not None:
#             calibration_metrics = metric_eval.compute_calibration_metrics(
#                 true_sources=source_data[:, :, :5],  # Match uncertainty data
#                 lower_bounds=lower_bounds,
#                 upper_bounds=upper_bounds,
#                 confidence_level=confidence_level
#             )
            
#             print(f"\n📈 Calibration Quality ({confidence_level:.0%} CI):")
#             for metric, value in calibration_metrics.items():
#                 print(f"  {metric}: {value:.4f}")
        
#     except Exception as e:
#         print(f"Error in performance evaluation: {e}")

# # %%
# # Automated benchmarking workflows
# # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# #
# # The :class:`~calibrain.Benchmark` class orchestrates complete experimental
# # workflows for systematic method comparison and parameter optimization. This
# # is particularly useful for large-scale studies or when you need to test
# # multiple conditions automatically.

# print("\n🔬 Benchmark Workflow Overview")
# print("=" * 40)

# # Define a parameter grid for systematic evaluation
# param_grid = {
#     "subject": ["fsaverage"],
#     "nnz": [1, 3, 5],                    # Number of active sources
#     "orientation_type": ["fixed"],        # Source orientations  
#     "alpha_SNR": [0.3, 0.5, 0.7],       # Noise levels
# }

# print("Example parameter combinations:")
# total_combinations = len(param_grid["nnz"]) * len(param_grid["alpha_SNR"])
# for nnz in param_grid["nnz"]:
#     for alpha in param_grid["alpha_SNR"]:
#         print(f"  • {nnz} active sources, α_SNR = {alpha}")

# print(f"\nTotal combinations: {total_combinations}")

# # Initialize benchmark (demonstration only)
# benchmark = Benchmark(
#     ERP_config=ERP_config,
#     data_param_grid=param_grid,
#     experiment_dir="./benchmark_results"
# )

# print("To run the complete benchmark:")
# print("results = benchmark.run(nruns=10)")
# print("This would generate a comprehensive performance database.")

# # %%
# # Creating comprehensive visualizations
# # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# #
# # The :class:`~calibrain.Visualizer` creates publication-ready figures for
# # all aspects of the analysis. It can automatically generate comprehensive
# # reports including time series, topographies, brain maps, and calibration plots.

# # Initialize visualizer
# viz = Visualizer(base_save_path="./tutorial_figures")

# print("📊 Visualization capabilities:")
# print("  • ERP time series and source activity plots")
# print("  • Sensor topographies and brain source maps") 
# print("  • Uncertainty quantification visualizations")
# print("  • Calibration analysis and performance curves")
# print("  • Automated report generation")

# # Example of creating a summary figure (conceptual)
# print("\nTo create comprehensive visualizations:")
# print("viz.create_analysis_summary(source_data, sensor_data, estimates, uncertainty)")

# # %%
# # .. _calibrain-workflow-summary:
# #
# # Summary and next steps
# # ^^^^^^^^^^^^^^^^^^^^^^
# #
# # This tutorial covered the essential CaliBrain workflow for brain source
# # localization with uncertainty quantification:
# #
# # ✅ **Forward modeling**: Built leadfield matrix with :class:`~calibrain.LeadfieldBuilder`
# #
# # ✅ **Source simulation**: Generated realistic ERPs with :class:`~calibrain.SourceSimulator`
# #
# # ✅ **Sensor simulation**: Applied forward model and noise with :class:`~calibrain.SensorSimulator`
# #
# # ✅ **Source estimation**: Solved inverse problem with :class:`~calibrain.SourceEstimator`
# #
# # ✅ **Uncertainty quantification**: Computed confidence intervals with :class:`~calibrain.UncertaintyEstimator`
# #
# # ✅ **Performance evaluation**: Assessed quality with :class:`~calibrain.MetricEvaluator`
# #
# # ✅ **Automated workflows**: Demonstrated systematic evaluation with :class:`~calibrain.Benchmark`
# #
# # The modular design allows you to use individual components for specific tasks
# # or combine them for complete analyses. The uncertainty quantification capabilities
# # make CaliBrain particularly suitable for rigorous assessment of source localization
# # reliability.

# print("\n🎯 Next steps:")
# print("  • Explore the detailed component tutorials")
# print("  • Try the examples with your own data")
# print("  • Read the API documentation for parameter details")
# print("  • Join the community discussions on GitHub")
# print("\n🧠 Happy source localizing!")

# ##############################################################################
# # The subsequent tutorials dive deeper into each component, covering advanced
# # features like custom source estimation methods, sophisticated uncertainty
# # analysis, large-scale benchmarking workflows, and integration with real
# # EEG/MEG data. CaliBrain's focus on uncertainty quantification makes it
# # particularly valuable for robust neuroscience research.