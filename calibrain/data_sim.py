import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.cm as cm # Import colormap functionality
from matplotlib.lines import Line2D # Import for custom legend
import os
import mne

# Generate a smoothed ERP-like signal using bandpass-filtered noise and a Hanning window
def generate_erp_signal(n_times, sfreq, freq_band, amplitude, seed):
    rng = np.random.RandomState(seed)
    white_noise = rng.randn(n_times)

    # Bandpass filter between given frequency range
    b, a = butter(4, [f / (sfreq / 2) for f in freq_band], btype='band')
    erp = filtfilt(b, a, white_noise)

    # Apply Hanning window to smooth the signal
    erp *= np.hanning(n_times)

    # Normalize and scale the ERP to the desired amplitude
    erp /= np.std(erp)
    erp *= amplitude
    return erp

# Simulate source-space signal and propagate to sensor space with controlled SNR
def simulate_timeseries_with_snr_source_space(
    leadfield,
    snr_db=10,
    sfreq=100,
    tmin=-0.5,
    tmax=0.5,
    stim_onset=0.0,
    nnz=5,
    orientation_type="fixed",
    seed=None,
    amplitude=5.0,
    freq_band=(1, 30)
):
    rng = np.random.RandomState(seed)
    n_sensors = leadfield.shape[0]

    # Define the time axis and number of time points
    times = np.arange(tmin, tmax, 1.0 / sfreq)
    n_times = len(times)

    # Simulate source activity
    if orientation_type == "fixed":
        n_sources = leadfield.shape[1]
        active_indices = rng.choice(n_sources, size=nnz, replace=False)

        x = np.zeros((n_sources, n_times))
        for i, src_idx in enumerate(active_indices):
            # Generate ERP for each active dipole, passing freq_band
            erp_signal = generate_erp_signal(
                n_times, sfreq, freq_band, amplitude=amplitude, seed=seed + i if seed else None
            )
            stim_idx = np.where(times >= stim_onset)[0][0]
            x[src_idx, stim_idx:] = erp_signal[stim_idx:]

        # Project to sensor space
        y_clean = leadfield @ x

    elif orientation_type == "free":
        n_sources, n_orient = leadfield.shape[1:3]
        assert n_orient == 3, "Expected 3 orientations for free orientation"

        active_indices = rng.choice(n_sources, size=nnz, replace=False)
        x = np.zeros((n_sources, n_orient, n_times))

        for i, src_idx in enumerate(active_indices):
            # Generate ERP for each active dipole, passing freq_band
            erp_signal = generate_erp_signal(
                n_times, sfreq, freq_band, amplitude=amplitude, seed=seed + i if seed else None
            )
            stim_idx = np.where(times >= stim_onset)[0][0]

            # Random 3D orientation vector
            orient = rng.randn(3)
            orient /= np.linalg.norm(orient)

            for j in range(3):
                x[src_idx, j, stim_idx:] = orient[j] * erp_signal[stim_idx:]

        # Project to sensor space using tensor contraction
        y_clean = np.einsum("nmr,mrt->nt", leadfield, x)

    else:
        raise ValueError("Invalid orientation_type. Choose 'fixed' or 'free'.")

    # Compute noise level from signal power and desired SNR
    signal_power = np.mean(y_clean ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power)

    # Add Gaussian noise
    noise = rng.normal(0, noise_std, size=y_clean.shape)
    y_noisy = y_clean + noise

    # Return all key components
    return {
        "sensor_data": y_noisy,              # Noisy sensor data
        "source_data": x,                    # Source signal in the brain
        "clean_sensor_data": y_clean,        # Clean sensor signal before noise
        "noise": noise,                      # Noise matrix
        "noise_variance": noise_power,       # True noise variance used
        "times": times,                      # Time vector
        "leadfield": leadfield,              # Leadfield used
        "active_indices": active_indices,    # Indices of active sources
    }


def plot_active_sources(x, times, active_indices, stim_onset, nnz, save_dir=None, figure_name=None):
    """
    Plot all active sources in subplots.

    Parameters:
    - x (np.ndarray): Source activity matrix (n_sources, n_times).
    - times (np.ndarray): Time vector corresponding to the signals.
    - active_indices (list or np.ndarray): Indices of active sources.
    - stim_onset (float): Time of stimulus onset.
    - nnz (int): Number of active sources.
    - save_dir (str or Path, optional): Directory to save the figure. If None, the figure is not saved.
    - figure_name (str, optional): Name of the figure file (without extension). Required if save_dir is provided.

    Returns:
    - None
    """
    n_cols = 3
    n_rows = int(np.ceil(nnz / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), constrained_layout=True, sharex=True, sharey=True)
    fig.suptitle("Active Source Signals (Subplots)", fontsize=16)
    axes = axes.flatten()

    for i, src_idx in enumerate(active_indices):
        axes[i].plot(times, x[src_idx], label=f"Source {src_idx}", linewidth=2)
        axes[i].axvline(x=stim_onset, linestyle="--", color="gray", label="Stimulus Onset")
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("Amplitude (nAm)")
        axes[i].set_title(f"Active Source {src_idx}")
        axes[i].legend()
        axes[i].grid(True)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    if save_dir and figure_name:
        output_dir = Path(save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"{figure_name}.png"
        plt.savefig(save_path, dpi=300)
        print(f"Subplots figure saved to {save_path}")
    # plt.show()
    plt.close()

def plot_sensor_signals(y_clean, y_noisy, sensor_indices=None, times=None, save_dir=None, figure_name=None):
    """
    Plot clean and noisy sensor signals for specific sensors in subplots.

    Parameters:
    - y_clean (np.ndarray): Clean sensor measurements (n_sensors, n_times).
    - y_noisy (np.ndarray): Noisy sensor measurements (n_sensors, n_times).
    - sensor_indices (list or np.ndarray, optional): Indices of sensors to plot. If None, plot the first sensor.
    - times (np.ndarray, optional): Time vector corresponding to the signals. If None, indices are used as the x-axis.
    - save_dir (str or Path, optional): Directory to save the figure. If None, the figure is not saved.
    - figure_name (str, optional): Name of the figure file (without extension). Required if save_dir is provided.

    Returns:
    - None
    """
    if sensor_indices is None:
        sensor_indices = [0]
    if times is None:
        times = np.arange(y_clean.shape[1])

    n_sensors_to_plot = len(sensor_indices)
    fig, axes = plt.subplots(n_sensors_to_plot, 1, figsize=(10, n_sensors_to_plot * 3), sharex=True, sharey=True)
    fig.suptitle("Specific Sensor Signals (Subplots)", fontsize=16)

    if n_sensors_to_plot == 1:
        axes = [axes]

    for i, sensor_idx in enumerate(sensor_indices):
        axes[i].plot(times, y_clean[sensor_idx], label="y_clean", linewidth=2)
        axes[i].plot(times, y_noisy[sensor_idx], label="y_noise")
        axes[i].set_title(f"Sensor {sensor_idx}")
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("Amplitude  (µV)")
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_dir and figure_name:
        output_dir = Path(save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"{figure_name}.png"
        plt.savefig(save_path, dpi=300)
        print(f"Sensor subplots figure saved to {save_path}")
    # plt.show()
    plt.close()

def plot_all_active_sources_single_figure(x, times, active_indices, stim_onset, save_dir=None, figure_name=None):
    """
    Plot all specified active source signals on a single figure.

    Parameters:
    - x (np.ndarray): Source activity matrix (n_sources, n_times).
    - times (np.ndarray): Time vector corresponding to the signals.
    - active_indices (list or np.ndarray): Indices of active sources to plot.
    - stim_onset (float): Time of stimulus onset.
    - save_dir (str or Path, optional): Directory to save the figure. If None, the figure is not saved.
    - figure_name (str, optional): Name of the figure file (without extension). Required if save_dir is provided.

    Returns:
    - None
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    fig.suptitle("All Active Source Signals (Single Figure)", fontsize=16)
    colors = cm.viridis(np.linspace(0, 1, len(active_indices)))

    for i, src_idx in enumerate(active_indices):
        ax.plot(times, x[src_idx], label=f"Source {src_idx}", linewidth=1.5, color=colors[i])

    ax.axvline(x=stim_onset, linestyle="--", color="gray", label="Stimulus Onset")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (nAm)")
    ax.legend(loc='best', fontsize='small')
    ax.grid(True, alpha=0.6)
    ax.set_title("Overlay of Active Sources")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_dir and figure_name:
        output_dir = Path(save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"{figure_name}.png"
        plt.savefig(save_path, dpi=300)
        print(f"Single figure source plot saved to {save_path}")
    # plt.show()
    plt.close()


def plot_all_sensor_signals_single_figure(y_noisy, times, sensor_indices=None, save_dir=None, figure_name=None):
    """
    Plot all specified sensor signals (noisy) and their average on a single figure.

    Parameters:
    - y_noisy (np.ndarray): Noisy sensor measurements (n_sensors, n_times).
    - times (np.ndarray): Time vector corresponding to the signals.
    - sensor_indices (list or np.ndarray, optional): Indices of sensors to plot. If None, plots all sensors.
    - save_dir (str or Path, optional): Directory to save the figure. If None, the figure is not saved.
    - figure_name (str, optional): Name of the figure file (without extension). Required if save_dir is provided.

    Returns:
    - None
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    if sensor_indices is None:
        sensor_indices = np.arange(y_noisy.shape[0])
        y_plot = y_noisy
    else:
        y_plot = y_noisy[sensor_indices, :]

    average_signal = np.mean(y_plot, axis=0)
    colors = cm.plasma(np.linspace(0, 1, len(sensor_indices)))

    for i, sensor_idx in enumerate(sensor_indices):
        ax.plot(times, y_plot[i, :], linewidth=0.8, color=colors[i], alpha=0.5)

    ax.plot(times, average_signal, label="Average Signal", linewidth=2.0, color='black')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.grid(True, alpha=0.6)
    ax.set_title("Simulated EEG Sensor Signals and Average")

    legend_elements = [
        Line2D([0], [0], color=colors[0], lw=0.8, alpha=0.5, label=f'Individual Signals ({len(sensor_indices)} channels)'),
        Line2D([0], [0], color='black', lw=2.0, label='Average Signal across channels')
    ]
    ax.legend(handles=legend_elements, loc='best')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_dir and figure_name:
        output_dir = Path(save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"{figure_name}.png"
        plt.savefig(save_path, dpi=300)
        print(f"Single figure sensor plot saved to {save_path}")
    # plt.show()
    plt.close()

# ... (previous code, including imports and other functions) ...

def plot_stacked_sensor_signals(y_noisy, times, sensor_indices=None, offset_scale=1.5, save_dir=None, figure_name=None):
    """
    Plots specified sensor signals overlaid on a single figure, without vertical offset.
    The individual signal values are plotted as provided (no scaling or smoothing applied here).

    Parameters:
    - y_noisy (np.ndarray): Noisy sensor measurements (n_sensors, n_times). Values are plotted directly.
    - times (np.ndarray): Time vector corresponding to the signals.
    - sensor_indices (list or np.ndarray, optional): Indices of sensors to plot. If None, plots all sensors.
    - offset_scale (float): This parameter is ignored in this version as signals are overlaid.
    - save_dir (str or Path, optional): Directory to save the figure. If None, the figure is not saved.
    - figure_name (str, optional): Name of the figure file (without extension). Required if save_dir is provided.

    Returns:
    - None
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6)) # Adjusted figsize for overlay
    fig.suptitle("Overlay of Sensor Signals", fontsize=16) # Updated title

    if sensor_indices is None:
        sensor_indices = np.arange(y_noisy.shape[0]) # Plot all sensors if None
        y_plot = y_noisy # Use all sensors
    else:
        y_plot = y_noisy[sensor_indices, :] # Use selected sensors

    n_plot_sensors = y_plot.shape[0]

    # Generate colors for the plots
    colors = cm.plasma(np.linspace(0, 1, n_plot_sensors))

    # Plot each sensor signal directly without vertical offset
    # The y_plot[i, :] values are plotted directly without scaling or smoothing
    for i in range(n_plot_sensors):
        sensor_idx = sensor_indices[i]
        # Plotting the original signal values without any offset
        ax.plot(times, y_plot[i, :], linewidth=1.0, color=colors[i], alpha=0.7, label=f"Sensor {sensor_idx}" if n_plot_sensors <= 15 else None) # Add label only if few sensors

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)") # Updated y-axis label
    ax.grid(True, alpha=0.6) # Grid on both axes is fine for overlay

    # Optional: Add legend if number of sensors is small
    if n_plot_sensors <= 15:
         ax.legend(loc='best', fontsize='x-small')

    ax.set_title(f"Overlay of {n_plot_sensors} Sensor Signals") # Updated title

    # Remove frame spines for cleaner look (optional)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

    # Save the figure if a save path is provided
    if save_dir and figure_name:
        output_dir = Path(save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Adjust figure name if desired, e.g., append "_overlay"
        save_path = output_dir / f"{figure_name}_overlay.png"
        plt.savefig(save_path, dpi=300)
        print(f"Overlay sensor plot saved to {save_path}")
    # plt.show()
    plt.close(fig) # Close the specific figure

# ... (rest of the script, including simulation parameters, loading, simulation) ...




# --------------------------------------------------------------
# --- Simulation Parameters ---
leadfield_file_path = "results/forward/fsaverage-leadfield-fixed.npz"
orientation_type = "fixed"
tmin = -0.5
tmax = 0.5
stim_onset = 0.0
sfreq = 250
freq_band = (1, 30)
snr_db = 6
amplitude = 5.0
nnz = 5
seed = 42

# --- Plotting Parameters ---
figure_save_dir = Path("results/figures/data_sim") # Define base directory for figures
sensor_subplots_indices = [0, 10, 20] # Indices for the subplot sensor plot

# --- Load Leadfield ---
# ... (Keep the leadfield loading and inspection code as before) ...
print("="*50)
print(f"Loading leadfield from: {leadfield_file_path}")
try:
    with np.load(leadfield_file_path) as data:
        leadfield_matrix = data["leadfield"]
    print(f"Leadfield loaded successfully. Shape: {leadfield_matrix.shape}", "dtype:", leadfield_matrix.dtype)

    # --- Inspect Leadfield Matrix Values ---
    min_lf = np.min(leadfield_matrix)
    max_lf = np.max(leadfield_matrix)
    mean_abs_lf = np.mean(np.abs(leadfield_matrix))
    std_lf = np.std(leadfield_matrix)

    # -- Check for extreme values
    print(f"Leadfield matrix mean: {np.mean(leadfield_matrix):.1e}, std: {std_lf:.2e}")
    print(f"Leadfield matrix min: {min_lf:.1e}, max: {max_lf:.1e}")
    print(f"Leadfield matrix std: {std_lf:.1e}")
    print(f"Leadfield matrix mean abs: {mean_abs_lf:.1e}")

    # Check for NaNs or Infs
    if np.isnan(leadfield_matrix).any() or np.isinf(leadfield_matrix).any():
        print("WARNING: Leadfield matrix contains NaN or Inf values!")
    # --- End Inspection ---

except FileNotFoundError:
    print(f"Error: Leadfield file not found at {leadfield_file_path}")
    exit()
except Exception as e:
    print(f"Error loading or inspecting leadfield file: {e}")
    exit()


# --- Simulate Data ---
print("Simulating time series data...")
simulation_results = simulate_timeseries_with_snr_source_space(
    leadfield=leadfield_matrix,
    snr_db=snr_db,
    sfreq=sfreq,
    tmin=tmin,
    tmax=tmax,
    stim_onset=stim_onset,
    nnz=nnz,
    orientation_type=orientation_type,
    seed=seed,
    amplitude=amplitude,
    freq_band=freq_band
)
print("Data simulation complete.")

# --- Extract Simulation Results (Data is already in µV) ---
source_data_x = simulation_results["source_data"]
active_source_indices = simulation_results["active_indices"]
clean_sensor_data_y = simulation_results["clean_sensor_data"]
noisy_sensor_data_y = simulation_results["sensor_data"]
time_vector = simulation_results["times"]

# --- Check Sensor Data Range (in µV) ---
min_signal_uV = np.min(noisy_sensor_data_y)
max_signal_uV = np.max(noisy_sensor_data_y)
mean_abs_signal_uV = np.mean(np.abs(noisy_sensor_data_y))
print(f"Simulated Sensor Data Range (µV): Min={min_signal_uV:.2f}, Max={max_signal_uV:.2f}, Mean Abs={mean_abs_signal_uV:.2f}")
# ... (Keep checks for plausible range) ...


# # --- Plot Active Sources (Single Figure) ---
# print("Plotting active source signals (single figure)...")
# plot_all_active_sources_single_figure(
#     x=source_data_x,
#     times=time_vector,
#     active_indices=active_source_indices,
#     stim_onset=stim_onset,
#     save_dir=figure_save_dir,
#     figure_name="active_sources_single_figure"
# )

# --- Plot Specific Sensor Signals (Subplots) ---
print("Plotting specific sensor signals (subplots)...")
plot_sensor_signals(
    y_clean=clean_sensor_data_y,
    y_noisy=noisy_sensor_data_y,
    sensor_indices=sensor_subplots_indices,
    times=time_vector,
    save_dir=figure_save_dir,
    figure_name="specific_sensor_signals_subplots"
)

# --- Plot All Sensor Signals (Single Figure - Overlay) ---
print("Plotting all sensor signals and average (single figure)...")
plot_all_sensor_signals_single_figure(
    y_noisy=noisy_sensor_data_y,
    times=time_vector,
    # sensor_indices=None, # Optional: Plot all sensors by default
    save_dir=figure_save_dir,
    figure_name="all_sensor_signals_single_figure"
)

# --- Plot All Sensor Signals (Single Figure - Stacked is now Overlay) ---
print("Plotting all sensor signals (overlay figure)...") # Updated print message
plot_stacked_sensor_signals(
    y_noisy=noisy_sensor_data_y, # Pass the raw noisy data
    times=time_vector,
    # sensor_indices=None, # Optional: Plot all sensors by default
    save_dir=figure_save_dir,
    figure_name="all_sensor_signals" # Adjusted figure name base
)

n_channels = leadfield_matrix.shape[0]
ch_names = [f"EEG{n:02}" for n in range(1, n_channels + 1)]
info = mne.create_info(sfreq=sfreq, ch_types="eeg", ch_names=ch_names)

raw = mne.io.RawArray(noisy_sensor_data_y, info)
fig = raw.plot()
plt.show()

print("Script finished.")
