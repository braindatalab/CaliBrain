
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from pathlib import Path

# Generate a smoothed ERP-like signal using bandpass-filtered noise and a Hanning window
def generate_erp_signal(n_times, sfreq, freq_band=(2, 5), amplitude=5.0, seed=None):
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
    nnz=1,
    orientation_type="fixed",
    seed=None,
    amplitude=5.0,
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
            # Generate ERP for each active dipole
            erp_signal = generate_erp_signal(n_times, sfreq, amplitude=amplitude, seed=seed + i if seed else None)
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
            erp_signal = generate_erp_signal(n_times, sfreq, amplitude=amplitude, seed=seed + i if seed else None)
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


def plot_active_sources(x, times, active_indices, stim_onset, nnz, save_path="results/figures/active_sources_subplots.png"):
    """
    Plot all active sources in subplots.

    Parameters:
    - x (np.ndarray): Source activity matrix (n_sources, n_times).
    - times (np.ndarray): Time vector corresponding to the signals.
    - active_indices (list or np.ndarray): Indices of active sources.
    - stim_onset (float): Time of stimulus onset.
    - nnz (int): Number of active sources.
    - save_path (str): Path to save the figure.

    Returns:
    - None
    """
    n_cols = 3  # Number of columns in the subplot grid
    n_rows = int(np.ceil(nnz / n_cols))  # Calculate the number of rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), constrained_layout=True, sharex=True, sharey=True)
    fig.suptitle("Active Source Signals", fontsize=16)

    # Flatten axes for easy indexing
    axes = axes.flatten()

    for i, src_idx in enumerate(active_indices):
        axes[i].plot(times, x[src_idx], label=f"Source {src_idx}", linewidth=2)
        axes[i].axvline(x=stim_onset, linestyle="--", color="gray", label="Stimulus Onset")
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("Amplitude (nAm)")
        axes[i].set_title(f"Active Source {src_idx}")
        axes[i].legend()
        axes[i].grid(True)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Adjust layout and save the figure
    plt.tight_layout()
    output_dir = Path(save_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Figure saved to {save_path}")
    plt.show()
    plt.close()
    
    
def plot_sensor_signals(y_clean, y_noisy, sensor_indices=None, times=None, save_path=None):
    """
    Plot clean and noisy sensor signals for specific sensors.

    Parameters:
    - y_clean (np.ndarray): Clean sensor measurements (n_sensors, n_times).
    - y_noisy (np.ndarray): Noisy sensor measurements (n_sensors, n_times).
    - sensor_indices (list or np.ndarray, optional): Indices of sensors to plot. If None, plot the first sensor.
    - times (np.ndarray, optional): Time vector corresponding to the signals. If None, indices are used as the x-axis.
    - save_path (str, optional): Path to save the figure. If None, the figure is not saved.

    Returns:
    - None
    """
    if sensor_indices is None:
        sensor_indices = [0]  # Default to the first sensor

    if times is None:
        times = np.arange(y_clean.shape[1])  # Use indices as the x-axis if no time vector is provided

    # Create a figure with subplots for each sensor
    n_sensors = len(sensor_indices)
    fig, axes = plt.subplots(n_sensors, 1, figsize=(10, n_sensors * 3), sharex=True, sharey=True)
    fig.suptitle("Sensor Signals", fontsize=16)

    if n_sensors == 1:
        axes = [axes]  # Ensure axes is iterable for a single sensor

    for i, sensor_idx in enumerate(sensor_indices):
        axes[i].plot(times, y_clean[sensor_idx], label="y_clean", linewidth=2)
        axes[i].plot(times, y_noisy[sensor_idx], label="y_noise")
        axes[i].set_title(f"Sensor {sensor_idx}")
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("Amplitude  (µV)")
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()

    # Save the figure if a save path is provided
    if save_path:
        output_dir = Path(save_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to {save_path}")
    plt.show()
    plt.close()
   



leadfield_path = "results/fsaverage-leadfield-fixed.npz"  
with np.load(leadfield_path) as data:
    leadfield = data["leadfield"]

# Simulate data
results = simulate_timeseries_with_snr_source_space(
    leadfield,
    snr_db=10,
    sfreq=100,
    tmin=-0.5,
    tmax=0.5,
    stim_onset=0.0,
    nnz=5,
    orientation_type="fixed",
    seed=42,
    amplitude=5.0,
)

# Extract data
x = results["source_data"]
active_indices = results["active_indices"]

# Extract clean and noisy sensor data
y_clean = results["clean_sensor_data"]
y_noisy = results["sensor_data"]
times = results["times"]

# Plot active sources
plot_active_sources(
    x=x,
    times=times,
    active_indices=active_indices,
    stim_onset=0.0,
    nnz=5,
    save_path="results/figures/active_sources_subplots.png"
)


# Plot specific sensors
plot_sensor_signals(
    y_clean=y_clean,
    y_noisy=y_noisy,
    sensor_indices=[0, 10, 20],  # Plot sensors 0, 1, and 2
    times=times,
    save_path="results/figures/specific_sensor_signals.png"
)