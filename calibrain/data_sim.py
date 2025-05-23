import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.cm as cm # Import colormap functionality
from matplotlib.lines import Line2D # Import for custom legend
import os
import mne
import logging
logger = logging.getLogger(__name__)

# Generate a smoothed ERP-like signal using bandpass-filtered noise and a Hanning window
def _generate_erp_signal(n_times, sfreq, freq_band, amplitude, seed, onset_sample=0,
                        randomize_erp_timing_within_post_stim=True,
                        min_erp_duration_samples=None):
    """
    Generates an ERP-like signal, with activity starting at or after onset_sample.

    The actual ERP waveform (noise, filter, Hanning window) is applied to a
    segment of the signal. The signal before `onset_sample` will be zero.
    The Hanning window provides a smooth onset and offset for the activity.

    If `randomize_erp_timing_within_post_stim` is True, the exact start time
    (offset from `onset_sample`) and duration of the ERP waveform within the
    post-`onset_sample` window are randomized. The ERP will still be contained
    entirely within the `onset_sample` to `n_times` interval.

    If `randomize_erp_timing_within_post_stim` is False, the ERP waveform
    spans the entire duration from `onset_sample` to `n_times`.

    Parameters:
    - n_times (int): Total number of time points for the output signal.
    - sfreq (float): Sampling frequency.
    - freq_band (tuple): Frequency band for the bandpass filter (e.g., (1, 30)).
    - amplitude (float): Amplitude of the ERP.
    - seed (int or np.random.RandomState): Random seed or generator.
    - onset_sample (int, optional): The sample index within n_times at or after which
                                    ERP activity should occur. Defaults to 0.
    - randomize_erp_timing_within_post_stim (bool, optional): If True, randomizes
                                    the ERP's exact start and duration within the
                                    post-onset_sample window. Defaults to True.
    - min_erp_duration_samples (int, optional): Minimum number of samples for the
                                    ERP activity segment. If None, uses an internal
                                    default. This is used as the lower bound for
                                    randomized duration.

    Returns:
    - np.ndarray: The generated ERP signal of length n_times.
    """
    _DEFAULT_MIN_ERP_LEN = 82  # For filter stability (filtfilt butter order 4) & meaningful Hanning window

    if not isinstance(seed, np.random.RandomState):
        rng = np.random.RandomState(seed)
    else:
        rng = seed
    
    output_signal = np.zeros(n_times)
    
    current_min_erp_len = min_erp_duration_samples if min_erp_duration_samples is not None else _DEFAULT_MIN_ERP_LEN

    # Maximum available duration for ERP activity after onset_sample
    max_available_post_stim_duration = n_times - onset_sample

    if max_available_post_stim_duration < current_min_erp_len:
        # Not enough samples in the post-stimulus window for a meaningful ERP
        return output_signal

    actual_placement_start_sample: int
    n_times_for_erp_activity: int

    if randomize_erp_timing_within_post_stim:
        # Randomize ERP duration: from current_min_erp_len up to max_available_post_stim_duration (inclusive)
        actual_erp_duration = rng.randint(current_min_erp_len, max_available_post_stim_duration + 1)
        
        # Randomize ERP start offset within the available post-stimulus window
        # Max possible start offset (from onset_sample) for the chosen actual_erp_duration
        max_start_offset_from_onset = max_available_post_stim_duration - actual_erp_duration
        start_offset_from_onset = rng.randint(0, max_start_offset_from_onset + 1)
            
        actual_placement_start_sample = onset_sample + start_offset_from_onset
        n_times_for_erp_activity = actual_erp_duration
    else:
        # ERP spans the entire available post-stimulus duration
        n_times_for_erp_activity = max_available_post_stim_duration
        actual_placement_start_sample = onset_sample

    # Safeguard, though preceding logic should ensure this
    if n_times_for_erp_activity < current_min_erp_len:
        return output_signal

    # Generate noise only for the determined duration of the ERP activity
    white_noise_for_erp = rng.randn(n_times_for_erp_activity)
    
    # Design a Butterworth bandpass filter
    lowcut, highcut = freq_band
    low = lowcut / (sfreq / 2)
    high = highcut / (sfreq / 2)
    
    epsilon = 1e-9
    low = max(epsilon, low)
    high = min(1.0 - epsilon, high)
    if low >= high:
        return output_signal # Invalid frequency band

    try:
        b, a = butter(4, [low, high], btype='band')
    except ValueError as e:
        return output_signal # Filter design failed

    # Filter the noise segment
    erp_segment = filtfilt(b, a, white_noise_for_erp)
    
    # Apply Hanning window over the ERP segment
    erp_segment *= np.hanning(n_times_for_erp_activity) 
    
    std_erp_segment = np.std(erp_segment)
    if std_erp_segment < 1e-9: # Check if standard deviation is effectively zero
        return output_signal # Avoid division by zero; segment is flat
        
    erp_segment /= std_erp_segment # Normalize
    erp_segment *= amplitude      # Scale
    
    # Place the generated ERP segment into the output signal at the determined start
    end_sample_for_erp_segment = actual_placement_start_sample + len(erp_segment)
    
    # Ensure placement is within bounds (should be guaranteed by earlier logic)
    if actual_placement_start_sample < n_times and end_sample_for_erp_segment <= n_times:
        output_signal[actual_placement_start_sample : end_sample_for_erp_segment] = erp_segment
    
    return output_signal

# Function to project source activity to sensor space
def project_sources_to_sensors(
    source_activity: np.ndarray,
    leadfield: np.ndarray,
    orientation_type: str
) -> np.ndarray:
    """
    Projects source activity to sensor space using the leadfield matrix.

    Parameters:
    - source_activity (np.ndarray): The source activity.
        Shape (n_sources, n_times) for 'fixed' orientation.
        Shape (n_sources, 3, n_times) for 'free' orientation.
    - leadfield (np.ndarray): The leadfield matrix.
        Shape (n_sensors, n_sources) for 'fixed' orientation.
        Shape (n_sensors, n_sources, 3) for 'free' orientation.
        If 'free' and leadfield is 2D (n_sensors, n_sources*3), it will be reshaped.
    - orientation_type (str): 'fixed' or 'free'.

    Returns:
    - np.ndarray: The clean sensor data (n_sensors, n_times).
    """
    n_sensors = leadfield.shape[0]

    if orientation_type == "fixed":
        if source_activity.ndim != 2 or leadfield.ndim != 2:
            raise ValueError("For fixed orientation, source_activity must be 2D and leadfield must be 2D.")
        if leadfield.shape[1] != source_activity.shape[0]:
            raise ValueError(
                f"Leadfield n_sources ({leadfield.shape[1]}) mismatch with source_activity n_sources ({source_activity.shape[0]}) for fixed orientation."
            )
        y_clean = leadfield @ source_activity
    elif orientation_type == "free":
        if source_activity.ndim != 3 or source_activity.shape[1] != 3:
            raise ValueError("For free orientation, source_activity must be 3D with 3 orientations (n_sources, 3, n_times).")

        L_reshaped = leadfield
        if leadfield.ndim == 2:  # Potentially (n_sensors, n_sources*3)
            if leadfield.shape[1] % 3 != 0:
                raise ValueError("Leadfield for 'free' orientation (if 2D) must have n_sources*3 columns.")
            n_sources_calc = leadfield.shape[1] // 3
            if n_sources_calc != source_activity.shape[0]:
                raise ValueError(
                    f"Leadfield n_sources ({n_sources_calc}) mismatch with source_activity n_sources ({source_activity.shape[0]}) for free orientation after reshape."
                )
            L_reshaped = leadfield.reshape(n_sensors, n_sources_calc, 3)
        elif leadfield.ndim == 3:
            if leadfield.shape[2] != 3:
                raise ValueError("Leadfield for 'free' orientation (if 3D) must have 3 components in the last dimension.")
            if leadfield.shape[1] != source_activity.shape[0]:
                raise ValueError(
                    f"Leadfield n_sources ({leadfield.shape[1]}) mismatch with source_activity n_sources ({source_activity.shape[0]}) for free orientation."
                )
        else:
            raise ValueError("Leadfield for 'free' orientation must be 2D or 3D.")

        # Project to sensor space: (n_sensors, n_sources, 3) @ (n_sources, 3, n_times) -> (n_sensors, n_times)
        y_clean = np.einsum("nsr,srt->nt", L_reshaped, source_activity)
    else:
        raise ValueError(f"Invalid orientation_type: {orientation_type}. Choose 'fixed' or 'free'.")
    return y_clean


def _generate_source_time_courses( 
    leadfield,
    sfreq=100,
    tmin=-0.5,
    tmax=0.5,
    stim_onset=0.0,
    nnz=5,
    orientation_type="fixed",
    seed=None,
    amplitude=5.0,
    freq_band=(1, 30),
    randomize_erp_timing_within_post_stim=True, # Added for consistency
    min_erp_duration_samples=None # Added for consistency
):
    """ Generates true source activity time courses. """
    if not isinstance(seed, np.random.RandomState):
        rng = np.random.RandomState(seed)
    else:
        rng = seed
        
    times = np.arange(tmin, tmax, 1.0 / sfreq)
    n_times = len(times)

    stim_indices = np.where(times >= stim_onset)[0]
    stim_idx_for_erp_onset = n_times if not stim_indices.size else stim_indices[0]

    x: np.ndarray 
    active_indices: np.ndarray

    if orientation_type == "fixed":
        n_sources = leadfield.shape[1]
        active_indices = rng.choice(n_sources, size=nnz, replace=False)
        x = np.zeros((n_sources, n_times))
        for i, src_idx in enumerate(active_indices):
            source_rng_seed = rng.randint(0, 2**32 -1) 
            erp_waveform = _generate_erp_signal(
                n_times, sfreq, freq_band, amplitude,
                seed=source_rng_seed, 
                onset_sample=stim_idx_for_erp_onset,
                randomize_erp_timing_within_post_stim=randomize_erp_timing_within_post_stim,
                min_erp_duration_samples=min_erp_duration_samples
            )
            x[src_idx, :] = erp_waveform
    elif orientation_type == "free":
        if leadfield.ndim == 3:
            n_sources = leadfield.shape[1]
        elif leadfield.ndim == 2 and leadfield.shape[1] % 3 == 0:
            n_sources = leadfield.shape[1] // 3
        else:
            raise ValueError(f"Cannot determine n_sources for free orientation from leadfield shape {leadfield.shape}")
        
        active_indices = rng.choice(n_sources, size=nnz, replace=False)
        x = np.zeros((n_sources, 3, n_times))
        for i, src_idx in enumerate(active_indices):
            source_rng_seed = rng.randint(0, 2**32 -1)
            erp_waveform = _generate_erp_signal(
                n_times, sfreq, freq_band, amplitude,
                seed=source_rng_seed,
                onset_sample=stim_idx_for_erp_onset,
                randomize_erp_timing_within_post_stim=randomize_erp_timing_within_post_stim,
                min_erp_duration_samples=min_erp_duration_samples
            )
            orient_coeffs = rng.randn(3)
            norm_orient = np.linalg.norm(orient_coeffs)
            orient_coeffs = np.array([1.0, 0.0, 0.0]) if norm_orient < 1e-9 else orient_coeffs / norm_orient
            for j_orient in range(3):
                x[src_idx, j_orient, :] = orient_coeffs[j_orient] * erp_waveform
    else:
        raise ValueError("Invalid orientation_type. Choose 'fixed' or 'free'.")

    return x, active_indices, times


# Function to add noise based on SNR
def _add_noise(y_clean, snr_db, rng=None):
    """
    Adds Gaussian noise to a clean signal based on a desired SNR level.

    Parameters:
    - y_clean (np.ndarray): The clean signal array (e.g., channels x times).
    - snr_db (float): The desired signal-to-noise ratio in decibels (dB).
    - rng (np.random.RandomState, optional): A random number generator state
      for reproducible noise generation. If None, uses default numpy random state.

    Returns:
    - tuple: A tuple containing:
        - y_noisy (np.ndarray): The signal with added noise.
        - noise (np.ndarray): The generated noise array.
        - noise_power (float): The calculated variance of the added noise.
    """
    if rng is None:
        rng = np.random # Use default numpy random state if none provided

    signal_power = np.mean(y_clean ** 2)
    if signal_power == 0:
        print("Warning: Clean signal power is zero. Cannot add noise based on SNR.")
        noise_power = 0
        noise = np.zeros_like(y_clean)
    else:
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power)
        noise = rng.normal(0, noise_std, size=y_clean.shape)

    y_noisy = y_clean + noise
    return y_noisy, noise, noise_power


def inspect_matrix_values(matrix, matrix_name="Matrix"):
    """
    Prints summary statistics and checks for invalid values in a NumPy array.

    Parameters:
    - matrix (np.ndarray): The matrix to inspect.
    - matrix_name (str): A name for the matrix used in print statements.
    """
    print(f"--- Inspecting {matrix_name} Values ---")
    if not isinstance(matrix, np.ndarray):
        print(f"Error: Input is not a NumPy array.")
        return
    if matrix.size == 0:
        print(f"Warning: {matrix_name} is empty.")
        return

    try:
        min_val = np.min(matrix)
        max_val = np.max(matrix)
        mean_val = np.mean(matrix)
        mean_abs_val = np.mean(np.abs(matrix))
        std_val = np.std(matrix)

        print(f"{matrix_name} mean: {mean_val:.2e}, std: {std_val:.2e}")
        print(f"{matrix_name} min: {min_val:.2e}, max: {max_val:.2e}")
        # print(f"{matrix_name} std: {std_val:.1e}") # Redundant with first line
        print(f"{matrix_name} mean abs: {mean_abs_val:.2e}")

        nan_check = np.isnan(matrix).any()
        inf_check = np.isinf(matrix).any()

        if nan_check:
            print(f"WARNING: {matrix_name} contains NaN values!")
        if inf_check:
            print(f"WARNING: {matrix_name} contains Inf values!")
        if not nan_check and not inf_check:
            print(f"{matrix_name} contains valid numbers (no NaNs or Infs detected).")

    except Exception as e:
        print(f"Error during inspection of {matrix_name}: {e}")
    print(f"--- End {matrix_name} Inspection ---")

def load_and_validate_leadfield(leadfield_file_path, orientation_type):
    """
    Loads a leadfield matrix from an .npz file and validates its shape
    based on the expected orientation type. Includes value inspection.

    Parameters:
    - leadfield_file_path (str or Path): Path to the .npz file containing the leadfield.
    - orientation_type (str): The expected orientation type ("fixed" or "free").

    Returns:
    - np.ndarray: The loaded and validated leadfield matrix.

    Raises:
    - FileNotFoundError: If the leadfield file does not exist.
    - KeyError: If the expected key is not found in the .npz file.
    - ValueError: If the loaded leadfield matrix shape is inconsistent with the orientation_type.
    - Exception: For other potential loading errors.
    """
    print(f"Loading leadfield from: {leadfield_file_path}")
    try:
        with np.load(leadfield_file_path) as data:
            # ... (loading logic as before) ...
            if 'leadfield' in data:
                 leadfield_matrix = data["leadfield"]
            elif 'leadfield_fixed' in data and orientation_type == "fixed":
                 leadfield_matrix = data['leadfield_fixed']
            elif 'leadfield_free' in data and orientation_type == "free":
                 leadfield_matrix = data['leadfield_free']
            elif 'leadfield' in data:
                 print("Warning: Loading generic 'leadfield' key. Ensure it matches orientation type.")
                 leadfield_matrix = data["leadfield"]
            else:
                 keys_found = list(data.keys())
                 raise KeyError(f"Could not find a suitable leadfield key ('leadfield', 'leadfield_fixed', 'leadfield_free') in .npz file. Found keys: {keys_found}")

        print(f"Leadfield loaded successfully. Initial Shape: {leadfield_matrix.shape}", "dtype:", leadfield_matrix.dtype)

        # --- Validate leadfield shape against orientation_type ---
        # ... (validation logic as before) ...
        if orientation_type == "fixed":
            if leadfield_matrix.ndim != 2:
                raise ValueError(f"Expected 2D leadfield for fixed orientation, got shape {leadfield_matrix.shape}")
        elif orientation_type == "free":
            if leadfield_matrix.ndim == 3:
                if leadfield_matrix.shape[2] != 3:
                     raise ValueError(f"Expected 3 components in last dimension for free orientation, got shape {leadfield_matrix.shape}")
            elif leadfield_matrix.ndim == 2:
                 if leadfield_matrix.shape[1] % 3 == 0:
                      print("Warning: Reshaping potentially flattened free orientation leadfield.")
                      n_sensors, n_sources_x_3 = leadfield_matrix.shape
                      n_sources = n_sources_x_3 // 3
                      leadfield_matrix = leadfield_matrix.reshape(n_sensors, n_sources, 3)
                      print(f"Reshaped leadfield to {leadfield_matrix.shape}")
                 else:
                      raise ValueError(f"Cannot reshape 2D leadfield (shape {leadfield_matrix.shape}) to 3D free orientation.")
            else:
                 raise ValueError(f"Expected 2D or 3D leadfield for free orientation, got {leadfield_matrix.ndim} dimensions with shape {leadfield_matrix.shape}")
        else:
             raise ValueError(f"Invalid orientation_type specified: {orientation_type}. Choose 'fixed' or 'free'.")


        print(f"Leadfield validated successfully. Final Shape: {leadfield_matrix.shape}")

        # --- Inspect Leadfield Matrix Values using the function ---
        inspect_matrix_values(leadfield_matrix, matrix_name="Leadfield")
        # --- End Inspection ---

        return leadfield_matrix

    except FileNotFoundError:
        print(f"Error: Leadfield file not found at {leadfield_file_path}")
        raise # Re-raise the exception
    except (KeyError, ValueError) as e:
        print(f"Error loading or validating leadfield: {e}")
        raise # Re-raise the specific error
    except Exception as e:
        print(f"An unexpected error occurred during leadfield loading: {e}")
        raise 


# --- Plotting Functions ---
def plot_sensor_signals(y_clean, y_noisy, sensor_indices=None, times=None, save_dir=None, figure_name=None, trial_idx=None):
    """ Plot clean and noisy sensor signals for specific sensors for a specific trial. """
    if sensor_indices is None:
        sensor_indices = [0]
    if times is None:
        times = np.arange(y_clean.shape[1])

    n_sensors_to_plot = len(sensor_indices)
    fig, axes = plt.subplots(n_sensors_to_plot, 1, figsize=(10, n_sensors_to_plot * 3), sharex=True, sharey=True)
    title_suffix = f" (Trial {trial_idx+1})" if trial_idx is not None else ""
    fig.suptitle(f"Specific Sensor Signals{title_suffix}", fontsize=16)

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
    plt.close(fig)

def plot_all_active_sources_single_figure(x, times, active_indices, stim_onset, save_dir=None, figure_name=None, trial_idx=None):
    """ Plot all specified active source signals on a single figure for a specific trial. """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    title_suffix = f" (Trial {trial_idx+1})" if trial_idx is not None else ""
    fig.suptitle(f"All Active Source Signals{title_suffix}", fontsize=16)
    colors = cm.viridis(np.linspace(0, 1, len(active_indices)))

    # Handle potential free orientation source data shape
    if x.ndim == 3:
        x_plot = np.linalg.norm(x, axis=1) # Plot magnitude
    else:
        x_plot = x

    for i, src_idx in enumerate(active_indices):
        ax.plot(times, x_plot[src_idx], label=f"Source {src_idx}", linewidth=1.5, color=colors[i])

    ax.axvline(x=stim_onset, linestyle="--", color="gray", label="Stimulus Onset")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (nAm)")
    ax.legend(loc='best', fontsize='small')
    ax.grid(True, alpha=0.6)
    ax.set_title("Active Sources")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_dir and figure_name:
        output_dir = Path(save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"{figure_name}.png"
        plt.savefig(save_path, dpi=300)
        print(f"Single figure source plot saved to {save_path}")
    plt.close(fig)

def plot_all_sensor_signals_single_figure(y_data, times, sensor_indices=None, save_dir=None, figure_name=None, trial_idx=None, average_epochs=False):
    """
    Plot sensor signals (overlay) for selected sensors.
    If average_epochs is True and y_data is 3D, plots the average across epochs for each channel.
    If average_epochs is False and y_data is 2D, plots the single trial data.
    Does NOT average across channels.

    Parameters:
    - y_data (np.ndarray): Sensor measurements. Can be 2D (n_channels, n_times) for a single trial
                           or 3D (n_trials, n_channels, n_times) for multiple trials.
    - times (np.ndarray): Time vector corresponding to the signals.
    - sensor_indices (list or np.ndarray, optional): Indices of sensors to plot. If None, plots all sensors.
    - save_dir (str or Path, optional): Directory to save the figure.
    - figure_name (str, optional): Name of the figure file (without extension).
    - trial_idx (int, optional): Index of the trial being plotted (used for title if y_data is 2D and average_epochs is False).
    - average_epochs (bool): If True and y_data is 3D, plot the average across trials.
                             If False and y_data is 3D, raises an error.
                             If y_data is 2D, this primarily affects the title.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    title_suffix = ""
    plot_individual_epochs = False # Flag to control plotting individual trials (currently always False)

    if y_data.ndim == 2: # Input is single trial or already averaged data
        y_plot = y_data # This is the data to plot (n_channels, n_times)
        if not average_epochs and trial_idx is not None:
             title_suffix = f" (Trial {trial_idx+1})"
        elif average_epochs: # Assume 2D input might be an average if flag is set
             title_suffix = " (Average across Trials)"
        # If 2D and not average_epochs and no trial_idx, title is generic
    elif y_data.ndim == 3: # Input is multi-trial data
        if average_epochs:
            y_plot = np.mean(y_data, axis=0) # Calculate average across trials (axis 0) -> shape (n_channels, n_times)
            title_suffix = " (Average across Trials)"
            # Do not plot individual epochs if averaging is requested
            plot_individual_epochs = False
        else:
            # If 3D data is passed but averaging is not requested, it's ambiguous.
            raise ValueError("Input y_data is 3D, but average_epochs is False. "
                             "Provide 2D data (single trial) or set average_epochs=True.")
    else:
        raise ValueError("Input y_data must be 2D or 3D")

    # Select specific sensors if requested from the data to be plotted (y_plot)
    if sensor_indices is None:
        sensor_indices_to_plot = np.arange(y_plot.shape[0]) # Use all channels
        y_plot_selected = y_plot
    else:
        # Ensure indices are valid for the potentially averaged data
        sensor_indices_to_plot = np.array(sensor_indices)[np.array(sensor_indices) < y_plot.shape[0]]
        if len(sensor_indices_to_plot) != len(sensor_indices):
             print("Warning: Some requested sensor_indices are out of bounds for the provided data.")
        y_plot_selected = y_plot[sensor_indices_to_plot, :]

    n_plot_sensors = y_plot_selected.shape[0]

    fig.suptitle(f"Sensor Signals {title_suffix}", fontsize=16)
    colors = cm.turbo(np.linspace(0, 1, n_plot_sensors))

    # --- Plotting Logic ---
    # Plot the main traces (either single trial or trial-averaged)
    for i in range(n_plot_sensors):
        actual_sensor_idx = sensor_indices_to_plot[i] # Get original index
        ax.plot(times, y_plot_selected[i, :], linewidth=1.0, color=colors[i], alpha=0.8, label=f"Ch {actual_sensor_idx}" if n_plot_sensors <= 15 else None)

    # Optional: Plot individual epoch traces lightly in the background (currently disabled)
    if plot_individual_epochs and y_data.ndim == 3:
         y_plot_all_selected = y_data[:, sensor_indices_to_plot, :] # Select sensors from original 3D data
         for i_trial in range(y_data.shape[0]):
             for i_ch in range(n_plot_sensors):
                 ax.plot(times, y_plot_all_selected[i_trial, i_ch, :], linewidth=0.2, color=colors[i_ch], alpha=0.1)


    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.grid(True, alpha=0.6)
    ax.set_title(f"{n_plot_sensors} channels")

    # Update legend
    if n_plot_sensors <= 15: # Show legend only for fewer channels
        ax.legend(loc='best', fontsize='small')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_dir and figure_name:
        output_dir = Path(save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"{figure_name}.png"
        plt.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_active_sources(x, times, active_indices, stim_onset, nnz, save_dir=None, figure_name=None, trial_idx=None):
    """ Plot active sources for a specific trial. """
    n_cols = 3
    n_rows = int(np.ceil(nnz / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), constrained_layout=True, sharex=True, sharey=True)
    title_suffix = f" (Trial {trial_idx+1})" if trial_idx is not None else ""
    fig.suptitle(f"Active Source Signals{title_suffix}", fontsize=16)
    axes = axes.flatten()

    # Handle potential free orientation source data shape (n_sources, n_orient, n_times)
    # Plot the norm or the first component for simplicity
    if x.ndim == 3:
        x_plot = np.linalg.norm(x, axis=1) # Plot magnitude for free orientation
        # Or plot first component: x_plot = x[:, 0, :]
    else:
        x_plot = x

    for i, src_idx in enumerate(active_indices):
        axes[i].plot(times, x_plot[src_idx], label=f"Source {src_idx}", linewidth=2)
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
    plt.close(fig)



# --------------------------------------------------------------


def main():

    # --- Simulation Parameters ---
    leadfield_file_path = "results/forward/fsaverage-leadfield-fixed.npz"
    orientation_type = "fixed" # "fixed" or "free"
    tmin = -0.5
    tmax = 0.5
    stim_onset = 0.0
    sfreq = 250
    fmin = 1
    fmax = 5
    freq_band = (fmin, fmax)
    snr_db = 6 # SNR for noise addition
    amplitude = 5.0 # nAm
    nnz = 5 # Number of active sources per trial
    seed = 42
    n_trials = 10 # <<<--- Number of trials to simulate

    leadfield_matrix = load_and_validate_leadfield(leadfield_file_path, orientation_type)

    # --- Plotting Parameters ---
    figure_save_dir = Path("results/figures/data_sim") # Define base directory for figures
    os.makedirs(figure_save_dir / "ERP_channels" , exist_ok=True)
    os.makedirs(figure_save_dir / "PSD_epochs" , exist_ok=True)
    sensor_subplots_indices = [0, 10, 20] # Indices for the subplot sensor plot

    # --- Simulate Multiple Trials (Clean Data First) ---
    print(f"Simulating {n_trials} trials (clean data)...")
    y_clean_all_trials = [] # Store clean sensor data per trial
    x_all_trials = [] # Store source data per trial
    active_indices_all_trials = [] # Store active indices per trial

    for i_trial in range(n_trials):
        print(f"  Simulating clean trial {i_trial + 1}/{n_trials}")
        trial_seed = seed + i_trial if seed is not None else None

        # Call the function that generates true source activity
        x_trial, active_indices, time_vector = _generate_source_time_courses(
            leadfield=leadfield_matrix, # Still pass leadfield for shape info
            sfreq=sfreq,
            tmin=tmin,
            tmax=tmax,
            stim_onset=stim_onset,
            nnz=nnz,
            orientation_type=orientation_type,
            seed=trial_seed,
            amplitude=amplitude,
            freq_band=freq_band
        )
        

        # Project source activity to sensors
        y_clean_trial = project_sources_to_sensors(
            source_activity=x_trial,
            leadfield=leadfield_matrix,
            orientation_type=orientation_type
        )

        y_clean_all_trials.append(y_clean_trial)
        x_all_trials.append(x_trial) # Store the source data
        active_indices_all_trials.append(active_indices)
    print("Clean simulation complete.")

    # Convert lists to NumPy arrays
    y_clean_all_trials = np.array(y_clean_all_trials) # Shape: (n_trials, n_channels, n_times)
    x_all_trials = np.array(x_all_trials) # Shape: (n_trials, n_sources, [n_orient,] n_times)
    active_indices_all_trials = np.array(active_indices_all_trials) # Shape: (n_trials, nnz)
    time_vector = np.array(time_vector) # Shape: (n_times,)

    # --- Add Noise Separately ---
    print(f"Adding noise (SNR={snr_db} dB) to clean trials...")
    y_all_trials = [] # To store noisy data
    noise_all_trials = [] # Optional: store noise itself
    noise_power_all_trials = [] # Optional: store noise power per trial

    noise_rng = np.random.RandomState(seed + n_trials) # Use a separate seed/state for noise if desired

    for i_trial in range(n_trials):
        # Get the clean data for this trial
        y_clean_trial = y_clean_all_trials[i_trial]

        # Add noise using the dedicated function
        # Use a trial-specific RNG state derived from noise_rng for reproducibility per trial
        trial_noise_rng = np.random.RandomState(noise_rng.randint(0, 2**32 - 1))
        y_noisy_trial, noise_trial, noise_power_trial = _add_noise(
            y_clean_trial,
            snr_db,
            rng=trial_noise_rng
        )

        y_all_trials.append(y_noisy_trial)
        noise_all_trials.append(noise_trial) # Optional
        noise_power_all_trials.append(noise_power_trial) # Optional

    # Convert noisy data list to NumPy array
    y_all_trials = np.array(y_all_trials) # Shape: (n_trials, n_channels, n_times)
    print("Noise addition complete.")











    # # Convert lists to NumPy arrays
    # y_all_trials_clean = np.array(y_clean_all_trials) # Shape: (n_trials, n_channels, n_times)
    # x_all_trials = np.array(x_all_trials) # Shape: (n_trials, n_sources, [n_orient,] n_times)
    # active_indices_all_trials = np.array(active_indices_all_trials) # Shape: (n_trials, nnz)
    # time_vector = np.array(time_vector) # Shape: (n_times,)


    # --- Plotting Examples ---

    # Plot data from the first trial
    first_trial_idx = 0
    print(f"\nPlotting results for trial {first_trial_idx + 1}...")

    # Now plot_sensor_signals uses the clean and noisy data generated separately
    plot_sensor_signals(
        y_clean=y_clean_all_trials[first_trial_idx], # Use stored clean data
        y_noisy=y_all_trials[first_trial_idx],       # Use stored noisy data
        sensor_indices=sensor_subplots_indices,
        times=time_vector,
        save_dir=figure_save_dir,
        figure_name=f"specific_sensor_signals_subplots_trial{first_trial_idx+1}",
        trial_idx=first_trial_idx
    )

    plot_all_active_sources_single_figure(
        x=x_all_trials[first_trial_idx],
        times=time_vector,
        active_indices=active_indices_all_trials[first_trial_idx],
        stim_onset=stim_onset,
        save_dir=figure_save_dir,
        figure_name=f"active_sources_single_figure_trial{first_trial_idx+1}",
        trial_idx=first_trial_idx
    )

    # Plot data from the first trial (y_all_trials[first_trial_idx] is 2D)
    # average_epochs=False tells the function to treat the 2D data as a single trial
    plot_all_sensor_signals_single_figure(
        y_data=y_all_trials[first_trial_idx], # Pass single trial noisy data (2D)
        times=time_vector,
        save_dir=figure_save_dir,
        figure_name=f"all_sensor_signals_single_figure_trial{first_trial_idx+1}",
        trial_idx=first_trial_idx,
        average_epochs=False
    )

    # Plot average across trials (y_all_trials is 3D)
    # average_epochs=True tells the function to average the 3D data across trials
    plot_all_sensor_signals_single_figure(
        y_data=y_all_trials, # Pass all trial noisy data (3D array)
        times=time_vector,
        save_dir=figure_save_dir,
        figure_name="all_sensor_signals_average_trials",
        average_epochs=True
    )    


    plot_active_sources(
        x=x_all_trials[first_trial_idx],
        times=time_vector,
        active_indices=active_indices_all_trials[first_trial_idx],
        stim_onset=stim_onset,
        nnz=nnz, # Pass nnz used in simulation
        save_dir=figure_save_dir,
        figure_name=f"active_sources_subplots_trial{first_trial_idx+1}",
        trial_idx=first_trial_idx
    )



    # --- Create MNE Epochs Object ---
    print("\nCreating MNE Epochs object...")

    n_times = len(time_vector)


    montage = mne.channels.make_standard_montage("easycap-M43")
    info = mne.create_info(ch_names=montage.ch_names, sfreq=sfreq, ch_types="eeg")
    info.set_montage(montage)

    # Structure: [sample_index_within_epoch, previous_event_id, event_id]
    # Since the stimulus onset is at the same time relative to the start (tmin)
    # for all simulated trials, the sample index within the epoch is the same.
    event_id = 1 # Define an event ID for the stimulus onset

    # Find the sample index corresponding to stim_onset within the time_vector
    stim_sample_in_epoch = np.argmin(np.abs(time_vector - stim_onset))
    events = np.zeros((n_trials, 3), int)

    for i_trial in range(n_trials):
        events[i_trial, 0] = stim_sample_in_epoch + i_trial * n_times

    # Assign the event ID to the third column for all trials
    events[:, 2] = event_id
    print("Generated events array for EpochsArray:")
    print(events) # Print to confirm

    # Create EpochsArray object
    # Data units are µV. MNE plotting functions usually handle this, but be aware.
    epochs = mne.EpochsArray(y_all_trials,
                            info,
                            events=events,
                            tmin=tmin,
                            event_id={'stim': event_id})
    print("\nMNE Epochs object created:")



    # -------------------------------------
    # --- Plot using MNE Epochs methods ---
    print("\nPlotting using MNE Epochs methods...")
    fig_epochs = epochs.plot(show=True, picks=info.ch_names, n_epochs=n_trials, n_channels=5, events=True, scalings=dict(eeg=2e3), event_color="red")
    fig_epochs.suptitle("Epochs across subset of channels (with Events Onset marked)", fontsize=16)
    plt.tight_layout()
    fig_epochs.savefig(figure_save_dir / "mne_epochs_plot.png")
    plt.close(fig_epochs)


    # ------------------------------------




    # ------------------------------------
    # spatial_colors=True helps visualize different channels
    fig_erp = epochs.average().plot(spatial_colors=True, show=False)
    fig_erp.suptitle("Average ERP across trials", fontsize=16)
    fig_erp.savefig(figure_save_dir / "mne_average_erp_plot.png")
    plt.close(fig_erp) # Close the figure after saving



    # -------------------------------------
    if True:
        # plot_image shows each trial as a row in an image map (Plot ER[P/F])
        # Draw the ER[P/F] below the image or not.
        figs_img = epochs.plot_image(show=False, combine=None, group_by=None, picks=info.ch_names, colorbar=True, title="Epochs ER[P/F]", evoked=True)


        for i, fig in enumerate(figs_img):
            fig.suptitle(f"Epochs Image Plot (channel {i+1})", fontsize=16)
            if figure_save_dir:
                image_save_path = figure_save_dir / "ERP_channels" / f"mne_epochs_image_plot_channel_{i+1}.png"
                fig.savefig(image_save_path)


    figs_img = epochs.plot_image(show=False, combine="mean", group_by=None, picks=info.ch_names, colorbar=True, title="Epochs ER[P/F] - average across channels", evoked=True)

    figs_img[0].savefig(figure_save_dir / "mne_epochs_image_plot_mean.png")
    for fig in figs_img:
            plt.close(fig)


    # -------
    # Plot sensor locations (based on montage)
    print("Plotting sensor locations...")
    fig_topo = epochs.plot_sensors(show_names=True, show=False, kind='topomap', ch_type='eeg')
    fig_topo.savefig(figure_save_dir / "mne_sensor_locations.png")
    plt.close(fig_topo)


    # -------------------------------



    for i in range(len(epochs)):
        fig = epochs[i].compute_psd(fmax=30).plot(
            average=False, # average across channels
            picks=info.ch_names,
            show=False,
            spatial_colors=True, 
        )
        fig.suptitle(f"PSD per channels - Epoch {i+1}", fontsize=16)
        fig.savefig(figure_save_dir / "PSD_epochs" / f"epoch{i}_psd_plot.png")
        plt.close(fig)



    fig = epochs.compute_psd(fmax=30).plot(
        average=False,
        picks=info.ch_names,
        show=False,
        spatial_colors=True, 
    )
    fig.suptitle("PSD per channels - averaged across epochs", fontsize=16)
    fig.savefig(figure_save_dir / "mne_psd_avg_across_epochs.png")
    plt.close(fig)


    # -# -------------------------------------
    # compute and plot time freuquency representation (TFR)
    # TFR using Morlet wavelets
    freqs = np.arange(fmin, fmax, 1)  # Frequencies from 1 to 100 Hz
    n_cycles = freqs / 2.0  # Different number of cycles per frequency
    time_bandwidth = 2.0  # Time-bandwidth product for Morlet wavelets

    fig = epochs.compute_tfr(
        method="multitaper",
        freqs=freqs,
        n_cycles=n_cycles,
        time_bandwidth=time_bandwidth,
        return_itc=False,
        average=True,  # Average across epochs
        decim=3,  # Decimate the data for faster computation
        picks=info.ch_names[0],
    ).plot(picks=[0], mode="mean", title="TFR - (Average across epochs, channel 1)", colorbar=True, show=False)

    fig[0].savefig(figure_save_dir / "tfr_average_across_epochs.png")
    plt.close(fig[0])



    fig0 = epochs[0].compute_tfr(
        method="multitaper",
        freqs=freqs,
        n_cycles=n_cycles,
        time_bandwidth=time_bandwidth,
        return_itc=False,
        average=False,  # Average across epochs
        decim=3,  # Decimate the data for faster computation
        picks=info.ch_names[0],
    ).plot([0], baseline=(None, None), title="TFR - (epoch 1, channel 1)", colorbar=True, show=False)

    fig0[0].savefig(figure_save_dir / "tfr_epoch0.png")
    plt.close(fig0[0])

    print("\n1")
    
if __name__ == "__main__":
    main()