
import numpy as np
from scipy.stats import wishart
from pathlib import Path
import logging

from calibrain import LeadfieldSimulator
from calibrain.utils import load_config

class DataSimulator:
    def __init__(
        self,
        n_sensors=50,
        n_sources=100,
        n_times=10,
        nnz=3,
        orientation_type="fixed",
        alpha_snr=0.99,
        noise_type="baseline",
        seed=None,
        logger=None,
        rng=None,
        leadfield_mode="random",
        leadfield_path=None,
        leadfield_config_path=None,
    ):
        """
        Initialize the DataSimulator class.
        Parameters:
        - n_sensors (int): Number of sensors.
        - n_sources (int): Number of sources.
        - n_times (int): Number of time points.
        - nnz (int): Number of non-zero sources.
        - orientation_type (str): Orientation type ('fixed' or 'free').
        - alpha_snr (float): Alpha SNR value.
        - noise_type (str): Type of noise ('random' or 'scaled_identity').
        - seed (int): Random seed for reproducibility.
        - logger (logging.Logger): Logger instance for logging messages.
        - rng (np.random.Generator): Random number generator instance.
        - leadfield_mode (str): Mode for leadfield generation ('load', 'simulate', or 'random'):
            - "load": Load the leadfield matrix from a file.
            - "simulate": Simulate the leadfield matrix using LeadfieldSimulator.
            - "random": Generate a random leadfield matrix.
        - leadfield_path (str): Path to the leadfield file for loading.
        - leadfield_config_path (str): Path to the configuration file for simulating leadfield.
        """
        
        self.n_sensors = n_sensors
        self.n_sources = n_sources
        self.n_times = n_times
        self.nnz = nnz
        self.orientation_type = orientation_type
        self.alpha_snr = alpha_snr
        self.noise_type = noise_type
        self.seed = seed
        self.logger = logger if logger else logging.getLogger(__name__)
        self.rng = rng
        self.leadfield_mode = leadfield_mode
        self.leadfield_path = leadfield_path
        self.leadfield_config_path = leadfield_config_path

    def _get_leadfield(self):
        """
        Get the leadfield matrix based on the specified leadfield_mode.
    
        Returns:
        - leadfield (np.ndarray): The leadfield matrix.
    
        Raises:
        - ValueError: If an invalid leadfield_mode is specified or required arguments are missing.
        """
        # Determine expected suffix and dimensions based on orientation type
        expected_suffix = "-free.npz" if self.orientation_type == "free" else "-fixed.npz"
        expected_dimensions = 3 if self.orientation_type == "free" else 2
    
        if self.leadfield_mode == "load":
            if not self.leadfield_path:
                raise ValueError("Path to the leadfield file must be provided when leadfield_mode='load'.")
            load_path = Path(self.leadfield_path)
    
            try:
                if not load_path.exists():
                    raise FileNotFoundError(f"Leadfield file does not exist: {load_path}")
    
                if not load_path.name.endswith(expected_suffix):
                    raise ValueError(f"Leadfield file name '{load_path.name}' does not match expected suffix '{expected_suffix}'.")
    
                self.logger.info(f"Loading leadfield matrix from file: {load_path}")
                with np.load(load_path) as data:
                    if "leadfield" not in data:
                        raise ValueError(f"File {load_path} does not contain 'leadfield' key.")
                    leadfield = data["leadfield"]
    
                # Validate the shape of the leadfield matrix
                if leadfield.ndim != expected_dimensions:
                    raise ValueError(
                        f"Leadfield matrix shape mismatch: expected {expected_dimensions} dimensions, "
                        f"but got {leadfield.ndim} dimensions."
                    )
    
                self.logger.info(f"Leadfield loaded with shape {leadfield.shape}")
    
            except (FileNotFoundError, ValueError) as e:
                self.logger.error(f"Failed to load leadfield matrix: {e}")
                raise
    
        elif self.leadfield_mode == "simulate":
            if not self.leadfield_config_path:
                raise ValueError("Path to the configuration file must be provided when leadfield_mode='simulate'.")
            self.logger.info(f"Simulating leadfield matrix using LeadfieldSimulator with config: {self.leadfield_config_path}")
    
            config = load_config(Path(self.leadfield_config_path))
            simulator = LeadfieldSimulator(config=config, logger=self.logger)
            leadfield = simulator.simulate()
            self.logger.info(f"Simulated leadfield matrix with shape {leadfield.shape}")
    
            # Validate the shape of the leadfield matrix
            if leadfield.ndim != expected_dimensions:
                raise ValueError(
                    f"Leadfield matrix shape mismatch: expected {expected_dimensions} dimensions, "
                    f"but got {leadfield.ndim} dimensions. Check the leadfield config file if the orientation type aligns with the intended simulation."
                )
    
        elif self.leadfield_mode == "random":
            self.logger.info("Generating a random leadfield matrix.")
            if self.orientation_type == "fixed":
                leadfield = self.rng.randn(self.n_sensors, self.n_sources)
            else:  # Free orientation
                leadfield = self.rng.randn(self.n_sensors, self.n_sources, 3)
            self.logger.info(f"Random leadfield generated with shape {leadfield.shape}")
    
        else:
            raise ValueError(f"Invalid leadfield mode '{self.leadfield_mode}'. Options are 'load', 'simulate', or 'random'.")
    
        # Extract n_sensors and n_sources from the leadfield shape
        if self.orientation_type == "fixed":
            self.n_sensors, self.n_sources = leadfield.shape
        elif self.orientation_type == "free":
            self.n_sensors, self.n_sources, _ = leadfield.shape
    
        self.logger.info(f"Extracted n_sensors={self.n_sensors}, n_sources={self.n_sources}")
        return leadfield

    def _generate_source_time_courses(self):
        """
        Generate synthetic source time courses in the source space.
    
        Returns:
        - x (np.ndarray): Source time courses.
            - For "fixed" orientation: Shape `(n_sources, n_times)`.
            - For "free" orientation: Shape `(n_sources, 3, n_times)`, where the second dimension corresponds to the three orientations (X, Y, Z).
    
        Notes:
        - The number of active sources (`nnz`) is randomly selected using `self.rng.choice`.
        - If `n_times = 1`, the output shapes of `x` remain consistent with the multi-time-point case, ensuring compatibility with downstream processing.
        """
        # Validate orientation type
        if self.orientation_type not in ["fixed", "free"]:
            raise ValueError(f"Unsupported orientation type: {self.orientation_type}")
    
        # Initialize source activity matrix
        if self.orientation_type == "fixed":
            x = np.zeros((self.n_sources, self.n_times))
        elif self.orientation_type == "free":
            n_orient = 3
            x = np.zeros((self.n_sources, n_orient, self.n_times))
    
        # Select random indices for active sources
        idx = self.rng.choice(self.n_sources, size=self.nnz, replace=False)
    
        # Generate random activity for active sources
        if self.orientation_type == "fixed":
            x[idx] = self.rng.randn(self.nnz, self.n_times)
        elif self.orientation_type == "free":
            x[idx] = self.rng.randn(self.nnz, n_orient, self.n_times)
    
        return x
    
    
    def _project_to_sensor_space(self, L, x):
        """
        Project the source activity to the sensor space using the leadfield matrix.
    
        Parameters:
        - L (np.ndarray): Leadfield matrix.
            - For "fixed" orientation: Shape `(n_sensors, n_sources)`.
            - For "free" orientation: Shape `(n_sensors, n_sources, 3)`.
        - x (np.ndarray): Source activity.
            - For "fixed" orientation: Shape `(n_sources, n_times)`.
            - For "free" orientation: Shape `(n_sources, 3, n_times)`.
    
        Returns:
        - y (np.ndarray): Sensor measurements.
            - Shape: `(n_sensors, n_times)`.
        """
        if self.orientation_type == "fixed":
            y = L @ x
        elif self.orientation_type == "free":
            y = np.einsum("nmr,mrd->nd", L, x)
        else:
            raise ValueError(f"Unsupported orientation type: {self.orientation_type}")
    
        return y

    def _add_noise(self, y_clean):
        cov = 1e-2 * np.eye(self.n_sensors)
        noise = self.rng.multivariate_normal(np.zeros(self.n_sensors), cov, size=self.n_times).T
        
        # Frobenius norm for signal and noise
        signal_norm = np.linalg.norm(y_clean, "fro")
        noise_norm = np.linalg.norm(noise, "fro")
        
        # Calculate SNR scaling factor
        snr_scaling_factor = ((1 - self.alpha_snr) / self.alpha_snr) * (signal_norm / noise_norm)
        
        noise_scaled = noise * snr_scaling_factor
        cov_scaled = cov * snr_scaling_factor ** 2
        y_noisy = y_clean + noise_scaled
        
        return y_noisy, cov_scaled, noise_scaled

    def simulate(self):
        L = self._get_leadfield()
        x = self._generate_source_time_courses()
        y_clean = self._project_to_sensor_space(L, x)
        y_noisy, cov_scaled, noise_scaled = self._add_noise(y_clean)

        # if self.orientation_type == "free":
        #     L = L.reshape(L.shape[0], -1)

        # if self.n_times == 1 and self.orientation_type == "fixed":
        #     y_noisy = y_noisy[:, 0]
        #     x = x[:, 0]

        visualize_signals(x, y_clean, y_noisy)
        visualize_leadfield(L, orientation_type=self.orientation_type, save_path="results/leadfield_matrix.png")
        
        
        info = LeadfieldSimulator(
            config=load_config(Path(self.leadfield_config_path)),
            logger=self.logger
        ).handle_info()
        
        visualize_leadfield(L, orientation_type=self.orientation_type, save_path="results/figures/leadfield_matrix.png")
        
        visualize_leadfield_topomap(
            leadfield_matrix=L,
            info=info,
            x=x,
            orientation_type=self.orientation_type,
            save_path="results/figures/leadfield_nnzs_topomap_.png",  # Path to save the combined figure
            title="Leadfield Topomap for active (Nonzero) Sources"  # Global title for the figure
        )
        
        return y_noisy, L, x, cov_scaled, noise_scaled
    
    

import matplotlib.pyplot as plt
def visualize_signals(
    x, 
    y_clean, 
    y_noisy, 
    active_sources=None, 
    nnz_to_plot=-1, 
    sfreq=100,  # Sampling frequency in Hz
    max_sensors=3, 
    plot_sensors_together=False, 
    shift=20, 
    figsize=(14, 10), 
    save_path='results/figures/data_sim.png'  # Path to save the figure
):
    """
    Visualize source activity and sensor measurements before and after adding noise.

    Parameters:
    - x (np.ndarray): Source activity (shape depends on orientation type).
    - y_clean (np.ndarray): Clean sensor measurements (n_sensors, n_times).
    - y_noisy (np.ndarray): Noisy sensor measurements (n_sensors, n_times).
    - active_sources (np.ndarray, optional): Indices of non-zero (active) sources.
    - nnz_to_plot (int): Number of non-zero sources to plot. If -1, plot all non-zero sources.
    - sfreq (float): Sampling frequency in Hz.
    - max_sensors (int): Maximum number of sensors to plot.
    - plot_sensors_together (bool): If True, plot all sensors on the same plot. If False, stack plots vertically.
    - shift (float): Vertical shift between sensors for better visualization.
    - figsize (tuple): Figure size for the plot.
    - save_path (str, optional): Path to save the figure. If None, the figure is not saved.
    """
    # Calculate the time vector based on sfreq and n_times
    n_times = y_clean.shape[1]  # Number of samples
    duration = n_times / sfreq  # Duration in seconds
    times = np.linspace(0, duration, n_times)  # Time vector

    # Extract active sources if not provided
    if active_sources is None:
        active_sources = np.where(np.any(x != 0, axis=-1))[0]  # Find indices of non-zero sources

    # Limit the number of non-zero sources to plot
    if nnz_to_plot != -1:
        active_sources = active_sources[:nnz_to_plot]

    # Calculate global y-axis limits for sensor signals
    y_min = min(y_clean.min(), y_noisy.min())
    y_max = max(y_clean.max(), y_noisy.max())

    # Create a figure with a dynamic number of subplots
    num_sensors = min(max_sensors, y_clean.shape[0])
    total_plots = 1 + (1 if plot_sensors_together else num_sensors)  # 1 for sources + sensors
    fig, axes = plt.subplots(
        total_plots, 
        1, 
        figsize=figsize, 
        gridspec_kw={"height_ratios": [1] + [1] * (total_plots - 1)}, 
        sharey=False  # Do not share y-axis for source signals
    )

    # Plot source activity (always on the first subplot)
    ax_sources = axes[0] if total_plots > 1 else axes
    for i in active_sources:  # Plot only the selected active sources
        ax_sources.plot(times, x[i].T, label=f"Source {i}")
    ax_sources.set_title("Non-Zero Simulated Source Activity (x)")
    ax_sources.set_ylabel("Amplitude (nAm)")  # Source signals in nanoamperes
    ax_sources.grid(True)

    # Plot sensor measurements
    if plot_sensors_together:
        # Plot all selected sensors on the same subplot
        ax_sensors = axes[1] if total_plots > 1 else axes
        for i in range(num_sensors):
            offset = i * shift  # Calculate vertical offset for each sensor
            ax_sensors.plot(times, y_clean[i] + offset, label=f"Clean Signal (Sensor {i})", linewidth=2)
            ax_sensors.plot(times, y_noisy[i] + offset, label=f"Noisy Signal (Sensor {i})", alpha=0.7)
        ax_sensors.set_title("Sensor Measurements (y)")
        ax_sensors.set_ylabel("Amplitude (µV)")  # Sensor signals in microvolts
        ax_sensors.set_ylim(y_min, y_max)  # Apply global y-axis limits
        ax_sensors.grid(True)
    else:
        # Plot each sensor on a separate subplot with shared y-axis
        for i in range(num_sensors):
            ax = axes[i + 1]  # Start from the second subplot
            ax.plot(times, y_clean[i], label=f"y_clean", linewidth=2)
            ax.plot(times, y_noisy[i], label=f"y_noisey", alpha=0.7)
            ax.set_title(f"Sensor {i}")
            ax.set_ylabel("Amplitude (µV)")  # Sensor signals in microvolts
            ax.set_ylim(y_min, y_max)  # Apply global y-axis limits
            ax.grid(True)

    # Add x-axis label only to the bottom-most plot
    if plot_sensors_together:
        ax_sensors.set_xlabel("Time (s)")
    else:
        axes[-1].set_xlabel("Time (s)")  # Set x-axis label for the last subplot

    # Add a shared legend to the right of the plots
    handles, labels = ax_sensors.get_legend_handles_labels() if plot_sensors_together else ax.get_legend_handles_labels()
    fig.legend(
        handles, 
        labels, 
        loc="center right", 
        bbox_to_anchor=(1.0, 0.5),  # Position the legend to the right of the plots
        borderaxespad=0, 
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for the legend

    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()
    
import matplotlib.pyplot as plt
import numpy as np

def visualize_leadfield(leadfield_matrix, orientation_type="fixed", save_path=None):
    """
    Visualize the leadfield matrix as a heatmap.

    Parameters:
    - leadfield_matrix (np.ndarray): The leadfield matrix.
        - For "fixed" orientation: Shape (n_sensors, n_sources).
        - For "free" orientation: Shape (n_sensors, n_sources, 3).
    - orientation_type (str): Orientation type ("fixed" or "free").
    - save_path (str, optional): Path to save the figure. If None, the figure is not saved.

    Returns:
    - None
    """
    if leadfield_matrix is None or not isinstance(leadfield_matrix, np.ndarray):
        raise ValueError("Invalid leadfield matrix. It must be a non-empty numpy array.")

    if orientation_type == "fixed":
        plt.figure(figsize=(10, 8))
        plt.imshow(leadfield_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
        plt.colorbar(label="Amplitude")
        plt.title("Leadfield Matrix (Fixed Orientation)")
        plt.xlabel("Sources")
        plt.ylabel("Sensors")
    elif orientation_type == "free":
        n_orient = leadfield_matrix.shape[-1]
        fig, axes = plt.subplots(1, n_orient, figsize=(15, 5), sharey=True)
        orientations = ["X", "Y", "Z"]
        for i in range(n_orient):
            ax = axes[i]
            ax.imshow(leadfield_matrix[:, :, i], aspect='auto', cmap='viridis', interpolation='nearest')
            ax.set_title(f"Leadfield Matrix (Orientation {orientations[i]})")
            ax.set_xlabel("Sources")
            if i == 0:
                ax.set_ylabel("Sensors")
        fig.colorbar(axes[0].images[0], ax=axes, location="right", label="Amplitude")
    else:
        raise ValueError("Invalid orientation type. Must be 'fixed' or 'free'.")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Leadfield matrix visualization saved to {save_path}")

    plt.show()
    
import os
import mne
import numpy as np
import matplotlib.pyplot as plt

def visualize_leadfield_topomap(leadfield_matrix, info, x, orientation_type="fixed", save_path=None, title=None):
    """
    Visualize the leadfield matrix as topological maps for nonzero sources in a single figure.

    Parameters:
    - leadfield_matrix (np.ndarray): The leadfield matrix.
        - For "fixed" orientation: Shape (n_sensors, n_sources).
        - For "free" orientation: Shape (n_sensors, n_sources, 3).
    - info (mne.Info): The MNE info object containing sensor locations.
    - x (np.ndarray): Source activity matrix to determine nonzero sources.
        - For "fixed" orientation: Shape (n_sources, n_times).
        - For "free" orientation: Shape (n_sources, 3, n_times).
    - orientation_type (str): Orientation type ("fixed" or "free").
    - save_path (str, optional): Path to save the figure. If None, the figure is not saved.
    - title (str, optional): Title for the entire figure.

    Returns:
    - None
    """
    if leadfield_matrix is None or not isinstance(leadfield_matrix, np.ndarray):
        raise ValueError("Invalid leadfield matrix. It must be a non-empty numpy array.")

    if x is None or not isinstance(x, np.ndarray):
        raise ValueError("Invalid source activity matrix. It must be a non-empty numpy array.")

    # Find nonzero sources
    if orientation_type == "fixed":
        nonzero_sources = np.where(np.any(x != 0, axis=-1))[0]
    elif orientation_type == "free":
        nonzero_sources = np.where(np.any(np.linalg.norm(x, axis=1) != 0, axis=-1))[0]
    else:
        raise ValueError("Invalid orientation type. Must be 'fixed' or 'free'.")

    # Create a figure with subplots for each nonzero source
    n_sources = len(nonzero_sources)
    n_cols = 5  # Number of columns in the figure
    n_rows = int(np.ceil(n_sources / n_cols))  # Calculate the number of rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), constrained_layout=True)

    # Flatten axes for easy indexing
    axes = axes.flatten()

    for i, source_idx in enumerate(nonzero_sources):
        if orientation_type == "fixed":
            # Extract the leadfield for the specified source
            leadfield_values = leadfield_matrix[:, source_idx]
        elif orientation_type == "free":
            # Extract the leadfield for the specified source and orientation
            leadfield_values = np.linalg.norm(leadfield_matrix[:, source_idx, :], axis=-1)
        else:
            raise ValueError("Invalid orientation type. Must be 'fixed' or 'free'.")

        # Plot the topomap for the current source
        mne.viz.plot_topomap(leadfield_values, info, axes=axes[i], cmap="viridis", show=False)
        axes[i].set_title(f"Source {source_idx}")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Add a global title if provided
    if title:
        fig.suptitle(title, fontsize=16, weight="bold")

    # Save the figure if a save path is provided
    if save_path:
        if not save_path.endswith(".png"):
            save_path += ".png"
        try:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"Combined leadfield topomap saved to {save_path}")
        except Exception as e:
            print(f"Failed to save the combined leadfield topomap: {e}")

    plt.show()