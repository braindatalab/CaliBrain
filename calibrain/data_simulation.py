
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
        noise_type="random",
        cov_type="diag",
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
        - cov_type (str): Type of covariance ('diag' or 'full').
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
        self.cov_type = cov_type
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
                return leadfield
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
            
            # validate the shape of the leadfield matrix
            if leadfield.ndim != expected_dimensions:
                raise ValueError(
                    f"Leadfield matrix shape mismatch: expected {expected_dimensions} dimensions, "
                    f"but got {leadfield.ndim} dimensions. Check the leadfield config file if the orientation type aligns with the intended simulation."
                )
            
            return leadfield
    
        elif self.leadfield_mode == "random":
            self.logger.info("Generating a random leadfield matrix.")
            if self.orientation_type == "fixed":
                leadfield = self.rng.randn(self.n_sensors, self.n_sources)
            else:  # Free orientation
                leadfield = self.rng.randn(self.n_sensors, self.n_sources, 3)
            self.logger.info(f"Random leadfield generated with shape {leadfield.shape}")
            return leadfield
    
        else:
            raise ValueError(f"Invalid leadfield mode '{self.leadfield_mode}'. Options are 'load', 'simulate', or 'random'.")

    def _generate_sources(self, L):
        """
        Generate synthetic source activity and corresponding sensor measurements.
    
        This function generates synthetic source activity (`x`) and computes the corresponding sensor measurements (`y`) using the provided leadfield matrix (`L`). The source activity can be generated for either "fixed" or "free" orientation types, and the number of active sources (`nnz`) is determined randomly.
    
        Parameters:
        - L (np.ndarray): Leadfield matrix.
            - For "fixed" orientation: Shape `(n_sensors, n_sources)`.
            - For "free" orientation: Shape `(n_sensors, n_sources, 3)`, where the last dimension corresponds to the three orientations (X, Y, Z).
    
        Returns:
        - x (np.ndarray): Source activity.
            - For "fixed" orientation: Shape `(n_sources, n_times)`.
            - For "free" orientation: Shape `(n_sources, 3, n_times)`, where the second dimension corresponds to the three orientations (X, Y, Z).
    
        - y (np.ndarray): Sensor measurements.
            - Shape: `(n_sensors, n_times)`.
    
        Notes:
        - The number of active sources (`nnz`) is randomly selected using `self.rng.choice`.
        - If `n_times = 1`, the output shapes of `x` and `y` remain consistent with the multi-time-point 
          case, ensuring compatibility with downstream processing.
    
        Example:
        - For `orientation_type = "fixed"`, `n_sources = 100`, `n_times = 10`, and `nnz = 5`:
            - `x` will have shape `(100, 10)`.
            - `y` will have shape `(n_sensors, 10)`.
    
        - For `orientation_type = "free"`, `n_sources = 100`, `n_times = 10`, and `nnz = 5`:
            - `x` will have shape `(100, 3, 10)`.
            - `y` will have shape `(n_sensors, 10)`.
    
        Raises:
        - ValueError: If the leadfield matrix (`L`) has an unexpected shape for the specified 
          orientation type.
        """
        if self.orientation_type == "fixed":
            self.n_sensors, self.n_sources = L.shape
        elif self.orientation_type == "free":
            self.n_sensors, self.n_sources, _ = L.shape
    
        # Select random indices for active sources
        idx = self.rng.choice(self.n_sources, size=self.nnz, replace=False)
    
        if self.orientation_type == "fixed":
            x = np.zeros((self.n_sources, self.n_times))
            x[idx] = self.rng.randn(self.nnz, self.n_times)
            y = L @ x
        else:
            n_orient = 3
            x = np.zeros((self.n_sources, n_orient, self.n_times))
            x[idx] = self.rng.randn(self.nnz, n_orient, self.n_times)
            y = np.einsum("nmr,mrd->nd", L, x)
    
        return x, y

    def _add_noise(self, y):
        if self.cov_type == "diag":
            if self.noise_type == "random":
                rv = wishart(df=self.n_sensors, scale=1e-3 * np.eye(self.n_sensors))
                cov = np.diag(np.diag(rv.rvs()))
            else:  # scaled_identity
                cov = 1e-2 * np.eye(self.n_sensors)
        else:
            rv = wishart(df=self.n_sensors, scale=1e-3 * np.eye(self.n_sensors))
            cov = rv.rvs()

        signal_norm = np.linalg.norm(y, "fro")
        noise = self.rng.multivariate_normal(np.zeros(self.n_sensors), cov, size=self.n_times).T
        noise_norm = np.linalg.norm(noise, "fro")
        noise_normalised = noise / noise_norm
        noise_scaled = ((1 - self.alpha_snr) / self.alpha_snr) * signal_norm * noise_normalised
        cov_scaled = cov * (((1 - self.alpha_snr) / self.alpha_snr) * (signal_norm / noise_norm)) ** 2
        y_noisy = y + noise_scaled
        return y_noisy, cov_scaled, noise_scaled

    def simulate(self):
        L = self._get_leadfield()
        x, y = self._generate_sources(L)
        y_noisy, cov_scaled, noise_scaled = self._add_noise(y)

        if self.orientation_type == "free":
            L = L.reshape(L.shape[0], -1)

        # if self.n_times == 1 and self.orientation_type == "fixed":
        #     y_noisy = y_noisy[:, 0]
        #     x = x[:, 0]

        return y_noisy, L, x, cov_scaled, noise_scaled

