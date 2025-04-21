import datetime
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.utils import check_random_state
from itertools import product
import mne
from itertools import combinations
from scipy.stats import wishart
from matplotlib.patches import Ellipse
from scipy.stats import chi2

from calibrain import SourceEstimator, DataSimulator, UncertaintyEstimator, gamma_map, eloreta

class Benchmark:
    def __init__(self, solver, solver_param_grid, data_param_grid, data_simulator, metrics, logger=None):
        """
        Initialize the Benchmark class.

        Parameters:
        - solver (callable): The solver function (e.g., gamma_map, eloreta).
        - solver_param_grid (dict): Grid of solver hyperparameters.
        - data_param_grid (dict): Grid of data generation hyperparameters.
        - metrics (list of callables): List of metric functions to evaluate the results.
        - logger (logging.Logger, optional): Logger instance for logging messages.
        """
        self.solver = solver
        self.solver_param_grid = solver_param_grid
        self.data_param_grid = data_param_grid
        self.data_simulator = data_simulator
        self.metrics = metrics
        self.logger = logger if logger else logging.getLogger(__name__)
    
    def create_experiment_directory(self, base_dir, params):
        """
        Create a directory structure for the experiment, with subdirectories for each parameter.
    
        Parameters:
        - base_dir (str): Base directory for the experiment.
        - params (dict): Dictionary of parameters.
    
        Returns:
        - experiment_dir (str): Path to the experiment directory.
        """
        # Sanitize params keys and values to ensure valid directory names
        sanitized_params = {
            k: str(v).replace("/", "_").replace("\\", "_").replace(" ", "_") for k, v in params.items()
        }
    
        # Create subdirectories for each parameter
        experiment_dir = base_dir
        for param, value in sanitized_params.items():
            experiment_dir = os.path.join(experiment_dir, f"{param}={value}")
    
        try:
            # Create the directory structure if it doesn't exist
            os.makedirs(experiment_dir, exist_ok=True)
        except OSError as e:
            self.logger.error(f"Failed to create experiment directory: {experiment_dir}. Error: {e}")
            raise
    
        self.logger.info(f"Experiment directory created: {experiment_dir}")
        return experiment_dir

    def run(self, nruns=1):
        """
        Run benchmarking by iterating over combinations of solver and data parameters.

        Returns:
        - results (pd.DataFrame): DataFrame containing the results for each parameter combination.
        """
        rng = check_random_state(42) 
        seeds = rng.randint(low=0, high=2 ** 32, size=nruns)

        results = []
        
        self.logger.info(f"Starting benchmark with {nruns} runs...")
        # Iterate over all combinations of solver and data parameters
        for solver_params, data_params, seed, in product(
            ParameterGrid(self.solver_param_grid),
            ParameterGrid(self.data_param_grid),
            seeds
        ):
            solver_name = getattr(self.solver, "__name__", str(self.solver))

            self.logger.info(f"Testing solver: {solver_name}")
            self.logger.info(f"Solver parameters: {solver_params}")
            self.logger.info(f"Data parameters: {data_params}")

            # Update data_simulator with data parameters
            self.data_simulator.seed = seed
            self.data_simulator.rng = rng
            self.data_simulator.__dict__.update(data_params)

            # Simulate data
            self.logger.info("Simulating data...")
            y_noisy, L, x, cov_scaled, noise_scaled = self.data_simulator.simulate()

            # Initialize SourceEstimator
            source_estimator = SourceEstimator(solver=self.solver, solver_params=solver_params, logger=self.logger)

            # Fit the estimator
            self.logger.info("Fitting the solver...")
            source_estimator.fit(L, y_noisy)

            # Estimate sources
            self.logger.info("Estimating sources...")
            x_hat, active_set, posterior_cov = source_estimator.predict(y_noisy)
                                    
            # Create experiment directory
            self.logger.info("Creating experiment directory...")
            
            experiment_dir = self.create_experiment_directory(
                base_dir="results/uncertainty_analysis",
                params={
                    "estimator": solver_name,
                    **solver_params, 
                    **data_params}
            )
            
            self.logger.info("Initializing uncertainty estimator...")
            uncertainty_estimator = UncertaintyEstimator(
                orientation_type=self.data_simulator.orientation_type,
                x=x,
                x_hat=x_hat,
                active_set=active_set,
                posterior_cov=posterior_cov,
                experiment_dir=experiment_dir,
                logger=self.logger,
            )

            full_posterior_cov = uncertainty_estimator.construct_full_covariance()
            
            uncertainty_estimator.plot_sorted_posterior_variances(top_k=10)
            uncertainty_estimator.visualize_sorted_covariances(top_k=10)
            
            if self.data_param_grid.get("orientation_type") == "free":
                uncertainty_estimator.plot_top_relevant_CE_pairs(top_k=5, confidence_level=0.95)
            
            uncertainty_estimator.plot_active_sources_single_time_step(time_step=0)
            uncertainty_estimator.plot_posterior_covariance_matrix()
            
            uncertainty_estimator.visualize_confidence_intervals()


            # Evaluate metrics
            self.logger.info("Evaluating metrics...")
            run_results = {
                "seed": seed,
                "solver": solver_name,
                "solver_params": solver_params,
                "data_params": data_params,
                "active_set_size": len(active_set),
            }
            
            try:
                for metric in self.metrics:
                    if metric is not None:
                        run_results[metric.__name__] = metric(x_hat, x)
            except Exception as e:
                self.logger.error(f"Error evaluating metrics: {e}")
                run_results["error"] = str(e)
                
            results.append(run_results)

        self.logger.info("Benchmarking completed.")
        # Convert results to a DataFrame
        return pd.DataFrame(results)


    
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    
    # sfreq = 150  # Sampling frequency in Hz
    # duration = 2 # 0.5  # Duration in seconds
    # tstep = 1.0 / sfreq  # Time step between samples
    # times = np.arange(0, duration, tstep)
    # n_times = len(times) # = int(sfreq * duration)  # Total number of time points

    data_param_grid = {
        "n_times": [100],
        "nnz": [5],
        "orientation_type": ["fixed"],
        "alpha_snr": [0.5],
    }
    gamma_map_params = {
        "gammas": [0.001], #  0.001, 1.0, or tuple for random values (0.001, 0.1)   
        "noise_type": ["oracle"], # "baseline", "oracle", "joint_learning", "CV"
    }

    estimators = [
        (gamma_map, gamma_map_params, data_param_grid),
        # (eloreta, {}, {}), 
    ]

    metrics = [None]
    nruns = 1
    data_simulator = DataSimulator(
        leadfield_mode='load',
        leadfield_path='results/fsaverage-leadfield-fixed.npz',
        n_sensors=None,
        n_sources=None,
        leadfield_config_path='configs/leadfield_sim_cfg.yml',
    )
    
    df = []
    
    for solver, solver_param_grid, data_param_grid in estimators:
        benchmark = Benchmark(
            solver=solver,
            solver_param_grid=solver_param_grid,
            data_param_grid=data_param_grid,
            data_simulator=data_simulator,
            metrics=metrics,
            logger=logger
        )
        results_df = benchmark.run(nruns=nruns)
        df.append(results_df)

    results_df = pd.concat(df)

    os.makedirs("results/benchmark_results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f"results/benchmark_results/benchmark_results_{timestamp}.csv", index=False)
    
    print(results_df.head())

if __name__ == "__main__":
    main()
    
    
    # def validate_source_space_dimensions(self):
    #     """
    #     Validate that the number of sources matches the source space.

    #     Raises:
    #     - ValueError: If the number of sources does not match the source space.
    #     """
    #     n_sources = sum(len(v) for v in self.vertices)
    #     if self.orientation_type == "fixed" and self.x.shape[0] != n_sources:
    #         raise ValueError(f"Data has {self.x.shape[0]} sources, but source space has {n_sources} sources!")
    #     elif self.orientation_type == "free" and self.x.shape[0] != n_sources * 3:
    #         raise ValueError(f"Data has {self.x.shape[0]} sources, but source space has {n_sources * 3} sources!")