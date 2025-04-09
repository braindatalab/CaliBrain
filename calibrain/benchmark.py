
import datetime
import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.utils import check_random_state
from itertools import product

from calibrain.inverse_estimation import SourceEstimator
from calibrain.data_simulation import DataSimulator
from calibrain.inverse_estimation import gamma_map, eloreta

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
    
    data_param_grid = {
        "n_times": [50],
        "nnz": [5],
        "orientation_type": ["fixed"],
        "alpha_snr": [0.5],
        "noise_type": ["scaled"], # random, scaled
        "cov_type": ["scaled_identity"], # diag, scaled_identity
    }
    gamma_map_params = {
        "gammas": [0.001], #  0.001, 1.0, or tuple for random values (0.001, 0.1)
        "sigma_squared": ["oracle"], # CV, joint_learning, oracle
        # "cov": [1.0],
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
    results_df.to_csv(f"benchmark_results_{timestamp}.csv", index=False)

    # Print the first few rows of results
    print(results_df.head())

if __name__ == "__main__":
    main()