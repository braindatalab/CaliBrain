import os
import datetime
import logging
import numpy as np
import pandas as pd
from calibrain import DataSimulator, gamma_map, eloreta
from calibrain import Benchmark


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
        "n_times": [1, 10],
        "nnz": [5, 50],
        "orientation_type": ["fixed"],
        "alpha_snr": [0.1, 0.5, 0.9],
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
        leadfield_mode='simulate', # simulate, random, load
        leadfield_path=f'results/forward/fsaverage-leadfield-{data_param_grid['orientation_type']}.npz',
        # n_sensors=None,
        # n_sources=None,
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