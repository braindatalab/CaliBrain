import os
import datetime
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from calibrain import DataSimulator, gamma_map, eloreta
from calibrain import Benchmark

# print(fwd['info']['chs'][0]['unit'])  # Will show 107 (FIFF_UNIT_V)
def main():
    os.makedirs("results/benchmark_results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"results/benchmark_results/benchmark_log_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode="w"),  # Overwrite log file each run
            logging.StreamHandler()                   # Also print to console
        ]
    )
    logger = logging.getLogger(__name__)
     
    # run this only once, to save the subjects' info if not already saved
    # save_subjects_mne_info(
    #     subjects=["CC120166", "CC120264", "CC120309", "CC120313"],
    #     fwd_dir='BSI-ZOO_forward_data'
    # )
    
    data_simulator_eeg = DataSimulator(
        tmin=-0.5,
        tmax=0.5,
        stim_onset=0.0,
        sfreq=250,
        fmin=1,
        fmax=5,
        amplitude=5,
        n_trials=4, # we will slice this to use only the first trial. TODO: Keep it like this for now
        leadfield_mode='load', # (simulate, random, load). NOTE: if `simulate` then align the config file of the leadfield simulation!
        channel_type='eeg',
        leadfield_dir=Path(f'BSI-ZOO_forward_data'),
        leadfield_config_path='configs/leadfield_sim_cfg.yml',
        mne_info_path='configs/mne_info.yml',
        logger=logger
    )
    
    data_simulator_meg = DataSimulator(
        tmin=-0.5,
        tmax=0.5,
        stim_onset=0.0,
        sfreq=250,
        fmin=1,
        fmax=5,
        amplitude=5,
        n_trials=4, # we will slice this to use only the first trial. TODO: Keep it like this for now
        leadfield_mode='load', # (simulate, random, load). NOTE: if `simulate` then align the config file of the leadfield simulation!
        channel_type='meg',
        leadfield_dir=Path(f'BSI-ZOO_forward_data'),
        leadfield_config_path='configs/leadfield_sim_cfg.yml',
        mne_info_path='configs/mne_info.yml',
        logger=logger
    )
        
    data_param_grid_meg = {
        "subject": ["CC120166", "fsaverage"], # "CC120166", "CC120264", "CC120309", "CC120313", "caliBrain_fsaverage", # "fsaverage", 
        "nnz": [3, 5],
        "orientation_type": ["fixed"], # "fixed", "free"
        "alpha_snr_db": [-40, 40],
    }
    
    data_param_grid_eeg = {
        "subject": ["caliBrain_fsaverage"], # "CC120166", "CC120264", "CC120309", "CC120313", "caliBrain_fsaverage", # "fsaverage", 
        "nnz": [3, 5],
        "orientation_type": ["fixed"], # "fixed", "free"
        "alpha_snr_db": [-40, 40],
    }
        
    gamma_map_params = {
        "gammas": [0.001], #  0.001, 1.0, or tuple for random values (0.001, 0.1)   
        "noise_type": ["oracle"], # "baseline", "oracle", "joint_learning", "CV"
    }
    
    estimators = [
        (gamma_map, gamma_map_params, data_param_grid_meg, data_simulator_meg),
        (gamma_map, gamma_map_params, data_param_grid_eeg, data_simulator_eeg),
        # (gamma_map, gamma_map_params, data_param_grid_2, data_simulator2),
        # (eloreta, {}, {}, data_simulator), 
    ]

    metrics = [
    "calibration_curve_metrics",
    "mean_posterior_std",
    # "f1",
    # "emd",
    # "accuracy",  
    ]
    
    nruns = 2
    df = []
    for solver, solver_param_grid, data_param_grid, data_simulator in estimators:
        benchmark = Benchmark(
            solver=solver,
            solver_param_grid=solver_param_grid,
            data_param_grid=data_param_grid,
            data_simulator=data_simulator,
            metrics=metrics,
            random_state=42,
            logger=logger
        )
        results_df = benchmark.run(nruns=nruns)
        df.append(results_df)

    results_df = pd.concat(df)
    results_df.to_csv(f"results/benchmark_results/benchmark_results_{timestamp}.csv", index=False)
    
    print(results_df.head())

if __name__ == "__main__":
    main()