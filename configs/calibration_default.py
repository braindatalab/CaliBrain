# If test_dataset is None, the train file is reused for evaluation.

config_name = "gamma_map_sflex_oracle"

CONFIG = {
    "train_dataset": f"results/calibration_datasets/{config_name}/posterior_dataset_train.npz",
    "test_dataset": f"results/calibration_datasets/{config_name}/posterior_dataset_test.npz",
    "output_dir": "results/calibration_eval",
    "run_name": config_name,
    "plot_curve": True,
    "nominal_coverages": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999],
}
