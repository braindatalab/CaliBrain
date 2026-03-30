## Workflow Cheat Sheet

- **Benchmarking** – Generates synthetic experiments, runs every inverse solver/hyperparameter combination, and records metrics plus posterior summaries. This stage is the “data factory” that tests how each solver behaves under the configured simulation grid.
- **Aggregation** – Collects the saved `posterior_summary*.h5` files, optionally filters them by metadata, and writes **one dataset per experiment run** (no pooling). Each NPZ contains the full set of sources from a single posterior summary so you can evaluate/calibrate every run independently.
- **Calibration** – Consumes the aggregated datasets to learn and evaluate uncertainty calibration mappings, reporting pre/post coverage gaps and optional reliability plots without re-running any simulations.

Edit the config files in this folder to choose solver grids, aggregation filters, and dataset paths, then run the three workflows in order to complete the full pipeline.
