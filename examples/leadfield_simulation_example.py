import argparse
from mne.datasets import sample
from calibrain.utils import load_config
import logging
from pathlib import Path
from calibrain import LeadfieldBuilder


def main(args=None):
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the Leadfield simulation pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file of leadfield simulation."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level. Overrides the value in the configuration file."
    )

    # Parse arguments (use provided args if given, otherwise use sys.argv)
    args = parser.parse_args(args)
    config = load_config(Path(args.config))

    # Determine log level
    valid_log_levels = ["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    log_level = args.log_level or config.get("log_level", "INFO").upper()
    if log_level not in valid_log_levels:
        logger.warning(f"Invalid log level '{log_level}' in configuration. Defaulting to 'INFO'.")
        log_level = "INFO"

    # Configure logging
    logging.basicConfig(level=getattr(logging, log_level), format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.info(f"Using configuration file: {Path(args.config)}")

    # Initialize simulation
    leadfield_sim = LeadfieldBuilder(config=config, logger=logger)

    # Run the pipeline
    try:
        leadfield = leadfield_sim.simulate()
        logger.info(f"Leadfield shape: {leadfield.shape}")
    except Exception as e:
        logger.error(f"An error occurred during the pipeline execution: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    # import sys
    # main(sys.argv[1:])
    main([
        "--config", "configs/leadfield_sim_cfg.yml",
        "--log-level", "INFO"
    ])