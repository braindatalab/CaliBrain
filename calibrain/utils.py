import yaml
import logging


def load_config(config_file: str, logger=None) -> dict:
    """
    Load the configuration from a YAML file.

    Parameters:
    - config_file (str): Path to the YAML configuration file.
    - logger (logging.Logger, optional): Logger instance for logging messages.
      If None, a default logger will be created.

    Raises:
    - FileNotFoundError: If the configuration file is not found.
    - yaml.YAMLError: If there is an error parsing the YAML file.
    - ValueError: If the configuration file is empty or invalid.

    Returns:
    - config (dict): The loaded configuration as a dictionary.
    """
    logger = logger if logger else logging.getLogger(__name__)
    logger.info(f"Loading configuration from file: {config_file}")

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise TypeError("The configuration file must contain a dictionary at the top level.")
        if not config:
            raise ValueError("The configuration file is empty or invalid.")
        logger.info("Configuration successfully loaded.")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_file}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise