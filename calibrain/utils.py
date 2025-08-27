from typing import Union, Optional
import yaml
import logging
from pathlib import Path
import mne
import os


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
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {config_file}")
    
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
    

def save_subjects_mne_info(subjects=["CC120166", "CC120264", "CC120309", "CC120313"], fwd_dir='examples/BSI-ZOO_forward_data'):

    for subject in subjects:
        fwd_file = f'{fwd_dir}/{subject}-fwd.fif'
        print(f'Processing forward solution for subject: {subject}')
        
        # Load the forward solution
        fwd = mne.read_forward_solution(fwd_file)

        # Extract the info object
        info = fwd['info']

        # Add artificial empty events to avoid KeyError
        if "events" not in info:
            info["events"] = []

        # Save the info object to a file
        info_file = f'{fwd_dir}/{subject}-info.fif'
        mne.io.write_info(info_file, info)        
        

def inspect_object(obj, show_private=False):
    """
    Print attributes and methods of a Python object separately.

    Parameters:
    - obj: The object to inspect.
    - show_private (bool): If True, include private attributes/methods (starting with '_').

    Returns:
    - dict with 'attributes' and 'methods' keys
    """
    def is_valid(name):
        return show_private or not name.startswith("_")

    attributes = [a for a in dir(obj)
                  if not callable(getattr(obj, a)) and is_valid(a)]
    
    methods = [m for m in dir(obj)
               if callable(getattr(obj, m)) and is_valid(m)]
    
    print("Attributes:")
    for attr in attributes:
        print(f"  - {attr}")
    
    print("\nMethods:")
    for method in methods:
        print(f"  - {method}")
    
    return {"attributes": attributes, "methods": methods}


def get_data_path(path: Optional[Union[str, Path]] = None) -> Path:
    """Get the path to CaliBrain data directory.
    
    Parameters
    ----------
    path : str | Path | None
        Custom path to data directory. If None, uses default location.
        
    Returns
    -------
    data_path : Path
        Path to CaliBrain data directory.
    """
    if path is None:
        # Use environment variable or default location in calibrain package
        if 'CALIBRAIN_DATA' in os.environ:
            data_path = Path(os.environ['CALIBRAIN_DATA'])
        else:
            # Default to calibrain/data directory relative to this module
            calibrain_root = Path(__file__).parent  # calibrain module directory
            data_path = calibrain_root / 'data'
        data_path = Path(data_path)
    else:
        data_path = Path(path)
    
    # Create directory if it doesn't exist
    data_path.mkdir(parents=True, exist_ok=True)
    
    return data_path
