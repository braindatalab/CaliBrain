from typing import Any, Dict, Union, Optional
import contextlib
import logging
from pathlib import Path
import os
import re
import runpy

import mne
import numpy as np
import yaml


def load_config(config_file: str, logger=None) -> dict:
    """
    Load the configuration from a YAML file.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file.
    logger : logging.Logger, optional
        Logger instance for logging messages.
      If None, a default logger will be created.

    Raises
    ------
    FileNotFoundError
        If the configuration file is not found.
    yaml.YAMLError
        If there is an error parsing the YAML file.
    ValueError
        If the configuration file is empty or invalid.

    Returns
    -------
    config : dict
        The loaded configuration as a dictionary.
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


def load_python_config(config_path: Path | str) -> Dict[str, Any]:
    """Execute a Python config file and return its CONFIG dict."""
    namespace = runpy.run_path(str(config_path))
    if "CONFIG" not in namespace:
        raise ValueError(f"Config {config_path} must define a CONFIG dict.")
    return namespace["CONFIG"]
    

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

    Parameters
    ----------
    obj
        The object to inspect.
    show_private : bool
        If True, include private attributes/methods (starting with '_').

    Returns
    -------
    dict
        Dictionary with 'attributes' and 'methods' keys.
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


# An absolute path: "/dir/file" or "C:\\dir\\file" (at least one directory level).
_LOCAL_PATH = re.compile(r"(?:[A-Za-z]:\\|/)(?:[^\s,;()<>\"'\\/]+[\\/])+[^\s,;()<>\"'\\/]*")


def strip_local_paths(info):
    """Remove local file paths from an MNE info (or forward info), in place.

    MNE stores the directory a file was made in (``working_dir``) and the paths of
    the files it was computed from (``mri_file``, ``meas_file``, ``command_line``).
    They contain user and machine names, so they must not end up in shared files.
    ``working_dir`` is emptied and the other fields keep only the file names.

    Parameters
    ----------
    info : mne.Info or dict
        The info to clean. It is modified in place.
    """
    unlock = info._unlock() if hasattr(info, "_unlock") else contextlib.nullcontext()
    with unlock:
        if "working_dir" in info:
            info["working_dir"] = ""
        for key in ("mri_file", "meas_file", "command_line"):
            value = info.get(key)
            if isinstance(value, str) and value:
                info[key] = _LOCAL_PATH.sub(
                    lambda m: m.group(0).replace("\\", "/").rstrip("/").split("/")[-1], value
                )
    return info


def get_data_path(path: Optional[Union[str, Path]] = None) -> Path:
    """Get the path to CaliBrain data directory.
    
    Parameters
    ----------
    path : str | Path | None
        Custom path to data directory. If None, the ``CALIBRAIN_DATA`` environment
        variable is used if it is set. Otherwise the default is ``<repository>/data``
        in a source checkout, and a per-user cache directory for an installed
        package (for example ``~/.cache/calibrain`` on Linux).
        
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
            calibrain_root = Path(__file__).parent.parent
            if (calibrain_root / "pyproject.toml").exists():
                # A source checkout (also an editable install): <repository>/data
                data_path = calibrain_root / 'data'
            else:
                # An installed package: a user directory, not the Python environment
                import pooch

                data_path = Path(pooch.os_cache("calibrain"))
        data_path = Path(data_path)
    else:
        data_path = Path(path)
    
    # Create directory if it doesn't exist
    data_path.mkdir(parents=True, exist_ok=True)
    
    return data_path


def restrict_fwd_to_sources(
    fwd,
    n_keep=1284,
    seed=None,
):
    """
    Randomly reduce a forward solution to a subset of sources on both hemispheres.

    Parameters
    ----------
    fwd : mne.Forward
        Forward solution to restrict.
    n_keep : int
        Number of sources to keep per hemisphere.
    seed : int | None
        Random seed for selecting vertices.
    """

    if not isinstance(n_keep, int):
        raise TypeError("n_keep must be an integer.")
    if n_keep <= 0:
        raise ValueError("n_keep must be positive.")

    vertices = [np.array([], dtype=int), np.array([], dtype=int)]
    rng = np.random.default_rng(seed)
    data_segments = []

    for hemi_idx in (0, 1):
        hemi_vertices = fwd["src"][hemi_idx]["vertno"]
        if n_keep > len(hemi_vertices):
            hemi = "lh" if hemi_idx == 0 else "rh"
            raise ValueError(f"Only {len(hemi_vertices)} sources in {hemi}")

        sel = np.sort(rng.choice(len(hemi_vertices), size=n_keep, replace=False))
        vertices[hemi_idx] = hemi_vertices[sel]
        data_segments.append(np.ones((n_keep, 1)))

    data = np.vstack(data_segments)
    stc = mne.SourceEstimate(data, vertices=vertices, tmin=0.0, tstep=1.0)

    return mne.forward.restrict_forward_to_stc(fwd, stc, on_missing="raise")
