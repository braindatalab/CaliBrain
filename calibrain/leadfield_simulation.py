"""
Module for simulating leadfield matrices using MNE-Python.

Provides a configurable framework for setting up and running a leadfield
simulation pipeline. Handles various MNE-Python components, such as source
space, BEM model, montage, info object, forward solution, and leadfield matrix,
allowing loading from files or creating them based on configuration.
"""

import logging
import numpy as np
from pathlib import Path
from mne.io.constants import FIFF
import mne
from typing import Dict, Optional, Union

class LeadfieldSimulator:
    """
    Simulates leadfield matrices based on a configuration dictionary.

    Handles the creation or loading of necessary MNE-Python components like
    source space, BEM model, montage, info, and forward solution.

    Attributes
    ----------
    config : dict
        The configuration dictionary driving the simulation process.
    logger : logging.Logger
        Logger instance for logging messages.
    data_path : Path
        Path to the data directory.
    subject : str
        Subject identifier.
    subjects_dir : Path
        Path to the FreeSurfer subjects directory.
    save_path : Path
        Base path where generated files (source space, BEM, etc.) will be saved.
    """
    def __init__(self, config: Dict, logger: Optional[logging.Logger] = None):
        """
        Initialize the LeadfieldSimulator.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing paths and parameters for
            each simulation step (data, source_space, bem_model, montage,
            info, forward_solution, leadfield).
        logger : logging.Logger, optional
            Custom logger instance. If None, a default logger is created,
            by default None.

        Raises
        ------
        FileNotFoundError
            If `data_path` or `subjects_dir` specified in the config do not exist.
        """
        self.logger = logger if logger else logging.getLogger(__name__)
        self.config = config

        data_cfg = config["data"]
        self.data_path = Path(data_cfg["data_path"])
        self.subject = data_cfg["subject"]
        self.subjects_dir = Path(data_cfg["subjects_dir"])
        self.save_path = Path(data_cfg["save_path"])
        
                    
        # NOTE: The channel type is hardcoded to "eeg" for now.
        # TODO: Make this configurable in the future. with something like:
        # channel_type = "eeg" if self.config["data"].get("channel_type") == "eeg" else "meg" and use it in the montage and info creation
        self.channel_type = "eeg" # Hardcoded for now, can be made configurable later

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")
        if not self.subjects_dir.exists():
            raise FileNotFoundError(f"Subjects directory does not exist: {self.subjects_dir}")
        if not self.save_path.exists():
            self.logger.info(f"Save path does not exist. Creating: {self.save_path}")
            self.save_path.mkdir(parents=True, exist_ok=True)

        self.logger.info("LeadfieldSimulator initialized successfully.")

    def handle_source_space(self) -> mne.SourceSpaces:
        """
        Load or create the source space.

        Reads the source space from the file specified in `config['source_space']['fname']`
        if it exists. Otherwise, creates a new source space using
        `mne.setup_source_space` with parameters from the configuration.
        Saves the created source space if `config['source_space']['save']` is True.

        Returns
        -------
        mne.SourceSpaces
            The loaded or created source space object.
        """
        source_space_cfg = self.config["source_space"]
        fname = source_space_cfg.get("fname")
        if fname:
            load_path = Path(fname)
            if load_path.exists():
                self.logger.info(f"Loading source space from file: {load_path}")
                return mne.read_source_spaces(load_path)
            else:
                self.logger.warning(f"Source space file does not exist: {load_path}. Creating from scratch...")
        else:
            self.logger.info("No source space file specified. Creating from scratch...")

        valid_kwargs = ["spacing", "surface", "add_dist", "n_jobs", "verbose"]
        kwargs = {key: value for key, value in source_space_cfg.items() if key in valid_kwargs}
        src = mne.setup_source_space(subject=self.subject, subjects_dir=self.subjects_dir, **kwargs)

        if source_space_cfg.get("save", False):
            save_fname = self.save_path / f"{self.subject}-src.fif"
            verbose = source_space_cfg.get("verbose") # Let MNE handle default if None
            overwrite = source_space_cfg.get("overwrite", False) # Default to False
            self.logger.info(f"Saving source space to file: {save_fname}")
            mne.write_source_spaces(save_fname, src, overwrite=overwrite, verbose=verbose)

        return src

    def handle_bem_model(self) -> mne.bem.ConductorModel:
        """
        Load or create the BEM (Boundary Element Model) solution.

        Reads the BEM solution from the file specified in `config['bem_model']['fname']`
        if it exists. Otherwise, creates a new BEM model using `mne.make_bem_model`
        and computes the solution using `mne.make_bem_solution` with parameters
        from the configuration. Saves the created BEM solution if
        `config['bem_model']['save']` is True.

        Returns
        -------
        mne.bem.ConductorModel
            The loaded or created BEM solution object.
        """
        bem_model_cfg = self.config["bem_model"]
        fname = bem_model_cfg.get("fname")
        if fname:
            load_path = Path(fname)
            if load_path.exists():
                self.logger.info(f"Loading BEM model from file: {load_path}")
                return mne.read_bem_solution(load_path)
            else:
                self.logger.warning(f"BEM model file does not exist: {load_path}. Creating from scratch...")
        else:
            self.logger.info("No BEM model file specified. Creating from scratch...")

        valid_model_kwargs = ["ico", "conductivity", "verbose"]
        model_kwargs = {key: value for key, value in bem_model_cfg.items() if key in valid_model_kwargs}
        bem_model = mne.make_bem_model(subject=self.subject, subjects_dir=self.subjects_dir, **model_kwargs)

        valid_sol_kwargs = ["solver", "verbose"]
        sol_kwargs = {key: value for key, value in bem_model_cfg.items() if key in valid_sol_kwargs}
        bem = mne.make_bem_solution(surfs=bem_model, **sol_kwargs)

        if bem_model_cfg.get("save", False):
            save_fname = self.save_path / f"{self.subject}-bem.fif"
            verbose = bem_model_cfg.get("verbose")
            overwrite = bem_model_cfg.get("overwrite", False)
            self.logger.info(f"Saving BEM model to file: {save_fname}")
            mne.write_bem_solution(save_fname, bem, overwrite=overwrite, verbose=verbose)

        return bem

    def handle_montage(self) -> mne.channels.DigMontage:
        """
        Load or create the sensor montage.

        Reads the montage from the file specified in `config['montage']['fname']`
        if it exists. Otherwise, creates a standard montage using
        `mne.channels.make_standard_montage` with parameters from the configuration.
        Saves the created montage if `config['montage']['save']` is True.

        Returns
        -------
        mne.channels.DigMontage
            The loaded or created montage object.
        """
        montage_cfg = self.config.get("montage", {})
        fname = montage_cfg.get("fname")
        if fname:
            load_path = Path(fname)
            if load_path.exists():
                self.logger.info(f"Loading montage from file: {load_path}")
                # Assuming FIF format based on save method
                return mne.channels.read_dig_fif(load_path)
            else:
                self.logger.warning(f"Montage file does not exist: {load_path}. Creating from scratch...")
        else:
            self.logger.info("No montage file specified. Creating from scratch...")

        valid_kwargs = ["kind", "head_size"] # Parameters for make_standard_montage
        kwargs = {key: value for key, value in montage_cfg.items() if key in valid_kwargs}
        if not kwargs.get("kind"):
             self.logger.warning("Montage 'kind' not specified in config, cannot create standard montage. Returning None.")
             return None # Cannot create without kind
        montage = mne.channels.make_standard_montage(**kwargs)

        if montage_cfg.get("save", False):
            save_fname = self.save_path / f"{self.subject}-montage.fif"
            verbose = montage_cfg.get("verbose")
            overwrite = montage_cfg.get("overwrite", False)
            self.logger.info(f"Saving montage to file: {save_fname}")
            montage.save(save_fname, overwrite=overwrite, verbose=verbose)

        return montage

    def handle_info(self) -> mne.Info:
        """
        Load or create the MNE measurement info object.

        Reads the info from the file specified in `config['info']['fname']`
        if it exists. Otherwise, creates a new info object using `mne.create_info`
        with parameters from the configuration, associating it with the montage
        obtained via `handle_montage`. Saves the created info object if
        `config['info']['save']` is True.

        Returns
        -------
        mne.Info
            The loaded or created info object.

        Raises
        ------
        ValueError
            If info cannot be created because the montage is missing or invalid.
        """
        info_cfg = self.config["info"]
        fname = info_cfg.get("fname")
        if fname:
            load_path = Path(fname)
            if load_path.exists():
                self.logger.info(f"Loading info object from file: {load_path}")
                return mne.io.read_info(load_path)
            else:
                self.logger.warning(f"Info file does not exist: {load_path}. Creating from scratch...")
        else:
            self.logger.info("No info file specified. Creating from scratch...")

        montage = self.handle_montage()
        if montage is None or not hasattr(montage, 'ch_names'):
             raise ValueError("Cannot create info object without a valid montage.")

        valid_kwargs = ["sfreq", "ch_types", "verbose"]
        kwargs = {key: value for key, value in info_cfg.items() if key in valid_kwargs}
        if not kwargs.get("sfreq") or not kwargs.get("ch_types"):
             raise ValueError("Cannot create info object without 'sfreq' and 'ch_types' in config.")

        info = mne.create_info(
            ch_names=montage.ch_names,
            **kwargs
        )
        info.set_montage(montage)

        if info_cfg.get("save", False):
            save_fname = self.save_path / f"{self.subject}-info.fif"
            # write_info doesn't have overwrite, it overwrites by default
            self.logger.info(f"Saving info object to file: {save_fname}")
            mne.io.write_info(save_fname, info)

        return info

    def handle_forward_solution(self, info: mne.Info, src: mne.SourceSpaces, bem: mne.bem.ConductorModel) -> mne.Forward:
        """
        Load or create the forward solution.

        Reads the forward solution from the file specified in
        `config['forward_solution']['fname']` if it exists and has a valid suffix.
        Otherwise, creates a new forward solution using `mne.make_forward_solution`
        with parameters from the configuration. Converts the solution to fixed
        orientation if `config['forward_solution']['orientation_type']` is 'fixed'.
        Saves the created forward solution if `config['forward_solution']['save']` is True.

        Parameters
        ----------
        info : mne.Info
            The measurement info object.
        src : mne.SourceSpaces
            The source space object.
        bem : mne.bem.ConductorModel
            The BEM solution object.

        Returns
        -------
        mne.Forward
            The loaded or created forward solution object.
        """
        forward_solution_cfg = self.config.get("forward_solution", {})
        orientation_type = forward_solution_cfg.get("orientation_type", "free") # Default to free
        fname = forward_solution_cfg.get("fname")

        if fname:
            load_path = Path(fname)
            if load_path.exists():
                # Basic check for expected suffix
                if not (load_path.name.endswith("fixed-fwd.fif") or load_path.name.endswith("free-fwd.fif")):
                    self.logger.warning(f"File {load_path} does not have a standard forward solution suffix. Attempting to load anyway.")
                try:
                    self.logger.info(f"Loading forward solution from file: {load_path}")
                    fwd = mne.read_forward_solution(load_path)
                    # Optional: Check if loaded orientation matches config
                    loaded_ori = "fixed" if fwd["source_ori"] == FIFF.FIFFV_MNE_FIXED_ORI else "free"
                    if loaded_ori != orientation_type:
                         self.logger.warning(f"Loaded forward solution orientation ('{loaded_ori}') does not match config ('{orientation_type}'). Using loaded orientation.")
                         orientation_type = loaded_ori # Update based on loaded file
                    return fwd
                except Exception as e:
                    self.logger.warning(f"Failed to load forward solution from file: {e}. Proceeding to create it.")
            else:
                self.logger.warning(f"Forward solution file does not exist: {load_path}. Proceeding to create it.")
        else:
            self.logger.info("No forward solution file specified. Proceeding to create it.")

        self.logger.info("Creating forward solution...")
        valid_kwargs = ["trans", "eeg", "meg", "mindist", "ignore_ref", "n_jobs", "verbose"]
        kwargs = {key: value for key, value in forward_solution_cfg.items() if key in valid_kwargs}
        # Ensure trans is provided if needed
        if 'trans' not in kwargs or not kwargs['trans']:
             self.logger.warning("Transformation file ('trans') not specified in forward_solution config. Using default 'fsaverage-trans.fif'. This might be incorrect.")
             kwargs['trans'] = 'fsaverage-trans.fif' # Or handle error differently

        fwd = mne.make_forward_solution(
            info=info,
            src=src,
            bem=bem,
            **kwargs
        )
        self.logger.info("Forward solution created successfully.")

        if orientation_type == "fixed":
            surf_ori = forward_solution_cfg.get("surf_ori", True)
            force_fixed = forward_solution_cfg.get("force_fixed", True)
            self.logger.info(f"Converting forward solution to fixed orientation (surf_ori={surf_ori}, force_fixed={force_fixed})...")
            fwd = mne.convert_forward_solution(fwd, force_fixed=force_fixed, surf_ori=surf_ori)
            self.logger.info("Orientation fixed successfully.")
        elif fwd["source_ori"] != FIFF.FIFFV_MNE_FREE_ORI:
             # If config asked for free but make_forward_solution didn't yield it (unlikely)
             self.logger.warning("Expected free orientation but forward solution is not. Check configuration.")


        if forward_solution_cfg.get("save", False):
            # Determine final orientation type for saving filename
            final_orientation = "fixed" if fwd["source_ori"] == FIFF.FIFFV_MNE_FIXED_ORI else "free"
            save_fname = self.save_path / f"{self.subject}-{final_orientation}-fwd.fif"
            verbose = forward_solution_cfg.get("verbose")
            overwrite = forward_solution_cfg.get("overwrite", False)
            self.logger.info(f"Saving forward solution to file: {save_fname}")
            mne.write_forward_solution(save_fname, fwd, overwrite=overwrite, verbose=verbose)

        return fwd

    def handle_leadfield(self, fwd: Optional[mne.Forward] = None) -> np.ndarray:
        """
        Load or extract the leadfield matrix.

        Reads the leadfield matrix from the .npz file specified in
        `config['leadfield']['fname']` if it exists. Otherwise, extracts the
        leadfield data from the provided forward solution `fwd`. Reshapes the
        matrix for free orientation if necessary. Saves the extracted leadfield
        matrix if `config['leadfield']['save']` is True.

        Parameters
        ----------
        fwd : mne.Forward, optional
            The forward solution object. Required if the leadfield matrix is not
            loaded from a file, by default None.

        Returns
        -------
        np.ndarray
            The leadfield matrix. Shape is (n_sensors, n_sources) for fixed
            orientation or (n_sensors, n_sources, 3) for free orientation.

        Raises
        ------
        ValueError
            If the leadfield cannot be loaded from a file and `fwd` is not provided,
            or if a loaded leadfield file is invalid.
        """
        leadfield_cfg = self.config.get("leadfield", {})
        fname = leadfield_cfg.get("fname")

        if fname:
            load_path = Path(fname)
            if load_path.exists():
                try:
                    self.logger.info(f"Loading leadfield matrix from file: {load_path}")
                    with np.load(load_path) as data:
                        if "leadfield" not in data:
                            raise ValueError(f"Invalid .npz file: 'leadfield' key not found in {load_path}")
                        leadfield = data["leadfield"]

                    # Basic validation based on expected dimensions
                    if leadfield.ndim == 2:
                        self.logger.info(f"Loaded fixed orientation leadfield: {leadfield.shape}")
                    elif leadfield.ndim == 3 and leadfield.shape[-1] == 3:
                        self.logger.info(f"Loaded free orientation leadfield: {leadfield.shape}")
                    else:
                        raise ValueError(f"Loaded leadfield has unexpected shape: {leadfield.shape}")
                    return leadfield
                except Exception as e:
                    self.logger.warning(f"Failed to load leadfield matrix from file: {e}. Proceeding to compute it.")
            else:
                self.logger.warning(f"Leadfield file does not exist: {load_path}. Proceeding to compute it.")
        else:
            self.logger.info("No leadfield file specified. Proceeding to compute it.")

        if fwd is None:
            raise ValueError("Forward solution (fwd) must be provided to compute leadfield matrix when not loading from file.")

        self.logger.info("Extracting leadfield matrix from the forward solution...")
        leadfield = fwd["sol"]["data"] 
        orientation_type = None

        if fwd["source_ori"] == FIFF.FIFFV_MNE_FIXED_ORI:
            self.logger.info(f"Extracted fixed orientation leadfield: {leadfield.shape}")
            orientation_type = "fixed"
            # Shape is already (n_sensors, n_sources)
        elif fwd["source_ori"] == FIFF.FIFFV_MNE_FREE_ORI:
            self.logger.info("Extracted free orientation leadfield (raw shape: %s)", leadfield.shape)
            n_sensors, n_sources_x_orient = leadfield.shape
            n_orient = 3
            if n_sources_x_orient % n_orient != 0:
                 raise ValueError(f"Cannot reshape free orientation leadfield. Shape {leadfield.shape} not divisible by 3.")
            n_sources = n_sources_x_orient // n_orient
            leadfield = leadfield.reshape(n_sensors, n_sources, n_orient)
            self.logger.info(f"Reshaped free orientation leadfield: {leadfield.shape}")
            orientation_type = "free"
        else:
            self.logger.warning("Unknown leadfield orientation type in forward solution.")

        if leadfield_cfg.get("save", False) and orientation_type:
            save_fname = self.save_path / f"{self.subject}-leadfield-{orientation_type}.npz"
            self.logger.info(f"Saving computed leadfield matrix to file: {save_fname}")
            np.savez(save_fname, leadfield=leadfield)

        return leadfield

    def simulate(self) -> np.ndarray:
        """
        Run the full simulation pipeline.

        Executes the sequence of handling/creating source space, BEM model,
        info object, forward solution, and finally the leadfield matrix,
        based on the provided configuration.

        Returns
        -------
        np.ndarray
            The final leadfield matrix. Shape depends on the orientation type.

        Raises
        ------
        RuntimeError
            If any step in the pipeline fails.
        """
        self.logger.info("Starting leadfield simulation pipeline...")
        try:
            src = self.handle_source_space()
            bem = self.handle_bem_model()
            info = self.handle_info()
            fwd = self.handle_forward_solution(info, src, bem)
            leadfield = self.handle_leadfield(fwd)
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True) # Log traceback
            raise RuntimeError("Simulation pipeline failed.") from e

        self.logger.info("Leadfield simulation pipeline completed successfully.")
        return leadfield