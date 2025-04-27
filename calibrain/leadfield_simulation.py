import logging
import numpy as np
from pathlib import Path
from mne.io.constants import FIFF
import mne

# Mdular and configurable framework for setting up and running leadfield simulations pipeline. Handle various MNE-Python components, such as source space, BEM model, montage, info object, forward solution, and leadfield matrix.
class LeadfieldSimulator:
    def __init__(self, config: dict, logger=None):
        """
        Initialize the LeadfieldSimulator class with a configuration dictionary.

        Parameters:
        - config (dict): Configuration dictionary.
        - logger (logging.Logger, optional): Custom logger. If None, a default logger is created.
        """
        self.logger = logger if logger else logging.getLogger(__name__)
        self.config = config

        # Extract common parameters
        data_cfg = config["data"]
        self.data_path = Path(data_cfg["data_path"])
        self.subject = data_cfg["subject"]
        self.subjects_dir = Path(data_cfg["subjects_dir"])
        self.save_path = Path(data_cfg["save_path"])  # Save path for all outputs

        # Validate paths
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
        Handle the source space by either loading it from a file or creating it.
    
        Returns:
        - src (mne.SourceSpaces): The source space object.
        """
        source_space_cfg = self.config["source_space"]
        if "fname" in source_space_cfg and source_space_cfg["fname"]:
            load_path = Path(source_space_cfg["fname"])
            if load_path.exists():
                self.logger.info(f"Loading source space from file: {load_path}")
                return mne.read_source_spaces(load_path)
            else:
                self.logger.warning(f"Source space file does not exist: {load_path}. Creating from scratch...")
        else:
            self.logger.info("No source space file specified. Creating from scratch...")
    
        # Create source space
        valid_kwargs = ["spacing", "surface", "add_dist", "n_jobs", "verbose"]
        kwargs = {key: value for key, value in source_space_cfg.items() if key in valid_kwargs}
        src = mne.setup_source_space(subject=self.subject, subjects_dir=self.subjects_dir, **kwargs)
    
        # Save the source space if configured
        if source_space_cfg.get("save", False):
            save_path = self.save_path / f"{self.subject}-src.fif"
            verbose = source_space_cfg.get("verbose", None)
            overwrite = source_space_cfg.get("overwrite", None)
            self.logger.debug(f"Saving source space to file: {save_path} (overwrite={overwrite}, verbose={verbose})")
            mne.write_source_spaces(save_path, src, overwrite=overwrite, verbose=verbose)
    
        return src
        
    def handle_bem_model(self) -> mne.bem.ConductorModel:
        """
        Handle the BEM model by either loading it from a file or creating it.
    
        Returns:
        - bem (mne.bem.ConductorModel): The BEM model object.
        """
        bem_model_cfg = self.config["bem_model"]
        if "fname" in bem_model_cfg and bem_model_cfg["fname"]:
            load_path = Path(bem_model_cfg["fname"])
            if load_path.exists():
                self.logger.info(f"Loading BEM model from file: {load_path}")
                return mne.read_bem_solution(load_path)
            else:
                self.logger.warning(f"BEM model file does not exist: {load_path}. Creating from scratch...")
        else:
            self.logger.info("No BEM model file specified. Creating from scratch...")
    
        # Create BEM model
        valid_model_kwargs = ["ico", "conductivity", "verbose"]
        model_kwargs = {key: value for key, value in bem_model_cfg.items() if key in valid_model_kwargs}
        bem_model = mne.make_bem_model(subject=self.subject, subjects_dir=self.subjects_dir, **model_kwargs)
    
        valid_sol_kwargs = ["solver", "verbose"]
        sol_kwargs = {key: value for key, value in bem_model_cfg.items() if key in valid_sol_kwargs}
        bem = mne.make_bem_solution(surfs=bem_model, **sol_kwargs)
    
        # Save the BEM model if configured
        if bem_model_cfg.get("save", False):
            save_path = self.save_path / f"{self.subject}-bem.fif"
            verbose = bem_model_cfg.get("verbose", None)
            overwrite = bem_model_cfg.get("overwrite", False)
            self.logger.debug(f"Saving BEM model to file: {save_path} (overwrite={overwrite}, verbose={verbose})")
            mne.write_bem_solution(save_path, bem, overwrite=overwrite, verbose=verbose)
    
        return bem
        
    def handle_montage(self) -> mne.channels.DigMontage:
        """
        Handle the montage by either loading it from a file or creating it.
    
        Returns:
        - montage (mne.channels.DigMontage): The montage object.
        """
        montage_cfg = self.config.get("montage", {})
        if "fname" in montage_cfg and montage_cfg["fname"]:
            load_path = Path(montage_cfg["fname"])
            if load_path.exists():
                self.logger.info(f"Loading montage from file: {load_path}")
                return mne.channels.read_dig_fif(load_path)
            else:
                self.logger.warning(f"Montage file does not exist: {load_path}. Creating from scratch...")
        else:
            self.logger.info("No montage file specified. Creating from scratch...")
    
        # Create montage
        valid_kwargs = ["kind", "head_size"]
        kwargs = {key: value for key, value in montage_cfg.items() if key in valid_kwargs}
        montage = mne.channels.make_standard_montage(**kwargs)
    
        # Save the montage if configured
        if montage_cfg.get("save", False):
            save_path = self.save_path / f"{self.subject}-montage.fif"
            verbose = montage_cfg.get("verbose", None)
            overwrite = montage_cfg.get("overwrite", False)
            self.logger.debug(f"Saving montage to file: {save_path} (overwrite={overwrite}, verbose={verbose})")
            montage.save(save_path, overwrite=overwrite, verbose=verbose)
    
        return montage
        
    def handle_info(self) -> mne.Info:
        """
        Handle the MNE info object by either loading it from a file or creating it.
    
        Returns:
        - info (mne.Info): The MNE info object.
        """
        info_cfg = self.config["info"]
        if "fname" in info_cfg and info_cfg["fname"]:
            load_path = Path(info_cfg["fname"])
            if load_path.exists():
                self.logger.info(f"Loading info object from file: {load_path}")
                return mne.io.read_info(load_path)
            else:
                self.logger.warning(f"Info file does not exist: {load_path}. Creating from scratch...")
        else:
            self.logger.info("No info file specified. Creating from scratch...")
    
        # Handle the montage
        montage = self.handle_montage()
    
        # Create the info object
        valid_kwargs = ["sfreq", "ch_types", "verbose"]
        kwargs = {key: value for key, value in info_cfg.items() if key in valid_kwargs}
        info = mne.create_info(
            ch_names=montage.ch_names,
            **kwargs
        )
        info.set_montage(montage)
    
        # Save the info object if configured
        if info_cfg.get("save", False):
            save_path = self.save_path / f"{self.subject}-info.fif"
            verbose = info_cfg.get("verbose", False)
            self.logger.debug(f"Saving info object to file: {save_path} (verbose={verbose})")
            mne.io.write_info(save_path, info)
    
        return info
    
    def handle_forward_solution(self, info, src, bem) -> mne.Forward:
        """
        Handle the forward solution by either loading it from a file or creating it.
    
        Parameters:
        - info (mne.Info): The measurement info object.
        - src (mne.SourceSpaces): The source space object.
        - bem (mne.bem.ConductorModel): The BEM model object.
    
        Returns:
        - fwd (mne.Forward): The forward solution object.
        """
        forward_solution_cfg = self.config.get("forward_solution", {})
        orientation_type = forward_solution_cfg.get("orientation_type")
        fname = forward_solution_cfg.get("fname")
    
        # Check if a file path is provided and if the file exists
        if fname:
            load_path = Path(fname)
            if load_path.exists():
                # Check if the file name ends with the expected suffix
                if not (load_path.name.endswith("fixed-fwd.fif") or load_path.name.endswith("free-fwd.fif")):
                    self.logger.warning(f"File {load_path} does not have a valid forward solution suffix ('fixed-fwd.fif' or 'free-fwd.fif'). Proceeding to create it.")
                else:
                    try:
                        self.logger.info(f"Loading forward solution from file: {load_path}")
                        fwd = mne.read_forward_solution(load_path)
                        return fwd
                    except Exception as e:
                        self.logger.warning(f"Failed to load forward solution from file: {e}. Proceeding to create it.")
            else:
                self.logger.warning(f"Forward solution file does not exist: {load_path}. Proceeding to create it.")
        else:
            self.logger.info("No forward solution file specified. Proceeding to create it.")
    
        # Create the forward solution if not loaded
        self.logger.info("Creating forward solution...")
        valid_kwargs = ["trans", "eeg", "meg", "mindist", "ignore_ref", "n_jobs", "verbose"]
        kwargs = {key: value for key, value in forward_solution_cfg.items() if key in valid_kwargs}
        fwd = mne.make_forward_solution(
            info=info,
            src=src,
            bem=bem,
            **kwargs
        )
        self.logger.info("Forward solution created successfully.")
    
        # Fix orientation if configured
        if orientation_type == "fixed":
            surf_ori = forward_solution_cfg.get("surf_ori", True)
            force_fixed = forward_solution_cfg.get("force_fixed", True)
            self.logger.info(f"Fixing orientation of the forward solution (surf_ori={surf_ori}, force_fixed={force_fixed})...")
            fwd = mne.convert_forward_solution(fwd, force_fixed=force_fixed, surf_ori=surf_ori)
            self.logger.info("Orientation fixed successfully.")
    
        # Save the forward solution if configured
        if forward_solution_cfg.get("save", False):
            save_path = self.save_path / f"{self.subject}-{orientation_type}-fwd.fif"
            verbose = forward_solution_cfg.get("verbose", False)
            overwrite = forward_solution_cfg.get("overwrite", False)
            self.logger.debug(f"Saving forward solution to file: {save_path} (overwrite={overwrite}, verbose={verbose})")
            mne.write_forward_solution(save_path, fwd, overwrite=overwrite, verbose=verbose)
            self.logger.info(f"Forward solution saved successfully to {save_path}.")
    
        return fwd

    def handle_leadfield(self, fwd=None) -> np.ndarray:
        """
        Handle the leadfield matrix by either loading it from a file or extracting it from the forward solution.
    
        Parameters:
        - fwd (mne.Forward, optional): The forward solution. If None, the leadfield matrix is computed.
    
        Returns:
        - leadfield (np.ndarray): The leadfield matrix.
    
        Raises:
        - ValueError: If neither a valid forward solution nor a valid file path is provided.
        """
        leadfield_cfg = self.config.get("leadfield", {})
        fname = leadfield_cfg.get("fname", "")  # File path to load the leadfield matrix
    
        # Check if a file path is provided and if the file exists
        if fname:
            load_path = Path(fname)
            if load_path.exists():
                try:
                    self.logger.info(f"Loading leadfield matrix from file: {load_path}")
                    with np.load(load_path) as data:
                        if "leadfield" not in data:
                            self.logger.error(f"File {load_path} does not contain 'leadfield' key.")
                            raise ValueError(f"Invalid .npz file: 'leadfield' key not found in {load_path}")
                        leadfield = data["leadfield"]
    
                    # Validate and reshape the leadfield matrix based on its orientation
                    if leadfield.ndim == 2:  # Fixed orientation
                        self.logger.info("Loaded leadfield matrix is fixed orientation.")
                        self.logger.info(f"Leadfield shape: {leadfield.shape[0]} sensors x {leadfield.shape[1]} sources")
                    elif leadfield.ndim == 3 and leadfield.shape[2] == 3:  # Free orientation
                        self.logger.info("Loaded leadfield matrix is free orientation.")
                        self.logger.info(f"Leadfield shape: {leadfield.shape[0]} sensors x {leadfield.shape[1]} sources x {leadfield.shape[2]} orientations")
                    else:
                        self.logger.error(f"Invalid leadfield matrix shape: {leadfield.shape}")
                        raise ValueError(f"Invalid leadfield matrix shape: {leadfield.shape}")
    
                    return leadfield
                except Exception as e:
                    self.logger.warning(f"Failed to load leadfield matrix from file: {e}. Proceeding to compute it.")
            else:
                self.logger.warning(f"Leadfield file does not exist: {load_path}. Proceeding to compute it.")
        else:
            self.logger.info("No leadfield file specified. Proceeding to compute it.")
    
        # If no file path is provided or loading fails, compute the leadfield matrix
        if fwd is None:
            self.logger.error("Forward solution not provided. Cannot compute leadfield matrix.")
            raise ValueError("Forward solution is missing. Cannot compute leadfield matrix.")
    
        self.logger.info("Extracting leadfield matrix from the forward solution...")
        leadfield = fwd["sol"]["data"]
    
        # Determine the orientation type and reshape/save accordingly
        if fwd["source_ori"] == FIFF.FIFFV_MNE_FIXED_ORI:  # Fixed orientation
            self.logger.info("Leadfield matrix is fixed orientation.")
            self.logger.info(f"Leadfield extracted with shape {leadfield.shape[0]} sensors x {leadfield.shape[1]} sources")
            orientation_type = "fixed"
        elif fwd["source_ori"] == FIFF.FIFFV_MNE_FREE_ORI:  # Free orientation
            self.logger.info("Leadfield matrix is free orientation.")
            n_sensors, n_sources_times_orientations = leadfield.shape
            n_orientations = 3  # Free orientation implies 3 orientations per source
            n_sources = n_sources_times_orientations // n_orientations
    
            # Reshape the leadfield to sensors x sources x n_orientations
            leadfield = leadfield.reshape(n_sensors, n_sources, n_orientations)
            self.logger.info(f"Leadfield reshaped to {leadfield.shape[0]} sensors x {leadfield.shape[1]} sources x {leadfield.shape[2]} orientations")
            orientation_type = "free"
        else:
            self.logger.warning("Unknown leadfield orientation type. Proceeding without saving.")
            orientation_type = None
    
        # Save the computed leadfield matrix if configured
        if leadfield_cfg.get("save", False) and orientation_type:
            save_path = self.save_path / f"{self.subject}-leadfield-{orientation_type}.npz"
            self.logger.info(f"Saving computed leadfield matrix to file: {save_path}")
            np.savez(save_path, leadfield=leadfield)
    
        self.logger.info(f"Leadfield computed with shape {leadfield.shape}")
        return leadfield
        
    def simulate(self) -> np.ndarray:
        """
        Run the full simulation pipeline using the configuration.
        This includes loading or creating the source space, BEM model, info object,
        forward solution, and extracting the leadfield matrix.

        Returns:
        - leadfield (np.ndarray): The extracted leadfield matrix.

        Raises:
        - Exception: If any step in the pipeline fails.
        """
        self.logger.info("Running the simulation pipeline...")

        try:
            # Handle or create the required objects
            src = self.handle_source_space()
            bem = self.handle_bem_model()
            info = self.handle_info()
            fwd = self.handle_forward_solution(info, src, bem)

            # Extract the leadfield matrix
            leadfield = self.handle_leadfield(fwd)
        except Exception as e:
            self.logger.error(f"Pipeline failed at step: {e}")
            raise RuntimeError("Simulation pipeline failed.") from e

        self.logger.info("Pipeline completed successfully.")
        return leadfield