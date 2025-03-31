import argparse
import logging
import numpy as np
from pathlib import Path
import mne
from mne.datasets import sample
from calibrain.utils import load_config


# Mdular and configurable framework for setting up and running dipole simulations pipeline. Handle various MNE-Python components, such as source space, BEM model, montage, info object, forward solution, and leadfield matrix.
class DipoleSimulation:
    def __init__(self, config: dict, logger=None):
        """
        Initialize the DipoleSimulation class with a configuration dictionary.

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

        self.logger.info("DipoleSimulation initialized successfully.")
        
    def handle_source_space(self) -> mne.SourceSpaces:
        """
        Handle the source space by either loading it from a file or creating it.
    
        Returns:
        - src (mne.SourceSpaces): The source space object.
        """
        source_space_cfg = self.config["source_space"]
        if "fname" in source_space_cfg and source_space_cfg["fname"]:
            self.logger.info(f"Loading source space from file: {source_space_cfg['fname']}")
            return mne.read_source_spaces(source_space_cfg["fname"])
        else:
            self.logger.info("Creating source space from scratch...")
            valid_kwargs = ["spacing", "surface", "add_dist", "n_jobs", "verbose"]
            kwargs = {key: value for key, value in source_space_cfg.items() if key in valid_kwargs}
            src = mne.setup_source_space(subject=self.subject, subjects_dir=self.subjects_dir, **kwargs)
    
            # Save the source space if configured
            if source_space_cfg.get("save", False):
                save_path = self.save_path / f"{self.subject}-src.fif"
                verbose = source_space_cfg.get("verbose", None) 
                overwrite = source_space_cfg.get("overwrite", False)
    
                # Log the save operation
                self.logger.info(f"Saving source space to file: {save_path} (overwrite={overwrite}, verbose={verbose})")
    
                # Save the source space
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
            self.logger.info(f"Loading BEM model from file: {bem_model_cfg['fname']}")
            return mne.read_bem_solution(bem_model_cfg["fname"])
        else:
            self.logger.info("Creating BEM model from scratch...")
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

                # Log the save operation
                self.logger.info(f"Saving BEM model to file: {save_path} (overwrite={overwrite}, verbose={verbose})")

                # Save the BEM model
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
            self.logger.info(f"Loading montage from file: {montage_cfg['fname']}")
            return mne.channels.read_dig_fif(montage_cfg["fname"])
        else:
            self.logger.info("Creating montage from scratch...")
            valid_kwargs = ["kind", "head_size"]
            kwargs = {key: value for key, value in montage_cfg.items() if key in valid_kwargs}
            montage = mne.channels.make_standard_montage(**kwargs)
    
            # Save the montage if configured
            if montage_cfg.get("save", False):
                save_path = self.save_path / f"{self.subject}-montage.fif"
                verbose = montage_cfg.get("verbose", None)
                overwrite = montage_cfg.get("overwrite", False)
    
                # Log the save operation
                self.logger.info(f"Saving montage to file: {save_path} (overwrite={overwrite}, verbose={verbose})")
    
                # Save the montage
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
            self.logger.info(f"Loading info object from file: {info_cfg['fname']}")
            return mne.io.read_info(info_cfg["fname"])
        else:
            self.logger.info("Creating info object from scratch...")
    
            # Handle the montage
            montage = self.handle_montage()
    
            # Filter valid kwargs for creating the info object
            valid_kwargs = ["sfreq", "ch_types", "verbose"]
            kwargs = {key: value for key, value in info_cfg.items() if key in valid_kwargs}
    
            # Create the info object
            info = mne.create_info(
                ch_names=montage.ch_names,
                **kwargs
            )
            info.set_montage(montage)
    
            # Save the info object if configured
            if info_cfg.get("save", False):
                save_path = self.save_path / f"{self.subject}-info.fif"
                verbose = info_cfg.get("verbose", False)
    
                # Log the save operation
                self.logger.info(f"Saving info object to file: {save_path} (verbose={verbose})")
    
                # Save the info object
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
        forward_solution_cfg = self.config["forward_solution"]
        if "fname" in forward_solution_cfg and forward_solution_cfg["fname"]:
            self.logger.info(f"Loading forward solution from file: {forward_solution_cfg['fname']}")
            return mne.read_forward_solution(forward_solution_cfg["fname"])
        else:
            self.logger.info("Creating forward solution from scratch...")
            valid_kwargs = ["trans", "eeg", "meg", "mindist", "ignore_ref", "n_jobs", "verbose"]
            kwargs = {key: value for key, value in forward_solution_cfg.items() if key in valid_kwargs}
            fwd = mne.make_forward_solution(
                info=info,
                src=src,
                bem=bem,
                **kwargs
            )
    
            # Save the forward solution if configured
            if forward_solution_cfg.get("save", False):
                save_path = self.save_path / f"{self.subject}-fwd.fif"
                verbose = forward_solution_cfg.get("verbose", None)
                overwrite = forward_solution_cfg.get("overwrite", False)
    
                # Log the save operation
                self.logger.info(f"Saving forward solution to file: {save_path} (overwrite={overwrite}, verbose={verbose})")
    
                # Save the forward solution
                mne.write_forward_solution(save_path, fwd, overwrite=overwrite, verbose=verbose)
    
            return fwd

    def handle_leadfield(self, fwd) -> np.ndarray:
        """
        Handle the leadfield matrix by either loading it from a file or extracting it from the forward solution.
    
        Parameters:
        - fwd (mne.Forward): The forward solution.
    
        Returns:
        - leadfield (np.ndarray): The leadfield matrix.
        """
        leadfield_cfg = self.config.get("leadfield", {})
        fname = leadfield_cfg.get("fname", "")  # File path to load the leadfield matrix
    
        # If a file path is provided and the file exists, load the leadfield matrix
        if fname:
            load_path = Path(fname)
            if load_path.exists():
                self.logger.info(f"Loading leadfield matrix from file: {load_path}")
                leadfield = np.load(load_path)
                self.logger.info(f"Leadfield loaded with shape {leadfield.shape}")
                return leadfield
            else:
                self.logger.warning(f"Specified leadfield file does not exist: {load_path}")
    
        # Ensure the forward solution is provided
        if fwd is None:
            self.logger.error("Forward solution not found. Generate it first.")
            raise ValueError("Forward solution is missing. Cannot extract leadfield matrix.")
    
        # Extract the leadfield matrix from the forward solution
        self.logger.info("Extracting leadfield matrix from the forward solution...")
        leadfield = fwd["sol"]["data"]
        self.logger.info(f"Leadfield extracted with shape {leadfield.shape}")
    
        # Save the leadfield matrix if configured
        if leadfield_cfg.get("save", False):
            save_path = self.save_path / f"{self.subject}-leadfield.npy"
            self.logger.info(f"Saving leadfield matrix to file: {save_path}")
            allow_pickle = leadfield_cfg.get("allow_pickle", False)
            np.save(save_path, leadfield, allow_pickle)
    
        return leadfield
        
    def run_pipeline(self) -> np.ndarray:
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
            self.logger.error(f"Pipeline failed: {e}")
            raise

        self.logger.info("Pipeline completed successfully.")
        return leadfield


def main(args=None):
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the dipole simulation pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file of dipole simulation."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level. Overrides the value in the configuration file."
    )

    # Parse arguments (use provided args if given, otherwise use sys.argv)
    args = parser.parse_args(args)

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    config = load_config(config_path)

    # Determine log level
    log_level = args.log_level or config.get("log_level", "INFO").upper()
    if log_level not in ["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        raise ValueError(f"Invalid log level: {log_level}")

    # Configure logging
    logging.basicConfig(level=getattr(logging, log_level), format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.info(f"Using configuration file: {config_path}")

    # Validate and update paths in the configuration
    data_cfg = config.get("data", {})
    if "data_path" not in data_cfg or "subjects_dir" not in data_cfg or "subject" not in data_cfg:
        logger.error("The configuration file must include 'data_path', 'subjects_dir', and 'subject' in the 'data' section.")
        raise ValueError("Invalid configuration file: Missing required keys in the 'data' section.")

    # Ensure paths are valid
    data_path = Path(data_cfg["data_path"])
    subjects_dir = Path(data_cfg["subjects_dir"])
    subject = data_cfg["subject"]

    if not data_path.exists():
        logger.error(f"Data path does not exist: {data_path}")
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    if not subjects_dir.exists():
        logger.error(f"Subjects directory does not exist: {subjects_dir}")
        raise FileNotFoundError(f"Subjects directory does not exist: {subjects_dir}")

    # Ensure save_path exists
    save_path = Path(data_cfg.get("save_path", "./results"))
    if not save_path.exists():
        logger.info(f"Save path does not exist. Creating: {save_path}")
        save_path.mkdir(parents=True, exist_ok=True)

    # Update config with validated paths
    config["data"]["data_path"] = str(data_path)
    config["data"]["subjects_dir"] = str(subjects_dir)
    config["data"]["subject"] = subject
    config["data"]["save_path"] = str(save_path)

    # Initialize simulation
    dipole_sim = DipoleSimulation(config=config, logger=logger)

    # Run the pipeline
    try:
        leadfield = dipole_sim.run_pipeline()
        logger.info(f"Leadfield shape: {leadfield.shape}")
    except Exception as e:
        logger.error(f"An error occurred during the pipeline execution: {e}")
        raise


if __name__ == "__main__":
    # Programmatically specify arguments
    main([
        "--config", "examples/dipole_sim_cfg.yml",
        "--log-level", "INFO"
    ])