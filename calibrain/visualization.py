import logging
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Tuple, Union, Sequence, Literal
from matplotlib import cm, gridspec
import mne
from mne.io.constants import FIFF
import matplotlib.lines as mlines # For creating custom legend handles

class Visualizer:
    def __init__(self, base_save_path: str = "results/figures", logger: Optional[logging.Logger] = None):
        self.base_save_path = Path(base_save_path)
        self.logger = logger or logging.getLogger(__name__)

    def _handle_figure_output(
        self,
        fig: plt.Figure,
        file_name: str,
        save_path: Optional[str] = None,
        show: bool = False,
    ):
        save_dir = Path(save_path) if save_path else self.base_save_path
        if not save_dir.is_absolute():
            save_dir = self.base_save_path / save_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        if Path(file_name).suffix.lower() not in ['.png', '.jpg', '.jpeg']:
            file_name += ".png"

        full_path = save_dir / file_name
        fig.savefig(full_path, bbox_inches="tight")
        self.logger.debug(f"Figure saved to {full_path}")
        if show:
            plt.show()
        # Don't close the figure immediately if show=True to allow Sphinx Gallery to capture it
        if not show:
            plt.close(fig)

    # --------------------------------------------
    # --- Vizualisation for source and sensor data
    # --------------------------------------------
    # --- plot sources
    def _plot_sources_single_trial(
        self,
        ERP_config: dict,
        x_trial: np.ndarray,
        active_indices: Optional[Sequence[int]],
        units: Optional[str],
        trial_idx: int,
        title: str,
    ) -> plt.Figure:
        tmin, tmax, stim_onset, _, times = self._get_plot_params(ERP_config, x_trial.shape[-1])

        x_plot = np.linalg.norm(x_trial, axis=1) if x_trial.ndim == 3 else x_trial
        if active_indices is None:
            active_indices = np.arange(x_plot.shape[0])

        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        colors = cm.viridis(np.linspace(0, 1, len(active_indices)))

        for i, src_idx in enumerate(active_indices):
            ax.plot(times, x_plot[src_idx], label=f"Source {src_idx}", linewidth=1.5, color=colors[i])

        ax.axvline(x=stim_onset, linestyle="--", color="gray", label="Stimulus Onset")
        ax.axvline(x=tmin, linestyle=":", color="black", linewidth=1.0)
        ax.axvline(x=tmax, linestyle=":", color="black", linewidth=1.0)

        ax.set_xticks([tmin, stim_onset, tmax])
        ax.set_xticklabels([f"{tick:.2f}s" for tick in [tmin, stim_onset, tmax]])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Amplitude ({units})")
        ax.set_title(f"{title} (Trial {trial_idx + 1})")
        ax.grid(True, alpha=0.6)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0., fontsize='small')
        return fig

    def _plot_sources_all_trials(
        self,
        ERP_config: dict,
        x_trials: np.ndarray,
        active_indices: Optional[Sequence[Sequence[int]]] = None,
        units: Optional[str] = None,
        title: str = "Source Signals"
    ) -> plt.Figure:
        tmin, tmax, stim_onset, _, times = self._get_plot_params(ERP_config, x_trials.shape[-1])        
        n_trials = x_trials.shape[0]

        fig, axes = plt.subplots(nrows=n_trials, ncols=1, figsize=(12, 3 * n_trials), sharex=True, constrained_layout=True, sharey=True)
        if n_trials == 1:
            axes = [axes]

        for i in range(n_trials):
            ax = axes[i]
            x_trial = x_trials[i]
            x_plot = np.linalg.norm(x_trial, axis=1) if x_trial.ndim == 3 else x_trial
            indices = active_indices[i] if active_indices is not None else np.arange(x_plot.shape[0])
            colors = cm.viridis(np.linspace(0, 1, len(indices)))

            for j, src_idx in enumerate(indices):
                ax.plot(times, x_plot[src_idx], label=f"Source {src_idx}", linewidth=1.0, color=colors[j])

            ax.axvline(x=stim_onset, linestyle="--", color="gray", label="Stimulus Onset")
            ax.axvline(x=tmin, linestyle=":", color="black", linewidth=0.8)
            ax.axvline(x=tmax, linestyle=":", color="black", linewidth=0.8)
            ax.set_ylabel(f"Amplitude ({units})")
            ax.set_title(f"{title} — Trial {i + 1}")
            ax.grid(True, alpha=0.5)
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0., fontsize='small')

        axes[-1].set_xticks([tmin, stim_onset, tmax])
        axes[-1].set_xticklabels([f"{tick:.2f}s" for tick in [tmin, stim_onset, tmax]])
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(title, fontsize=16)

        return fig
    
    def plot_source_signals(
        self,
        ERP_config: dict,
        x_trials: np.ndarray,
        x_active_indices: Optional[Union[np.ndarray, Sequence[Sequence[int]]]] = None,
        units: Optional[str] = None,
        trial_idx: Optional[int] = None,
        title: Optional[str] = "Source Signals",
        save_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        show: bool = True,
    ):
        # convert the data from A to nAm for better readability
        if units == FIFF.FIFF_UNIT_AM:
            x_scaled = x_trials * 1e9  
            units = "nAm"
        else:
            raise ValueError(f"Unsupported units for source signals: {units}. Expected FIFF.FIFF_UNIT_AM.")

        if trial_idx is not None:
            indices = None
            if x_active_indices is not None:
                indices = x_active_indices[trial_idx] if isinstance(x_active_indices, (list, np.ndarray)) else x_active_indices

            fig = self._plot_sources_single_trial(
                ERP_config=ERP_config,
                x_trial=x_scaled if x_scaled.ndim < 3 else x_scaled[trial_idx],
                active_indices=indices,
                units=units,
                trial_idx=trial_idx,
                title=title,
            )
            file_name = file_name or f"source_signals_trial_{trial_idx + 1}"
        else:
            fig = self._plot_sources_all_trials(
                ERP_config=ERP_config,
                x_trials=x_scaled,
                active_indices=x_active_indices,
                units=units,
                title=title,
            )
            file_name = file_name or "source_signals_all_trials"

        self._handle_figure_output(fig, file_name, save_dir, show)

    # --- plot sensors        
    def _plot_sensors_single_trial(
        self,
        ERP_config: dict,
        y: np.ndarray,
        trial_idx: int,
        channels: Optional[Union[Sequence[int], str]],
        units: Optional[str],
        title: str,
        save_dir: Optional[str],
        file_name: Optional[str],
        show: bool
    ):
        tmin, tmax, stim_onset, _, times= self._get_plot_params(ERP_config, y.shape[-1])
        channels_to_plot = self._resolve_channels(y.shape[0], channels)
        
        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        self._plot_sensors(
            ax, y[channels_to_plot], times, stim_onset, tmin, tmax, channels_to_plot, units
        )
        ax.set_title(f"{title} (Trial {trial_idx + 1})")
        
        self._handle_figure_output(fig, file_name or f"{title.lower().replace(' ', '_')}_trial_{trial_idx + 1}", save_dir, show)
        
    def _plot_sensors_all_trials(
        self,
        ERP_config: dict,
        y_trials: np.ndarray,
        channels: Optional[Union[Sequence[int], str]],
        units: Optional[str],
        title: str,
        save_dir: Optional[str],
        file_name: Optional[str],
        show: bool
    ):
        n_trials = y_trials.shape[0]
        tmin, tmax, stim_onset, _, times = self._get_plot_params(ERP_config, y_trials.shape[-1])
        channels_to_plot = self._resolve_channels(y_trials.shape[1], channels)

        fig, axes = plt.subplots(nrows=n_trials, ncols=1, figsize=(12, 3 * n_trials), sharex=True, constrained_layout=True, sharey=True)
        if n_trials == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            self._plot_sensors(ax, y_trials[i, channels_to_plot], times, stim_onset, tmin, tmax, channels_to_plot, units)
            ax.set_title(f"Trial {i + 1}")

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(title, fontsize=16)
        self._handle_figure_output(fig, file_name or f"{title.lower().replace(' ', '_')}_all_trials", save_dir, show)

    def _plot_concatenated_sensor_trials(
        self,
        y_trials: np.ndarray,
        ERP_config: dict,
        channels: Optional[Union[Sequence[int], str]],
        units: Optional[str],
        title: str,
        save_dir: Optional[str],
        file_name: Optional[str],
        show: bool
    ):
        tmin, tmax, stim_onset, sfreq, _ = self._get_plot_params(ERP_config, y_trials.shape[-1])

        n_trials, n_sensors, _ = y_trials.shape
        trial_duration = tmax - tmin
        times_single = np.arange(tmin, tmax, 1.0 / sfreq)
        channels_to_plot = self._resolve_channels(n_sensors, channels)
        
        fig, ax = plt.subplots(figsize=(15, 6))
        colors = cm.viridis(np.linspace(0, 1, len(channels_to_plot)))

        for i in range(n_trials):
            trial_times = times_single + i * trial_duration
            for j, ch in enumerate(channels_to_plot):
                label = f"Ch {ch}" if i == 0 else None
                ax.plot(trial_times, y_trials[i, ch], color=colors[j], linewidth=1.2, alpha=0.85, label=label)
            ax.axvline(i * trial_duration + stim_onset, linestyle="--", color="gray", linewidth=1.0)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Amplitude ({units})")
        ax.set_title(title)
        ax.grid(True, alpha=0.4)
        if n_trials > 1 and len(channels_to_plot) <= 10:
            ax.legend(loc="upper right", fontsize="small")

        self._handle_figure_output(fig, file_name or f"{title.lower().replace(' ', '_')}_concatenated", save_dir, show)

    def _get_plot_params(self, ERP_config, n_times):
        tmin = ERP_config['tmin']
        tmax = ERP_config['tmax']
        stim_onset = ERP_config['stim_onset']
        sfreq = ERP_config['sfreq']
        times = np.arange(tmin, tmax, 1.0 / sfreq)[:n_times]
        return tmin, tmax, stim_onset, sfreq, times

    def _resolve_channels(self, n_sensors, channels):
        if channels is None or channels == "all":
            return np.arange(n_sensors)
        return np.array(channels)

    def _plot_sensors(self, ax, y: np.ndarray, times: np.ndarray, stim_onset: float, tmin: float, tmax: float, channels: Sequence[int], units: str):
        for i, ch in enumerate(y):
            label = f"Ch {channels[i]}" if len(channels) <= 10 else None
            ax.plot(times, ch, linewidth=1.0, alpha=0.8, label=label)
        ax.axvline(x=stim_onset, linestyle="--", color="gray", label="Stimulus Onset")
        ax.axvline(x=tmin, linestyle=":", color="black", linewidth=0.8)
        ax.axvline(x=tmax, linestyle=":", color="black", linewidth=0.8)
        ax.set_xticks([tmin, stim_onset, tmax])
        ax.set_xticklabels([f"{tmin:.2f}s", f"{stim_onset:.2f}s", f"{tmax:.2f}s"])
        ax.set_ylabel(f"Amplitude ({units})")
        ax.grid(True, alpha=0.5)
        if len(channels) <= 10:
            ax.legend(loc="upper right", fontsize="small")

    def plot_sensor_signals(
        self,
        ERP_config: dict,
        y_trials: np.ndarray,
        trial_idx: Optional[int] = None,
        channels: Optional[Union[Sequence[int], str]] = None,
        units: Optional[str] = None,
        mode: str = "stack",  # "stack" | "concatenate"
        title: str = "Sensor Signals",
        save_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        show: bool = True,
    ):
        if ERP_config is None:
            ERP_config = {
                "tmin": -0.5,
                "tmax": 0.5,
                "stim_onset": 0,
                "sfreq": 250,
                "fmin": 1,
                "fmax": 5,
                "amplitude": 20.0,
                "random_erp_timing": True,
                "erp_min_length": None,
        }
        # for better readability convert the data from T to fT for MEG magnetometers channels and T/m to fT/m for MEG gradiometers channels, and V to μV for EEG channels
        if units == FIFF.FIFF_UNIT_T:
            y_trials_scaled = y_trials * 1e15  # Convert Tesla to femtoTesla (fT)
            units = "fT"
        elif units == FIFF.FIFF_UNIT_T_M:
            y_trials_scaled = y_trials * 1e15  # Convert T/m to fT/m
            units = "fT/m"
        elif units == FIFF.FIFF_UNIT_V:
            y_trials_scaled = y_trials * 1e6  # Convert V to μV
            units = "μV"
        else:
            raise ValueError(f"Unsupported units for sensor signals: {units}. Expected FIFF.FIFF_UNIT_T, FIFF.FIFF_UNIT_T_M, or FIFF.FIFF_UNIT_V.")

        if mode == "stack":
            self._plot_sensors_all_trials(
                ERP_config, y_trials_scaled, channels, units, title, save_dir, file_name, show
            )
        elif mode == "concatenate":
            self._plot_concatenated_sensor_trials(
                y_trials_scaled, ERP_config, channels, units, title, save_dir, file_name, show
            )
        elif mode == "single":
            if trial_idx is None:
                trial_idx = 0
                self.logger.warning("No trial index provided, defaulting to 0.")
            self._plot_sensors_single_trial(
                ERP_config, y_trials_scaled[trial_idx], trial_idx, channels, units, title, save_dir, file_name, show
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def plot_stc_3d_brain(
        self,
        ERP_config: dict,
        x_one_trial,
        x_hat_one_trial,
        orientations,
        source_units,
        sample_idx,
        subject,
        subjects_dir,
        fwd_path,
        save_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        show: bool = False,
    ):
        """Plot source estimates with complete headless support."""
        
        if not show:
            # Skip 3D visualization entirely for headless mode
            self.logger.info("Skipping 3D brain visualization (show=False)")
            return
        
        # Only do 3D plotting when show=True
        original_backend = mne.viz.get_3d_backend()
        
        try:
            mne.viz.set_3d_backend('pyvistaqt')
            
            tmin, tmax, stim_onset, sfreq, times = self._get_plot_params(ERP_config, x.shape[-1])

            # Load forward solution
            fwd = mne.read_forward_solution(f"{fwd_path}/{subject}-fwd.fif")
            vertices = [src_hemi['vertno'] for src_hemi in fwd['src']]
            
            # Create source estimates
            stc_x_t0 = mne.SourceEstimate(x_one_trial[:, sample_idx], vertices=vertices, tmin=0, tstep=1/sfreq)
            stc_x_hat_t0 = mne.SourceEstimate(x_hat_one_trial[:, sample_idx], vertices=vertices, tmin=0, tstep=1/sfreq)

            if source_units == FIFF.FIFF_UNIT_AM:
                x_scaled = stc_x_t0 * 1e9  
                x_hat_scaled = stc_x_hat_t0 * 1e9  
                source_units = "nAm"
            else:
                raise ValueError(f"Unsupported units for source signals: {source_units}. Expected FIFF.FIFF_UNIT_AM.")

            source_estimates = [
                (x_scaled, 'Ground Truth'),
                (x_hat_scaled, 'Posterior Mean'),
            ]
            
            for stc, title in source_estimates:
                brain = stc.plot(
                    hemi="both",
                    subject=subject,
                    subjects_dir=subjects_dir,
                    spacing='ico4',
                    title=title,
                    time_viewer=True,  # Can be True since show=True
                    views=orientations,
                )
                
                if save_dir is not None:
                    brain_dir = os.path.join(save_dir, 'brain')
                    os.makedirs(brain_dir, exist_ok=True)
                    file_name = f"{file_name}_{title.replace(' ', '_').lower()}.png"
                    full_path = os.path.join(brain_dir, file_name)
                    brain.save_image(full_path)
                    self.logger.debug(f"Brain plot saved to {full_path}")

                brain.close()
                
        except Exception as e:
            self.logger.error(f"Error in 3D visualization: {e}")
            
        finally:
            try:
                mne.viz.set_3d_backend(original_backend)
            except:
                pass

    def plot_source_and_sensors(
        self,
        x_one_trial: np.ndarray,
        x_active_indices: np.ndarray,
        y_clean_one_trial: np.ndarray,
        y_noisy_one_trial: np.ndarray,
        nnz: int,
        ERP_config: dict = None,
        source_units: str = FIFF.FIFF_UNIT_AM,
        sensor_units: str = FIFF.FIFF_UNIT_V,
        orientation_type: str = "fixed",
        max_sensors: int = 3,
        plot_sensors_together: bool = False,
        shift: float = 20.0,
        figsize: Tuple[float, float] = (14, 10),
        file_name: str = 'data_simulation.png',
        save_path: Optional[str] = 'results/figures/data_sim.png',
        show: bool = False
    ) -> None:
        """
        Visualize source activity and sensor measurements.

        Plots active source time courses and compares clean vs. noisy sensor signals.
        Includes a line indicating stimulus onset.
        Uses self.tmin and self.tmax for the time axis.

        Parameters
        ----------
        x : np.ndarray
            Source activity. Shape depends on orientation type.
        y_clean : np.ndarray
            Clean sensor measurements (n_sensors, n_times).
        y_noisy : np.ndarray
            Noisy sensor measurements (n_sensors, n_times).
        active_sources : Optional[np.ndarray], optional
            Indices of non-zero (active) sources. If None, they are determined from x, by default None.
        nnz_to_plot : int, optional
            Number of non-zero sources to plot. If -1, plot all non-zero sources found, by default -1.
        sfreq : float, optional
            Sampling frequency in Hz, by default self.sfreq.
        max_sensors : int, optional
            Maximum number of sensors to plot, by default 3.
        plot_sensors_together : bool, optional
            If True, plot all sensors on the same subplot. If False, stack plots vertically, by default False.
        shift : float, optional
            Vertical shift between sensors when plotting together, by default 20.0.
        figsize : Tuple[float, float], optional
            Figure size for the plot, by default (14, 10).
        save_path : Optional[str], optional
            Path to save the figure. If None, the figure is not saved, by default 'results/figures/data_sim.png'.
        show : bool, optional
            If True, display the plot, by default False.
        """
        # Use self.sfreq if the passed sfreq is the default placeholder, otherwise use passed sfreq
        
        if sensor_units == FIFF.FIFF_UNIT_T:
            y_clean_scaled = y_clean_one_trial * 1e15  # Convert Tesla to femtoTesla (fT)
            y_noisy_scaled = y_noisy_one_trial * 1e15  # Convert Tesla to femtoTesla (fT)
            sensor_units = "fT"
        elif sensor_units == FIFF.FIFF_UNIT_T_M:
            y_clean_scaled = y_clean_one_trial * 1e15  # Convert T/m to fT/m
            y_noisy_scaled = y_noisy_one_trial * 1e15  # Convert T/m to fT/m
            sensor_units = "fT/m"
        elif sensor_units == FIFF.FIFF_UNIT_V:
            y_clean_scaled = y_clean_one_trial * 1e6  # Convert V to μV
            y_noisy_scaled = y_noisy_one_trial * 1e6  # Convert V to μV
            sensor_units = "μV"
        else:
            raise ValueError(f"Unsupported units for sensor signals: {sensor_units}. Expected FIFF.FIFF_UNIT_T, FIFF.FIFF_UNIT_T_M, or FIFF.FIFF_UNIT_V.")

        if source_units == FIFF.FIFF_UNIT_AM:
            x_scaled = x_one_trial * 1e9  # Convert A/m to nAm
            source_units = "nAm"
        else:
            raise ValueError(f"Unsupported units for source activity: {source_units}. Expected FIFF.FIFF_UNIT_AM.")

        n_times_from_data = y_clean_scaled.shape[1]
        tmin, tmax, stim_onset, _, times = self._get_plot_params(ERP_config, x_scaled.shape[-1])

        y_min = min(y_clean_scaled.min(), y_noisy_scaled.min())
        y_max = max(y_clean_scaled.max(), y_noisy_scaled.max())
        y_range = y_max - y_min if y_max > y_min else 1.0

        num_sensors_to_plot = min(max_sensors, y_clean_scaled.shape[0])
        total_plots = 1 + (1 if plot_sensors_together else num_sensors_to_plot)
        fig, axes = plt.subplots(
            total_plots,
            1,
            figsize=figsize,
            gridspec_kw={"height_ratios": [1] * total_plots}, 
            sharex=True
        )
        if total_plots == 1:
            axes = [axes]

        ax_sources = axes[0]
        if orientation_type == "fixed":
            for i in x_active_indices:
                ax_sources.plot(times, x_scaled[i].T, label=f"Source {i}")
        elif orientation_type == "free":
            for i in x_active_indices:
                source_norm = np.linalg.norm(x_scaled[i], axis=0)
                ax_sources.plot(times, source_norm, label=f"Source {i} (Norm)")

        ax_sources.axvline(stim_onset, color='k', linestyle='--', linewidth=1, label='Stimulus Onset')
        ax_sources.set_title(f"{nnz} Active Simulated Source Activity")
        ax_sources.set_ylabel(f"Amplitude ({source_units})")
        ax_sources.grid(True)
        ax_sources.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        sensor_axes = axes[1:]
        if plot_sensors_together:
            ax_sensors = sensor_axes[0]
            current_shift = 0
            for i in range(num_sensors_to_plot):
                ax_sensors.plot(times, y_clean_scaled[i] + current_shift, label=f"Clean (Sensor {i})", linewidth=1.5)
                ax_sensors.plot(times, y_noisy_scaled[i] + current_shift, label=f"Noisy (Sensor {i})", linewidth=1.5)
                current_shift += shift
            ax_sensors.axvline(stim_onset, color='k', linestyle='--', linewidth=1, label='Stimulus Onset')
            ax_sensors.set_title("Sensor Measurements")
            ax_sensors.set_ylabel(f"Amplitude ({sensor_units})") 
            ax_sensors.grid(True)
            # Consolidate legend for "Stimulus Onset" if it's plotted multiple times
            handles, labels = ax_sensors.get_legend_handles_labels()
            by_label = dict(zip(labels, handles)) # Remove duplicate labels
            ax_sensors.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            for idx, ax_sens in enumerate(sensor_axes):
                ax_sens.plot(times, y_clean_scaled[idx], label=f"Clean", linewidth=1.5)
                ax_sens.plot(times, y_noisy_scaled[idx], label=f"Noisy", linewidth=1)
                ax_sens.axvline(stim_onset, color='k', linestyle='--', linewidth=1, label='Stimulus Onset')
                ax_sens.set_title(f"Sensor {idx}")
                ax_sens.set_ylabel(f"Amplitude ({sensor_units})")
                ax_sens.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range) 
                ax_sens.grid(True)
                handles, labels = ax_sens.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax_sens.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))

        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        self._handle_figure_output(fig=fig, file_name=file_name, save_path=save_path, show=show)

    # -------------------------------------------
    # --- Visualization for Uncertainty
    # -------------------------------------------

    def plot_active_sources(
        self,
        x_one_trial_one_time: np.ndarray,
        x_hat_one_trial_one_time: np.ndarray,
        x_active_indices: np.ndarray,
        x_hat_active_indices: np.ndarray,
        n_sources: int,
        source_units: str = FIFF.FIFF_UNIT_AM,
        orientation_type: str = "fixed",
        save_path: Optional[str] = None,
        file_name: Optional[str] = None,
        title: Optional[str] = None,
        show: bool = True
    ):
        """Plot the active sources at a specific time step, or averaged across time, comparing ground truth and estimated values.

        Parameters
        ----------
        x : np.ndarray
            Ground truth values for components specified by active_indices.
        x_hat : np.ndarray
            Estimated values for components specified by active_indices.
        x_active_indices : np.ndarray
            Indices of active sources in the ground truth.
        x_hat_active_indices : np.ndarray
            Indices of active sources in the estimated values.
        n_sources : int
            Total number of sources.
        source_units : str, optional
            Units for the source signals, by default FIFF.FIFF_UNIT_AM
        orientation_type : str, optional
            Orientation type for the plot, by default "fixed"
        save_path : Optional[str], optional
            Path to save the plot, by default None
        file_name : Optional[str], optional
            Name of the file to save the plot, by default None
        title : Optional[str], optional
            Title of the plot, by default None
        show : bool, optional
            Whether to show the plot, by default True
        """        
        # convert the data from A to nAm for better readability
        if source_units == FIFF.FIFF_UNIT_AM:
            x_active_scaled = x_one_trial_one_time[x_active_indices] * 1e9
            x_hat_active_scaled = x_hat_one_trial_one_time[x_hat_active_indices] * 1e9
            source_units = "nAm"
        else:
            raise ValueError(f"Unsupported units for source signals: {source_units}. Expected FIFF.FIFF_UNIT_AM.")

        if orientation_type == 'fixed':
            plt.figure(figsize=(12, 6))

            plt.scatter(x_hat_active_indices, x_hat_active_scaled, color='blue', marker='o', alpha=0.6, label=f'Non-Zero Posterior Mean - Estimated active ({len(x_hat_active_indices)} sources)')


            plt.scatter(x_active_indices, x_active_scaled, color='red', marker='x', label=f'Non-Zero Ground Truth ({len(x_active_indices)} simulated Sources)')

            plt.xlabel('Active voxels')
            plt.ylabel(f'Amplitude of averaged sources (across time) and their estimates ({source_units})')
            plt.title(title or f'Active Sources fixed orientation, (Only Non-Zero Sources) of Averaged Activities across Time Steps')
            plt.legend(title=f'Total Sources: {n_sources}', loc='best')
            plt.grid(True, alpha=0.5)
            plt.tight_layout(rect=[0, 0.05, 1, 0.96])
            fig = plt.gcf()

        else:
            fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
            orientations = ['X', 'Y', 'Z']

            x_active_indices_flat = x_active_indices // 3
            x_active_indices_orientations_flat = x_active_indices % 3
            # Create a map from original source index to its value for each orientation
            x_active_indices_map = [{} for _ in range(3)]
            for idx, val in enumerate(x):
                if val != 0: # Only consider non-zero ground truth
                    orient = x_active_indices_orientations_flat[idx]
                    src_idx = x_active_indices_flat[idx]
                    x_active_indices_map[orient][src_idx] = val
            
            # For Estimated (x_hat)
            x_hat_active_indices_flat = x_hat_active_indices // 3
            x_hat_active_indices_orientations_flat = x_hat_active_indices % 3
            x_hat_active_indices_map = [{} for _ in range(3)]
            for idx, val in enumerate(x_hat):
                orient = x_hat_active_indices_orientations_flat[idx]
                src_idx = x_hat_active_indices_flat[idx]
                x_hat_active_indices_map[orient][src_idx] = val


            for i, ax in enumerate(axes): # i is the target orientation index (0, 1, 2)
                if not x_active_indices and not x_hat_active_indices:
                    ax.set_title(f'Orientation {orientations[i]} (No active components to plot)')
                    ax.grid(True, alpha=0.5)
                    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
                    continue
                
                ax.scatter(x_hat_active_indices, x_hat_one_trial_one_time, color='blue', marker='o', alpha=0.6, label=f'Non-Zero Posterior Mean - Estimated active ({len(x_hat_active_indices)} sources)')

                ax.scatter(x_active_indices, x_active_indices, color='red', marker='x', label=f'Non-Zero Ground Truth ({len(x_active_indices)} simulated Sources)')
                
                ax.set_xlabel('Index of Active (Non-zero) Sources')
                ax.set_ylabel(f'Amplitude of averaged sources (across time) and their estimates ({source_units})')
                ax.set_title(f'Active Sources Comparison for free orientation, (Only Non-Zero Sources) of Averaged Activities across Time Steps')

                # all_unique_src_indices_on_axis = sorted(list(set(x_active_indices + active_indices)))
                all_unique_src_indices_on_axis = np.arange(n_sources)
                # n_sources_this_axis = len(all_unique_src_indices_on_axis)
                ax.legend(title=f'Total Sources: {n_sources}', loc='best')

                ax.grid(True, alpha=0.5)
                # ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
                if all_unique_src_indices_on_axis:
                    ax.set_xticks(all_unique_src_indices_on_axis)
                    ax.set_xticklabels([str(s_idx) for s_idx in all_unique_src_indices_on_axis])

            fig.text(0.5, 0.04, 'Original Source Index', ha='center', va='center')
            plt.tight_layout(rect=[0, 0.05, 1, 0.96]) 
            fig.suptitle(f"Active Sources Comparison for free orientation, (Only Non-Zero Sources) of Averaged Activities across Time Steps", fontsize=16)
            
        self._handle_figure_output(fig, file_name or f"active_sources", save_path, show)
     
    def plot_ci(
        self,
        x_one_trial_one_time: np.array,
        x_hat_one_trial_one_time: np.array,
        x_active_indices: np.array,
        x_hat_active_indices: np.array,
        n_sources: int,
        source_units: str,
        ci_lower: np.array,
        ci_upper: np.array,
        confidence_levels: list,
        orientation_type: str = "fixed",
        sharey: bool = True,
        file_name: str = "active_sources_ci",
        save_path: str = None,
        show: bool = True,
        figsize: tuple = (12, 6),
    ):
        if source_units == FIFF.FIFF_UNIT_AM: #TODO: check whether we need to copy()
            x_scaled = x_one_trial_one_time.copy().flatten() * 1e9
            x_hat_scaled = x_hat_one_trial_one_time.copy().flatten() * 1e9
            ci_lower_scaled = ci_lower.copy() * 1e9
            ci_upper_scaled = ci_upper.copy() * 1e9
            source_units = "nAm"        
        else:
            raise ValueError(f"Unsupported units for source signals: {source_units}. Expected FIFF.FIFF_UNIT_AM.")            
    
        # Create grid of subplots for all confidence levels
        n_levels = len(confidence_levels)
        n_cols = min(4, n_levels)
        n_rows = int(np.ceil(n_levels / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1]), squeeze=False, sharex=True, sharey=sharey)
        axes = axes.flatten()
        
        matched_indices = x_hat_active_indices[np.isin(x_hat_active_indices, x_active_indices)]
        
        for idx, confidence_level_val in enumerate(confidence_levels):
            ax = axes[idx]
            
            if len(matched_indices) != 0:
                ci_lower_current = ci_lower_scaled[idx].flatten()
                ci_upper_current = ci_upper_scaled[idx].flatten()

                yerr_lower = np.abs(
                    x_hat_scaled[matched_indices] - ci_lower_current
                )
                yerr_upper = np.abs(
                    ci_upper_current - x_hat_scaled[matched_indices]
                )
                yerr = np.stack([yerr_lower, yerr_upper])
                     
                ax.errorbar(
                    matched_indices,
                    x_hat_scaled[matched_indices],
                    yerr,
                    fmt='o',
                    color='blue',
                    # alpha=0.7,
                    capsize=5,
                    label=(
                        f'Active posterior mean ({len(x_hat_active_indices)}/{n_sources})'
                        f'\nMatched voxel locations: {len(matched_indices)}'
                    ),
                    zorder=1
                )
            else:
                # Add a dummy handle for the error bar to the legend
                errorbar_handle = mlines.Line2D(
                    [], [], color='blue', marker='o', linestyle='None', markersize=8,
                    label=(
                        f'Active posterior mean ({len(x_hat_active_indices)}/{n_sources})'
                        f'\nMatched indices: {len(matched_indices)}'
                    )
                )

            ax.scatter(x_active_indices, x_scaled[x_active_indices], marker='x', s=70, color='red', label=f'Active simulated sources ({len(x_active_indices)}/{n_sources})', zorder=2)

            ax.set_title(f'CI Level={confidence_level_val:.2f}')
            ax.axhline(0, color='grey', lw=0.8, ls='--')
            ax.grid(True, alpha=0.5)

        # Shared legend: collect all handles/labels
        handles, labels = [], []
        for ax in axes[:n_levels]:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        by_label = dict(zip(labels, handles))

        # Hide unused subplots
        for ax in axes[n_levels:]:
            ax.axis('off')
                    
        # Place the legend below the supertitle, centered
        fig.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='large', frameon=True, bbox_to_anchor=(0.6, 1.015))

        # Place the legend in the empty subplot
        # axes[11].legend(by_label.values(), by_label.keys(), loc='center', fontsize='large', frameon=True)
        # axes[11].set_title("Legend", fontsize=16)

        # Shared x/y labels for the whole figure
        fig.supxlabel('Active voxels', fontsize=14)
        fig.supylabel(f'Amplitude ({source_units})', fontsize=14)
        fig.suptitle('Confidence Intervals for Active Reconstructed Sources', fontsize=18, y=1.05)
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])        
        
        self._handle_figure_output(fig, file_name, save_path, show)
                          
    def plot_calibration_curve(
        self,
        confidence_levels,
        empirical_coverage,
        result=None, # This dictionary is expected to contain the metrics
        which_legend="active_indices", # or "all_sources"
        file_name='calibration_curve',
        save_path=None,
        show=True,
    ):
        """
        Visualizes the calibration curve.

        Parameters:
        - empirical_coverage (np.ndarray): 1D array of empirical coverage values,
                                            corresponding to each confidence level in self.confidence_levels.
        - results (dict): Dictionary possibly containing calibration metrics.
        - which_legend (str): Specifies which set of metrics to display in the legend.
        - file_name (str): Base name for the saved plot file.
        """            
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot the empirical coverage line and scatter points
        ax.plot(confidence_levels, empirical_coverage, label="Empirical Coverage", marker='o', linestyle='-')
        ax.scatter(confidence_levels, empirical_coverage, color='blue', s=50, zorder=5)

        # Plot the ideal calibration line (diagonal)
        ax.plot(confidence_levels, confidence_levels, '--', label="Ideal Calibration", color='gray')

        # Fill the area between empirical and ideal calibration
        ax.fill_between(
            confidence_levels, 
            empirical_coverage, 
            confidence_levels, 
            color='orange', 
            alpha=0.3, 
            label="AUC Deviation Area"
        )
        
        ax.set_xlabel("Nominal Confidence Level")
        ax.set_ylabel("Empirical Coverage")
        ax.set_title(file_name.replace('_', ' ').title())
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.set_aspect('equal', adjustable='box')

        # Prepare legend: start with existing plot elements
        handles, labels = ax.get_legend_handles_labels()
        
        # Determine which set of metrics to display
        if which_legend == "active_indices":
            metrics_to_display = {
                'auc_deviation_active_indices': 'AUC area',
                'max_positive_deviation_active_indices': 'Max Positive Dev.',
                'max_negative_deviation_active_indices': 'Max Negative Dev.',
                'max_absolute_deviation_active_indices': 'Max Abs Dev.',
            }
        elif which_legend == "all_sources":
            metrics_to_display = {
                'auc_deviation_all_sources': 'AUC area',
                'max_positive_deviation_all_sources': 'Max Positive Dev.',
                'max_negative_deviation_all_sources': 'Max Negative Dev.',
                'max_absolute_deviation_all_sources': 'Max Abs Dev.',
            }
        else:
            self.logger.error(f"Unknown which_legend value: {which_legend}. Expected 'active_indices' or 'all_sources'.")
            return

        if result:
            separator_handle = mlines.Line2D([], [], color='none', marker='', linestyle='None', label="---------------------------")
            handles.append(separator_handle)
            labels.append(separator_handle.get_label())

            for key, display_name in metrics_to_display.items():
                if key in result and result[key] is not None:
                    value = result[key]
                    dummy_handle = mlines.Line2D([], [], color='none', marker='', linestyle='None', label=f"{display_name}: {value:.3f}")
                    handles.append(dummy_handle)
                    labels.append(f"{display_name}: {value:.3f}")

        # Create the legend with potentially added metric values
        ax.legend(handles, labels, loc='best', fontsize='small')
        fig.tight_layout(rect=[0.05, 0.05, 1, 0.96]) 

        self._handle_figure_output(fig, file_name, save_path, show)


    # ------------------------------------------------
    # Wrap up all visualizations into a single method
    # ------------------------------------------------

    def plot_all(
        self,
        x_trials: np.ndarray,
        x_active_indices_trials: np.ndarray,
        x_hat_one_trial: np.ndarray,        
        x_hat_active_indices_one_trial: np.ndarray,
        y_clean_trials: np.ndarray,
        y_noisy_trials: np.ndarray,
        trial_idx: int = 0,
        n_sources: int = 1,
        subject: str = None,
        subjects_dir: str = None,
        fwd_path: str = None,
        nnz: int = 1,
        ERP_config: dict = None,
        sample_idx: int = 0,
        source_units: str = FIFF.FIFF_UNIT_AM,
        sensor_units: str = FIFF.FIFF_UNIT_V,
        confidence_levels: np.ndarray = None,
        empirical_coverages: dict = None,
        ci_lower: np.ndarray = None,
        ci_upper: np.ndarray = None,
        orientation_type: str = "fixed",
        result: dict = None,
        experiment_dir: str = None,
    ):
        """
        Comprehensive visualization wrapper that generates all plots for experiment analysis.
        
        This is a high-level wrapper function that orchestrates multiple visualization 
        methods to create a complete set of plots for ERP source localization analysis.
        
        **Generated Visualizations:**
        - Source signal plots (individual and all trials)
        - Sensor signal plots (stacked and concatenated)
        - 3D brain surface plots (if available)
        - Source-sensor comparison plots
        - Active source analysis plots
        - Confidence interval plots
        - Calibration curve plots
        
        **Called Methods:**
        - :meth:`plot_source_signals` - Source time series visualization
        - :meth:`plot_sensor_signals` - Sensor measurements visualization  
        - :meth:`plot_stc_3d_brain` - 3D brain surface visualization
        - :meth:`plot_source_and_sensors` - Combined source-sensor plots
        - :meth:`plot_active_sources` - Active source comparison
        - :meth:`plot_ci` - Confidence interval visualization
        - :meth:`plot_calibration_curve` - Uncertainty calibration analysis
        
        Parameters
        ----------
        x_trials : np.ndarray
            Source activity trials, shape (n_trials, n_sources, n_times)
        x_active_indices_trials : np.ndarray
            Active source indices per trial
        x_hat_one_trial : np.ndarray
            Estimated source activity for one trial
        x_hat_active_indices_one_trial : np.ndarray
            Estimated active source indices for one trial
        y_clean_trials : np.ndarray
            Clean sensor measurements, shape (n_trials, n_sensors, n_times)
        y_noisy_trials : np.ndarray
            Noisy sensor measurements, shape (n_trials, n_sensors, n_times)
        trial_idx : int, optional
            Trial index for single-trial visualizations, by default 0
        n_sources : int, optional
            Total number of sources, by default 1
        subject : str, optional
            Subject name for brain plots
        subjects_dir : str, optional
            FreeSurfer subjects directory
        fwd_path : str, optional
            Forward model path
        nnz : int, optional
            Number of non-zero sources, by default 1
        ERP_config : dict, optional
            ERP configuration parameters
        sample_idx : int, optional
            Time sample index for brain plots, by default 0
        source_units : str, optional
            Source signal units, by default FIFF.FIFF_UNIT_AM
        sensor_units : str, optional
            Sensor signal units, by default FIFF.FIFF_UNIT_V
        confidence_levels : np.ndarray, optional
            Confidence levels for uncertainty analysis
        empirical_coverages : dict, optional
            Empirical coverage data for calibration
        ci_lower : np.ndarray, optional
            Lower confidence bounds
        ci_upper : np.ndarray, optional
            Upper confidence bounds
        orientation_type : str, optional
            Source orientation type, by default "fixed"
        result : dict, optional
            Analysis results containing metrics
        experiment_dir : str, optional
            Experiment directory for saving plots
            
        Notes
        -----
        This wrapper function automatically saves all plots to organized subdirectories:
        - `data_simulation/` - Data visualization plots
        - `uncertainty_analysis/` - Uncertainty quantification plots
        
        All plots are saved with `show=False` for batch processing compatibility.
        
        Examples
        --------
        >>> viz = Visualizer(base_save_path="results/figures")
        >>> viz.plot_all(
        ...     x_trials=x_trials,
        ...     x_active_indices_trials=active_indices,
        ...     x_hat_one_trial=x_hat,
        ...     # ... other parameters
        ... )
        """
        x_one_trial = x_trials[trial_idx]
        x_active_indices_one_trial = x_active_indices_trials[trial_idx]
        x_one_trial_avg_time = np.mean(x_one_trial, axis=1, keepdims=True)
        x_hat_one_trial_avg_time = np.mean(x_hat_one_trial, axis=1, keepdims=True)
        y_clean_one_trial = y_clean_trials[trial_idx]
        y_noisy_one_trial = y_noisy_trials[trial_idx]
        
        # =========================
        # 1. Plot simulated data
        # =========================
        
        # Plot sources (all trials)
        self.plot_source_signals(
            ERP_config=ERP_config,
            x_trial=x_trials,
            x_active_indices=x_active_indices_trials,
            units=source_units,
            trial_idx=None,
            title="Source Trials (All)",
            save_dir="data_simulation",
            file_name="source_trials_all",
            show=False,
        )

        # Plot sensors (all trials) with selected channels: y_noisy
        self.plot_sensor_signals(
            ERP_config=ERP_config,
            y_trials=y_noisy_trials,
            channels="all",  # or "all"
            units=sensor_units,
            mode="stack",
            title="Sensor Signals (All Trials stacked)",
            save_dir="data_simulation",
            file_name="sensor_stack_trials_noisy",
            show=False
        )

        # Concatenated trials: y_noisy
        self.plot_sensor_signals(
            ERP_config=ERP_config,
            y_trials=y_noisy_trials,
            # channels=[0, 10],  # or "all"
            units=sensor_units,
            mode="concatenate",
            title="Sensor Signals (All trials concatenated)",
            save_dir="data_simulation",
            file_name="sensor_concatenate_trials_noisy",
            show=False
        )

        # Concatenated trials: y_clean
        self.plot_sensor_signals(
            ERP_config=ERP_config,
            y_trials=y_clean_trials,
            # channels=[0, 10],  # or "all"
            units=sensor_units,
            mode="concatenate",
            title="Sensor Signals (All trials concatenated)",
            save_dir="data_simulation",
            file_name="sensor_concatenate_trials_clean",
            show=False
        )

        self.plot_stc_3d_brain(
            ERP_config=ERP_config,
            x_one_trial=x_one_trial,
            x_hat_one_trial=x_hat_one_trial,
            orientations=['lateral', 'medial', 'dorsal', 'ventral'],
            source_units=source_units,
            sample_idx=sample_idx,
            subject=subject,
            subjects_dir=subjects_dir,
            fwd_path=fwd_path,
            save_dir=experiment_dir,
            file_name="brain",
            show=False,
        )

        self.plot_source_and_sensors(
            x_one_trial=x_one_trial,
            x_active_indices=x_active_indices_one_trial,
            y_clean_one_trial=y_clean_one_trial,
            y_noisy_one_trial=y_noisy_one_trial,
            nnz=nnz,
            ERP_config=ERP_config,
            source_units=source_units,
            sensor_units=sensor_units,
            orientation_type=orientation_type,
            max_sensors=3,
            plot_sensors_together=False,
            file_name='source_sensor_data.png',
            save_path='data_simulation',
            show=False
        )        
        
        # =========================
        # 2. Plot uncertainty analysis figures
        # =========================
        # Plot active sources
        self.plot_active_sources(
            x_one_trial_one_time=x_one_trial_avg_time,
            x_hat_one_trial_one_time=x_hat_one_trial_avg_time,
            x_active_indices=x_active_indices_one_trial,
            x_hat_active_indices=x_hat_active_indices_one_trial,
            n_sources=n_sources,
            source_units=source_units,
            orientation_type= orientation_type,
            save_path="uncertainty_analysis",
            file_name="active_sources",
            show=False
        )
        # Plot confidence intervals - unshared y-axis
        self.plot_ci(
            x_one_trial_one_time=x_one_trial_avg_time,
            x_hat_one_trial_one_time=x_hat_one_trial_avg_time,
            x_active_indices=x_active_indices_one_trial,
            x_hat_active_indices=x_hat_active_indices_one_trial,
            n_sources=n_sources,
            source_units=source_units,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_levels=confidence_levels,
            orientation_type=orientation_type,
            sharey=False,
            save_path="uncertainty_analysis",
            file_name="confidence_intervals",
            show=False,
            figsize=(18, 13)
        )

        # Plot confidence intervals - shared y-axis
        self.plot_ci(
            x_one_trial_one_time=x_one_trial_avg_time,
            x_hat_one_trial_one_time=x_hat_one_trial_avg_time,
            x_active_indices=x_active_indices_one_trial,
            x_hat_active_indices=x_hat_active_indices_one_trial,
            n_sources=n_sources,
            source_units=source_units,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_levels=confidence_levels,
            orientation_type=orientation_type,
            sharey=True,
            save_path="uncertainty_analysis",
            file_name="confidence_intervals_shared-Yaxis",
            show=False,
            figsize=(18, 13)
        )
        
        # plot calibration curve - active sources
        self.plot_calibration_curve(
            confidence_levels=confidence_levels,
            empirical_coverage=empirical_coverages['active_indices'],
            result=result,
            which_legend="active_indices", # or "all_sources"
            file_name='calibration_curve_active_sources',
            save_path='uncertainty_analysis',
            show=True,
        )



def main():
    from calibrain import SourceSimulator
    from calibrain import SensorSimulator

    logging.basicConfig(
        level=logging.INFO,  # or DEBUG
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),  # Console
            logging.FileHandler("Vizualisation.log", mode="w")  # Log to file
        ]
    )
    logger = logging.getLogger("SourceSimulator")

    ERP_config = {
        "tmin": -0.5,
        "tmax": 0.5,
        "stim_onset": 0.0,
        "sfreq": 250,
        "fmin": 1,
        "fmax": 5,
        "amplitude": 10.0, # 30.0
        "random_erp_timing": True,
        "erp_min_length": None,
    }

    n_trials=4
    orientation_type="fixed"
    n_sources=10
    nnz=5
    global_seed=42

    source_simulator = SourceSimulator(
        ERP_config=ERP_config,
        logger=logger
    )
    print(f"Default units for source signals: {source_simulator.source_units}")

    x_trials, active_indices_trials = source_simulator.simulate(
        orientation_type=orientation_type,
        n_sources=n_sources,
        nnz=nnz,
        n_trials=n_trials,
        global_seed=global_seed,
    )
    # source_simulator.source_units = "Am"  # Set units for source signals
    trial_idx = 0

    viz = Visualizer(base_save_path="testViz", logger=logger)

    # Plot sources (single trial)
    viz.plot_source_signals(
        ERP_config=ERP_config,
        x=x_trials,
        active_indices=active_indices_trials,
        units=source_simulator.source_units,
        trial_idx=trial_idx,
        title=f"Source Activity - Trial {trial_idx+1}",
        save_dir="data_simulation",
        file_name=f"source_trial_{trial_idx+1}",
        show=False,
    )

    # Plot sources (all trials)
    viz.plot_source_signals(
        ERP_config=ERP_config,
        x=x_trials,
        active_indices=active_indices_trials,
        units=source_simulator.source_units,
        trial_idx=None,
        title="Source Activity - All Trials",
        save_dir="data_simulation",
        file_name="source_trials_all",
        show=False,
    )


    sensor_simulator = SensorSimulator(
        logger=logger,
    )
    print(f"Default units for sensor signals: {sensor_simulator.sensor_units}")

    n_sensors = 10
    L = np.random.randn(n_sensors, n_sources)

    # Simulate sensor data
    y_clean, y_noisy, noise, noise_var = sensor_simulator.simulate(
        x_trials,
        L,
        orientation_type="fixed",
        alpha_SNR=0.5,
        n_trials=n_trials,
    )
    # sensor_simulator.sensor_units = "T"


    # Plot sensors (single trial) with selected channels: y_clean
    viz.plot_sensor_signals(
        ERP_config=ERP_config,
        y_trials=y_clean,
        trial_idx=trial_idx,
        # channels=[0, 1],  # or "all"
        units=sensor_simulator.sensor_units,
        mode="single",
        title=f"Sensor Trial {trial_idx+1}",
        save_dir="data_simulation",
        file_name=f"sensor_trial_{trial_idx+1}_clean",
        show=True
    )

    # Plot sensors (all trials) with selected channels: y_clean
    viz.plot_sensor_signals(
        ERP_config=ERP_config,
        y_trials=y_clean,
        mode="stack",
        channels=[2, 3],
        units=sensor_simulator.sensor_units,
        save_dir="data_simulation",
        title="Sensor Signals (All Trials)",
        file_name="sensor_all_trials_clean",
        show=False
    )

    # Concatenated trials: y_clean
    viz.plot_sensor_signals(
        ERP_config=ERP_config,
        y_trials=y_clean,
        mode="concatenate",
        channels=[0, 2, 3],  # or "all"
        units=sensor_simulator.sensor_units,
        title="Sensor Signals (Concatenated)",
        save_dir="data_simulation",
        file_name="sensor_concatenated_clean",
        show=False
    )

    # Plot sensors (single trial) with selected channels: y_noisy
    viz.plot_sensor_signals(
        ERP_config=ERP_config,
        y_trials= y_noisy,
        trial_idx=trial_idx,
        channels=[0, 1],  # or "all"
        units=sensor_simulator.sensor_units,
        mode="single",
        title=f"Sensor Trial {trial_idx+1}",
        save_dir="data_simulation",
        file_name=f"sensor_trial_{trial_idx+1}_noisy",
        show=False
    )

    # Plot sensors (all trials) with selected channels: y_noisy
    viz.plot_sensor_signals(
        ERP_config=ERP_config,
        y_trials=y_noisy,
        mode="stack",
        channels="all",  # or "all"
        units=sensor_simulator.sensor_units,
        title="Sensor Signals (All Trials)",
        save_dir="data_simulation",
        file_name="sensor_all_trials_noisy",
        show=False
    )

    # Concatenated trials: y_noisy
    viz.plot_sensor_signals(
        ERP_config=ERP_config,
        y_trials=y_noisy,
        mode="concatenate",
        channels=[0, 2],  # or "all"
        units=sensor_simulator.sensor_units,
        title="Sensor Signals (Concatenated)",
        save_dir="data_simulation",
        file_name="sensor_concatenated_noisy",
        show=False
    )


if __name__ == "__main__":
    main()



# ------------------------------------------------------------------

# visualize_leadfield_sensor_boxplot(
#     L,
#     orientation_type=self.orientation_type, 
#     sensor_indices_to_plot=list(range(self.n_sensors)), 
#     max_sensors_to_plot=20,
#     save_path=os.path.join(save_path, "leadfield_sensor_boxplot.png"),
#     show=False
# )

# visualize_leadfield_distribution(
#     L,
#     orientation_type=self.orientation_type,
#     bins=100,
#     save_path=os.path.join(save_path, "leadfield_distribution.png"),
#     show=False
# )

# visualize_leadfield(
#      L,
#      orientation_type=self.orientation_type,
#      save_path=os.path.join(save_path, "leadfield_matrix.png"),
#      show=False
# )

# visualize_leadfield_summary(
#     L,
#     orientation_type=self.orientation_type,
#     bins=100,
#     sensor_indices_to_plot=list(range(self.n_sensors)),
#     # max_sensors_to_plot=10, # Let the function select if sensor_indices_to_plot is None
#     save_path=os.path.join(save_path, "leadfield_summary.png"),
#     show=False
# )

# visualize_leadfield_topomap(
#     leadfield_matrix=L,
#     x=x_all_trials[first_trial_idx],
#     orientation_type=self.orientation_type,
#     title="Leadfield Topomap for Some Active (Nonzero) Sources",
#     save_path=os.path.join(save_path, "leadfield_topomap.png"),
#     show=False,
# )

# print(f"\nPlotting results for trial {first_trial_idx + 1}...")

# time_vector = np.arange(self.tmin, self.tmax, 1.0 / self.sfreq)

# # Now plot_sensor_signals uses the clean and noisy data generated separately
# plot_sensor_signals( 
#     y_clean=y_clean_all_trials[first_trial_idx], # Use stored clean data
#     y_noisy=y_noisy_all_trials[first_trial_idx],       # Use stored noisy data
#     sensor_indices=sensor_subplots_indices,
#     times=time_vector,
#     save_dir=save_path,
#     figure_name=f"specific_sensor_signals_subplots_trial{first_trial_idx}",
#     trial_idx=first_trial_idx
# )

# plot_active_sources(
#     x=x_all_trials[first_trial_idx],
#     times=time_vector,
#     active_indices=active_indices_all_trials[first_trial_idx],
#     stim_onset=self.stim_onset,
#     nnz=self.nnz,
#     save_dir=save_path,
#     figure_name=f"active_sources_subplots_trial{first_trial_idx}",
#     trial_idx=first_trial_idx
# )


# plot_sorted_posterior_variances(top_k=10)
# visualize_sorted_covariances(top_k=10)
# plot_posterior_covariance_matrix()

# ======================================================================================

# def visualize_leadfield_summary(
#     self,
#     leadfield_matrix: np.ndarray,
#     orientation_type: str = "fixed",
#     bins: int = 100,
#     sensor_indices_to_plot: Optional[List[int]] = None,
#     max_sensors_to_plot: int = 10,
#     main_title: Optional[str] = None,
#     save_path: Optional[str] = None,
#     show: bool = False
# ) -> None:
#     # ... (initial parameter validation and actual_sensor_indices_to_plot logic remains the same) ...
#     if leadfield_matrix is None or not isinstance(leadfield_matrix, np.ndarray) or leadfield_matrix.size == 0:
#         self.logger.error("Invalid leadfield matrix provided for summary visualization.")
#         return

#     fig = None
#     try:
#         num_total_sensors_in_lf = leadfield_matrix.shape[0]
#         actual_sensor_indices_to_plot: np.ndarray

#         if sensor_indices_to_plot is None:
#             if num_total_sensors_in_lf > max_sensors_to_plot:
#                 actual_sensor_indices_to_plot = np.linspace(0, num_total_sensors_in_lf - 1, max_sensors_to_plot, dtype=int)
#             else:
#                 actual_sensor_indices_to_plot = np.arange(num_total_sensors_in_lf)
#         else:
#             actual_sensor_indices_to_plot = np.array(sensor_indices_to_plot, dtype=int)
#             if np.any(actual_sensor_indices_to_plot < 0) or np.any(actual_sensor_indices_to_plot >= num_total_sensors_in_lf):
#                 self.logger.error("Summary Plot: Invalid sensor_indices_to_plot: indices out of bounds.")
#                 if num_total_sensors_in_lf > 0 :
#                     actual_sensor_indices_to_plot = np.arange(min(num_total_sensors_in_lf, max_sensors_to_plot))
#                     self.logger.warning(f"Defaulting to plotting first {len(actual_sensor_indices_to_plot)} sensors for heatmap/boxplot.")
#                 else:
#                     actual_sensor_indices_to_plot = np.array([])

#         fig = plt.figure(figsize=(16, 18)) # Adjust figsize as needed

#         # Main GridSpec: 2 rows, 1 column. Each row will be further divided.
#         gs_rows = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1]) # Adjust height_ratios if needed

#         # --- Top Row: Heatmap and its Colorbar ---
#         # To make the heatmap image wider, increase the first ratio (e.g., 0.95)
#         # and decrease the second (e.g., 0.03), ensuring they make sense for the space.
#         gs_top_row = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_rows[0],
#                                                         width_ratios=[0.50, 0.03], # Example: Heatmap image gets 93%, colorbar 5% of top row width
#                                                         wspace=0.5) # Adjust space between heatmap image and its colorbar
#         ax_heatmap_img = fig.add_subplot(gs_top_row[0, 0])
#         cax_heatmap_cb = fig.add_subplot(gs_top_row[0, 1])

#         # --- Bottom Row: Boxplot and Histogram ---
#         gs_bottom_row = gridspec.GridSpecFromSubplotSpec(1, 2,
#                                                             subplot_spec=gs_rows[1],
#                                                             width_ratios=[0.75, 0.25], # Example: Boxplot 75%, histogram 25% of bottom row width
#                                                             wspace=0.02) # Adjust space between boxplot and histogram
#         ax_boxplot = fig.add_subplot(gs_bottom_row[0, 0], sharex=ax_heatmap_img) # Boxplot shares X with heatmap IMAGE
#         ax_hist_y = fig.add_subplot(gs_bottom_row[0, 1], sharey=ax_boxplot)  # Histogram shares Y with boxplot

#         if main_title is None:
#             default_main_title = f"Leadfield Matrix Summary ({orientation_type.capitalize()} Orientation)"
#             fig.suptitle(default_main_title, fontsize=18, y=0.99)
#         elif main_title:
#             fig.suptitle(main_title, fontsize=18, y=0.99)

#         # ... (rest of the plotting logic for heatmap, boxplot, histogram remains the same as the previous version) ...
#         # --- Prepare data for heatmap (lf_for_heatmap: sources on Y, selected sensors on X) ---
#         if orientation_type == "fixed":
#             if leadfield_matrix.ndim != 2:
#                 raise ValueError(f"Heatmap: Expected 2D leadfield for fixed, got {leadfield_matrix.ndim}D")
#             lf_norm_for_heatmap = leadfield_matrix
#             heatmap_title_suffix = "(Fixed Orientation)"
#         elif orientation_type == "free":
#             if leadfield_matrix.ndim != 3 or leadfield_matrix.shape[-1] != 3:
#                 raise ValueError(f"Heatmap: Expected 3D leadfield (..., 3) for free, got {leadfield_matrix.shape}")
#             lf_norm_for_heatmap = np.linalg.norm(leadfield_matrix, axis=-1)
#             heatmap_title_suffix = "(Free Orientation - Norm)"
#         else:
#             raise ValueError("Heatmap: Invalid orientation type.")

#         if len(actual_sensor_indices_to_plot) > 0:
#             lf_selected_sensors = lf_norm_for_heatmap[actual_sensor_indices_to_plot, :]
#             data_for_heatmap_display = lf_selected_sensors.T
#         else:
#             data_for_heatmap_display = np.array([[]])
#             ax_heatmap_img.text(0.5, 0.5, "No sensors for heatmap.", ha='center', va='center')

#         # --- Subplot 1: Flipped Leadfield Heatmap (ax_heatmap_img) & Colorbar (cax_heatmap_cb) ---
#         if data_for_heatmap_display.size > 0 :
#             im = ax_heatmap_img.imshow(data_for_heatmap_display, aspect='auto', cmap='viridis', interpolation='nearest')
#             fig.colorbar(im, cax=cax_heatmap_cb, label=f"Amplitude ({self.leadfield_units})")
#             ax_heatmap_img.set_title(f"Leadfield Matrix {heatmap_title_suffix}", fontsize=14)
#             ax_heatmap_img.set_ylabel("Sources", fontsize=12)
#             ax_heatmap_img.set_xlabel("Sensor Index", fontsize=12)
#         else:
#             ax_heatmap_img.set_title(f"Leadfield Matrix {heatmap_title_suffix}", fontsize=14)
#             ax_heatmap_img.set_ylabel("Sources", fontsize=12)
#             ax_heatmap_img.set_xlabel("Sensor Index", fontsize=12) # Fallback if no data

#         # --- Data for Histogram (Overall Distribution) ---
#         leadfield_values_flat = leadfield_matrix.flatten()

#         # --- Subplot 2: Leadfield Sensor Box Plots (ax_boxplot) ---
#         labels_for_boxplot = [str(idx) for idx in actual_sensor_indices_to_plot]
#         all_q1_values_for_boxplot_sensors = [] 
#         all_q2_values_for_boxplot_sensors = [] 
#         all_min_no_outliers_per_sensor = [] # Store min (no outliers) for each sensor's boxplot data
#         all_max_no_outliers_per_sensor = [] # Store max (no outliers) for each sensor's boxplot data

#         if len(actual_sensor_indices_to_plot) > 0:
#             data_for_boxplot = []
#             for sensor_idx in actual_sensor_indices_to_plot:
#                 current_sensor_data = None
#                 if orientation_type == "fixed":
#                     current_sensor_data = leadfield_matrix[sensor_idx, :]
#                 elif orientation_type == "free":
#                     sensor_values_3d = leadfield_matrix[sensor_idx, :, :]
#                     current_sensor_data = np.linalg.norm(sensor_values_3d, axis=-1)
#                 else: 
#                     self.logger.error(f"Boxplot: Invalid orientation type '{orientation_type}' encountered unexpectedly. Raising ValueError.")
#                     raise ValueError("Boxplot: Invalid orientation type.")
#                 data_for_boxplot.append(current_sensor_data)

#                 if current_sensor_data.size > 0:
#                     all_q1_values_for_boxplot_sensors.append(np.percentile(current_sensor_data, 25))
#                     all_q2_values_for_boxplot_sensors.append(np.percentile(current_sensor_data, 50))

#                     # Calculate min/max without outliers for THIS sensor's data
#                     q1_sensor = np.percentile(current_sensor_data, 25)
#                     q3_sensor = np.percentile(current_sensor_data, 75)
#                     iqr_sensor = q3_sensor - q1_sensor
#                     lower_bound_sensor = q1_sensor - 1.5 * iqr_sensor
#                     upper_bound_sensor = q3_sensor + 1.5 * iqr_sensor
                    
#                     sensor_data_no_outliers = current_sensor_data[
#                         (current_sensor_data >= lower_bound_sensor) &
#                         (current_sensor_data <= upper_bound_sensor)
#                     ]
                    
#                     if sensor_data_no_outliers.size > 0:
#                         all_min_no_outliers_per_sensor.append(np.min(sensor_data_no_outliers))
#                         all_max_no_outliers_per_sensor.append(np.max(sensor_data_no_outliers))
#                     else:
#                         # If all data for a sensor are outliers or it's empty after filtering
#                         all_min_no_outliers_per_sensor.append(np.nan)
#                         all_max_no_outliers_per_sensor.append(np.nan)
#                 else: # current_sensor_data.size == 0
#                     all_min_no_outliers_per_sensor.append(np.nan)
#                     all_max_no_outliers_per_sensor.append(np.nan)
            
#             boxprops = dict(facecolor='skyblue', alpha=0.7, edgecolor='black')
#             medianprops = dict(color="navy", linewidth=1.5)
            
#             bp = ax_boxplot.boxplot(data_for_boxplot, patch_artist=True, labels=labels_for_boxplot,
#                                     boxprops=boxprops, medianprops=medianprops, vert=True)
            
#             ax_boxplot.set_title("Leadfield Amplitude per Sensor", fontsize=14)
#             ax_boxplot.set_ylabel(f"Leadfield Amplitude ({self.leadfield_units})", fontsize=12)
#             ax_boxplot.grid(True, linestyle='--', alpha=0.6, axis='y')
#             ax_boxplot.set_xlabel("Selected Sensor Index", fontsize=12) # This label will be visible
#             plt.setp(ax_boxplot.get_xticklabels(), rotation=45, ha="right" if len(labels_for_boxplot) > 5 else "center")
#         else:
#             ax_boxplot.text(0.5, 0.5, "No sensors for boxplot.", ha='center', va='center')
#             ax_boxplot.set_title("Leadfield Amplitude per Sensor", fontsize=14)
#             ax_boxplot.set_xlabel("Selected Sensor Index", fontsize=12)
#             ax_boxplot.set_ylabel(f"Leadfield Amplitude ({self.leadfield_units})", fontsize=12)
#             self.logger.info("No boxplots generated as no sensors were selected.")

#         # Configure shared X-axis: Heatmap image X-ticks are based on boxplot's
#         if len(actual_sensor_indices_to_plot) > 0 and data_for_heatmap_display.size > 0:
#             ax_heatmap_img.set_xticks(np.arange(len(actual_sensor_indices_to_plot)))
#             plt.setp(ax_heatmap_img.get_xticklabels(), visible=False)
#         # ax_heatmap_img.set_xlabel("") # This was commented out in the provided context, keeping it so

#         # --- Subplot 3: Rotated Histogram (ax_hist_y) ---
#         ax_hist_y.hist(leadfield_values_flat, bins=bins, color='lightcoral', edgecolor='black', alpha=0.7, orientation='horizontal')
#         ax_hist_y.set_title("Overall Distribution", fontsize=14)
#         ax_hist_y.set_xlabel("Frequency", fontsize=12)
#         plt.setp(ax_hist_y.get_yticklabels(), visible=False)
#         ax_hist_y.grid(True, linestyle='--', alpha=0.7, axis='x')

#         mean_val = np.mean(leadfield_values_flat)
#         median_val = np.median(leadfield_values_flat)
#         mean_abs_val = np.mean(np.abs(leadfield_values_flat))
#         std_val = np.std(leadfield_values_flat)
#         min_val_flat = np.min(leadfield_values_flat) # Overall min (with outliers)
#         max_val_flat = np.max(leadfield_values_flat) # Overall max (with outliers)

#         # Calculate mean of Q1 and Q2 values from the boxplot data
#         mean_of_boxplot_q1s = np.nanmean(all_q1_values_for_boxplot_sensors) if all_q1_values_for_boxplot_sensors else np.nan
#         mean_of_boxplot_q2s = np.nanmean(all_q2_values_for_boxplot_sensors) if all_q2_values_for_boxplot_sensors else np.nan
        
#         # Calculate mean of sensor-wise min/max (no outliers)
#         mean_of_sensor_mins_no_outliers = np.nanmean(all_min_no_outliers_per_sensor) if all_min_no_outliers_per_sensor else np.nan
#         mean_of_sensor_maxs_no_outliers = np.nanmean(all_max_no_outliers_per_sensor) if all_max_no_outliers_per_sensor else np.nan
        
#         self.logger.info(f"Leadfield overall flat data stats: N_values={len(leadfield_values_flat)}, Mean={mean_val:.2e}, Std={std_val:.2e}, Median={median_val:.2e}, Min={min_val_flat:.2e}, Max={max_val_flat:.2e}, Mean Abs={mean_abs_val:.2e}")
#         self.logger.info(f"Leadfield boxplot sensors stats: Mean of Q1s={mean_of_boxplot_q1s:.2e}, Mean of Q2s (Medians)={mean_of_boxplot_q2s:.2e} (for {len(all_q1_values_for_boxplot_sensors)} sensors)")
#         self.logger.info(f"Leadfield boxplot sensors (no outliers): Mean of Mins={mean_of_sensor_mins_no_outliers:.2e}, Mean of Maxs={mean_of_sensor_maxs_no_outliers:.2e}")
        
#         stats_text = (f"Overall Mean: {mean_val:.2e}\n"
#                         f"Overall Median: {median_val:.2e}\n"
#                         f"Overall Std: {std_val:.2e}\n"
#                         f"Overall Min: {min_val_flat:.2e}\n"
#                         f"Overall Max: {max_val_flat:.2e}\n"
#                         f"Mean Abs: {mean_abs_val:.2e}\n"
#                         f"Mean Boxplot Q1s: {mean_of_boxplot_q1s:.2e}\n"
#                         f"Mean Boxplot Q2s: {mean_of_boxplot_q2s:.2e}\n"
#                         f"Mean Sensor Min (no outliers): {mean_of_sensor_mins_no_outliers:.2e}\n"
#                         f"Mean Sensor Max (no outliers): {mean_of_sensor_maxs_no_outliers:.2e}")
        
#         ax_hist_y.text(0.95, 0.95, stats_text, transform=ax_hist_y.transAxes, fontsize=9,verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

        
#         fig.tight_layout(rect=[0, 0, 1, 0.97] if main_title else [0,0,1,1])

#         if save_path:
#             save_dir = Path(save_path).parent
#             save_dir.mkdir(parents=True, exist_ok=True)
#             plt.savefig(save_path, bbox_inches="tight", dpi=150)
#             self.logger.info(f"Leadfield summary visualization saved to {save_path}")
#         if show:
#             plt.show()

#     except Exception as e:
#             self.logger.error(f"Failed during leadfield summary visualization: {e}", exc_info=True)
#     finally:
#             if fig:
#                 plt.close(fig)
                                    
# def visualize_leadfield_topomap(
#     self,
#     leadfield_matrix: np.ndarray,
#     x: np.ndarray,
#     info: mne.Info=None,
#     orientation_type: str = "fixed",
#     save_path: Optional[str] = None,
#     title: Optional[str] = None,
#     show: bool = False
# ) -> None:
#     """
#     Visualize leadfield patterns as topomaps for active sources.

#     Parameters
#     ----------
#     leadfield_matrix : np.ndarray
#         The leadfield matrix.
#         - 'fixed': Shape (n_sensors, n_sources).
#         - 'free': Shape (n_sensors, n_sources, 3).
#     info : mne.Info
#         MNE info object containing sensor locations.
#     x : np.ndarray
#         Source activity matrix to determine active sources.
#         - 'fixed': Shape (n_sources, n_times).
#         - 'free': Shape (n_sources, 3, n_times).
#     orientation_type : str, optional
#         Orientation type ('fixed' or 'free'), by default "fixed".
#     save_path : Optional[str], optional
#         Path to save the figure. If None, not saved, by default None.
#     title : Optional[str], optional
#         Title for the entire figure, by default None.
#     show : bool, optional
#         If True, display the plot, by default False.

#     Raises
#     ------
#     ValueError
#         If inputs are invalid or orientation_type is unsupported.
#     """
#     try:
#         if self.channel_type == "eeg":
#             ch_types = ["eeg"] * self.n_sensors
#         elif self.channel_type == "meg":
#             ch_types = ["grad"] * self.n_sensors  # or "mag" if you want magnetometers
#         else:
#             raise ValueError(f"Unsupported channel_type: {self.channel_type}")

#         info = mne.create_info(
#             ch_names=[f"{self.channel_type}{i:03}" for i in range(self.n_sensors)],
#             sfreq=self.sfreq,
#             ch_types=ch_types,
#         )
        
#     except Exception as e:
#         self.logger.error(f"Failed to create MNE info: {e}")
#         info = None
        
#     if leadfield_matrix is None or not isinstance(leadfield_matrix, np.ndarray) or leadfield_matrix.size == 0:
#         self.logger.error("Invalid leadfield matrix provided for topomap visualization.")
#         return
#     if x is None or not isinstance(x, np.ndarray) or x.size == 0:
#         self.logger.error("Invalid source activity matrix provided for topomap visualization.")
#         return
#     if info is None or not isinstance(info, mne.Info):
#             self.logger.error("Invalid MNE info object provided for topomap visualization.")
#             return

#     fig = None # Initialize fig
#     try:
#         if orientation_type == "fixed":
#             if leadfield_matrix.ndim != 2:
#                     raise ValueError(f"Expected 2D leadfield for fixed orientation, got {leadfield_matrix.ndim}D")
#             active_sources = np.where(np.any(x != 0, axis=-1))[0]
#         elif orientation_type == "free":
#             if leadfield_matrix.ndim != 3 or leadfield_matrix.shape[-1] != 3:
#                     raise ValueError(f"Expected 3D leadfield (..., 3) for free orientation, got shape {leadfield_matrix.shape}")
#             self.logger.info("Calculating norm of source activity to find active sources for free orientation.")
#             active_sources = np.where(np.any(x != 0, axis=(1, 2)))[0]
#         else:
#             raise ValueError("Invalid orientation type. Must be 'fixed' or 'free'.")

#         if len(active_sources) == 0:
#                 self.logger.warning("No active sources found to visualize topomaps.")
#                 return

#         n_active = len(active_sources)
#         n_cols = min(5, n_active) # Max 5 columns
#         n_rows = int(np.ceil(n_active / n_cols))
#         fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3), squeeze=False)
#         axes_flat = axes.flatten()

#         # Determine global color limits for consistency
#         all_leadfield_values = []
#         for i, source_idx in enumerate(active_sources):
#             if orientation_type == "fixed":
#                 leadfield_values = leadfield_matrix[:, source_idx]
#             else: # free
#                 # Visualize the norm of the 3 components for simplicity
#                 leadfield_values = np.linalg.norm(leadfield_matrix[:, source_idx, :], axis=-1)
#             all_leadfield_values.append(leadfield_values)

#         if not all_leadfield_values:
#                 self.logger.warning("Could not extract leadfield values for any active source.")
#                 return

#         vmax = np.max(all_leadfield_values)
#         vmin = np.min(all_leadfield_values)

#         for i, source_idx in enumerate(active_sources):
#             leadfield_values = all_leadfield_values[i]
#             im, _ = mne.viz.plot_topomap(
#                 leadfield_values, info, axes=axes_flat[i], cmap="RdBu_r", # Use diverging colormap
#                 # vlim=(vmin, vmax), 
#                 show=False,
#                 contours=6
#             )
#             axes_flat[i].set_title(f"Source {source_idx}")

#         # Add a single colorbar
#         fig.colorbar(im, ax=axes.ravel().tolist(), label=f'Leadfield Amplitude ({self.leadfield_units})', shrink=0.6, aspect=10)

#         # Hide unused subplots
#         for j in range(n_active, len(axes_flat)):
#             axes_flat[j].axis("off")

#         if title:
#             fig.suptitle(title, fontsize=16) # Removed weight="bold"

#         # plt.tight_layout(rect=[0, 0, 1, 0.95] if title else [0, 0, 1, 1])

#         if save_path:
#             save_dir = Path(save_path).parent
#             save_dir.mkdir(parents=True, exist_ok=True)
#             plt.savefig(save_path, bbox_inches="tight")
#             self.logger.info(f"Leadfield topomap visualization saved to {save_path}")

#         if show:
#             plt.show()

#     except Exception as e:
#             self.logger.error(f"Failed during leadfield topomap visualization: {e}")
#     finally:
#             if fig:
#                 plt.close(fig)
            
# def inspect_matrix_values(self, matrix, matrix_name="Matrix"):
#     """
#     Prints summary statistics and checks for invalid values in a NumPy array.

#     Parameters:
#     - matrix (np.ndarray): The matrix to inspect.
#     - matrix_name (str): A name for the matrix used in print statements.
#     """
#     print(f"--- Inspecting {matrix_name} Values ---")
#     if not isinstance(matrix, np.ndarray):
#         print(f"Error: Input is not a NumPy array.")
#         return
#     if matrix.size == 0:
#         print(f"Warning: {matrix_name} is empty.")
#         return

#     try:
#         min_val = np.min(matrix)
#         max_val = np.max(matrix)
#         mean_val = np.mean(matrix)
#         mean_abs_val = np.mean(np.abs(matrix))
#         std_val = np.std(matrix)

#         print(f"{matrix_name} mean: {mean_val:.2e}, std: {std_val:.2e}")
#         print(f"{matrix_name} min: {min_val:.2e}, max: {max_val:.2e}")
#         # print(f"{matrix_name} std: {std_val:.1e}") # Redundant with first line
#         print(f"{matrix_name} mean abs: {mean_abs_val:.2e}")

#         nan_check = np.isnan(matrix).any()
#         inf_check = np.isinf(matrix).any()

#         if nan_check:
#             print(f"WARNING: {matrix_name} contains NaN values!")
#         if inf_check:
#             print(f"WARNING: {matrix_name} contains Inf values!")
#         if not nan_check and not inf_check:
#             print(f"{matrix_name} contains valid numbers (no NaNs or Infs detected).")

#     except Exception as e:
#         print(f"Error during inspection of {matrix_name}: {e}")
#     print(f"--- End {matrix_name} Inspection ---")

# def load_and_validate_leadfield(self, leadfield_file_path, orientation_type):
#     """
#     Loads a leadfield matrix from an .npz file and validates its shape
#     based on the expected orientation type. Includes value inspection.

#     Parameters:
#     - leadfield_file_path (str or Path): Path to the .npz file containing the leadfield.
#     - orientation_type (str): The expected orientation type ("fixed" or "free").

#     Returns:
#     - np.ndarray: The loaded and validated leadfield matrix.

#     Raises:
#     - FileNotFoundError: If the leadfield file does not exist.
#     - KeyError: If the expected key is not found in the .npz file.
#     - ValueError: If the loaded leadfield matrix shape is inconsistent with the orientation_type.
#     - Exception: For other potential loading errors.
#     """
#     print(f"Loading leadfield from: {leadfield_file_path}")
#     try:
#         with np.load(leadfield_file_path) as data:
#             # ... (loading logic as before) ...
#             if 'lead_field' in data:
#                 leadfield_matrix = data["lead_field"]
#             elif 'lead_field_fixed' in data and orientation_type == "fixed":
#                 leadfield_matrix = data['lead_field_fixed']
#             elif 'lead_field_free' in data and orientation_type == "free":
#                 leadfield_matrix = data['lead_field_free']
#             elif 'lead_field' in data:
#                 print("Warning: Loading generic 'lead_field' key. Ensure it matches orientation type.")
#                 leadfield_matrix = data["lead_field"]
#             else:
#                 keys_found = list(data.keys())
#                 raise KeyError(f"Could not find a suitable leadfield key ('lead_field', 'lead_field_fixed', 'lead_field_free') in .npz file. Found keys: {keys_found}")

#         print(f"Leadfield loaded successfully. Initial Shape: {leadfield_matrix.shape}", "dtype:", leadfield_matrix.dtype)

#         # --- Validate leadfield shape against orientation_type ---
#         # ... (validation logic as before) ...
#         if orientation_type == "fixed":
#             if leadfield_matrix.ndim != 2:
#                 raise ValueError(f"Expected 2D leadfield for fixed orientation, got shape {leadfield_matrix.shape}")
#         elif orientation_type == "free":
#             if leadfield_matrix.ndim == 3:
#                 if leadfield_matrix.shape[2] != 3:
#                     raise ValueError(f"Expected 3 components in last dimension for free orientation, got shape {leadfield_matrix.shape}")
#             elif leadfield_matrix.ndim == 2:
#                 if leadfield_matrix.shape[1] % 3 == 0:
#                     print("Warning: Reshaping potentially flattened free orientation leadfield.")
#                     n_sensors, n_sources_x_3 = leadfield_matrix.shape
#                     n_sources = n_sources_x_3 // 3
#                     leadfield_matrix = leadfield_matrix.reshape(n_sensors, n_sources, 3)
#                     print(f"Reshaped leadfield to {leadfield_matrix.shape}")
#                 else:
#                     raise ValueError(f"Cannot reshape 2D leadfield (shape {leadfield_matrix.shape}) to 3D free orientation.")
#             else:
#                 raise ValueError(f"Expected 2D or 3D leadfield for free orientation, got {leadfield_matrix.ndim} dimensions with shape {leadfield_matrix.shape}")
#         else:
#             raise ValueError(f"Invalid orientation_type specified: {orientation_type}. Choose 'fixed' or 'free'.")


#         print(f"Leadfield validated successfully. Final Shape: {leadfield_matrix.shape}")

#         # --- Inspect Leadfield Matrix Values using the function ---
#         self.inspect_matrix_values(leadfield_matrix, matrix_name="Leadfield")
#         # --- End Inspection ---

#         return leadfield_matrix

#     except FileNotFoundError:
#         print(f"Error: Leadfield file not found at {leadfield_file_path}")
#         raise # Re-raise the exception
#     except (KeyError, ValueError) as e:
#         print(f"Error loading or validating leadfield: {e}")
#         raise # Re-raise the specific error
#     except Exception as e:
#         print(f"An unexpected error occurred during leadfield loading: {e}")
#         raise 

# # --- Plotting Functions ---
# def plot_sensor_signals(self, y_clean, y_noisy, sensor_indices=None, times=None, save_dir=None, figure_name=None, trial_idx=None):
#     """ Plot clean and noisy sensor signals for specific sensors for a specific trial. """
#     if sensor_indices is None:
#         sensor_indices = [0]
#     if times is None:
#         times = np.arange(y_clean.shape[1])

#     n_sensors_to_plot = len(sensor_indices)
#     fig, axes = plt.subplots(n_sensors_to_plot, 1, figsize=(10, n_sensors_to_plot * 3), sharex=True, sharey=True)
#     title_suffix = f" (Trial {trial_idx+1})" if trial_idx is not None else ""
#     fig.suptitle(f"Specific Sensor Signals{title_suffix}", fontsize=16)

#     if n_sensors_to_plot == 1:
#         axes = [axes]

#     for i, sensor_idx in enumerate(sensor_indices):
#         axes[i].plot(times, y_clean[sensor_idx], label="y_clean", linewidth=2)
#         axes[i].plot(times, y_noisy[sensor_idx], label="y_noise")
#         axes[i].set_title(f"Sensor {sensor_idx}")
#         axes[i].set_xlabel("Time (s)")
#         axes[i].set_ylabel(f"Amplitude  ({self.sensor_units})")
#         axes[i].legend()
#         axes[i].grid(True)

#     plt.tight_layout(rect=[0, 0, 1, 0.95])

#     if save_dir and figure_name:
#         output_dir = Path(save_dir)
#         output_dir.mkdir(parents=True, exist_ok=True)
#         save_path = output_dir / f"{figure_name}.png"
#         plt.savefig(save_path, dpi=300)
#         print(f"Sensor subplots figure saved to {save_path}")
    
#     plt.close(fig)

# def plot_all_active_sources_single_figure(self, x, times, active_indices, stim_onset, save_dir=None, figure_name=None, trial_idx=None):
#     """ Plot all specified active source signals on a single figure for a specific trial. """
#     fig, ax = plt.subplots(1, 1, figsize=(12, 6))
#     title_suffix = f" (Trial {trial_idx+1})" if trial_idx is not None else ""
#     fig.suptitle(f"All Active Source Signals{title_suffix}", fontsize=16)
#     colors = cm.viridis(np.linspace(0, 1, len(active_indices)))

#     # Handle potential free orientation source data shape
#     if x.ndim == 3:
#         x_plot = np.linalg.norm(x, axis=1) # Plot magnitude
#     else:
#         x_plot = x

#     for i, src_idx in enumerate(active_indices):
#         ax.plot(times, x_plot[src_idx], label=f"Source {src_idx}", linewidth=1.5, color=colors[i])

#     ax.axvline(x=stim_onset, linestyle="--", color="gray", label="Stimulus Onset")
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel(f"Amplitude ({self.source_units})")
#     ax.legend(loc='best', fontsize='small')
#     ax.grid(True, alpha=0.6)
#     ax.set_title("Active Sources")
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#     if save_dir and figure_name:
#         output_dir = Path(save_dir)
#         output_dir.mkdir(parents=True, exist_ok=True)
#         save_path = output_dir / f"{figure_name}.png"
#         plt.savefig(save_path, dpi=300)
#         print(f"Single figure source plot saved to {save_path}")
#     plt.close(fig)

# def plot_all_sensor_signals_single_figure(self, y_data, times, sensor_indices=None, save_dir=None, figure_name=None, trial_idx=None, average_epochs=False):
#     """
#     Plot sensor signals (overlay) for selected sensors.
#     If average_epochs is True and y_data is 3D, plots the average across epochs for each channel.
#     If average_epochs is False and y_data is 2D, plots the single trial data.
#     Does NOT average across channels.

#     Parameters:
#     - y_data (np.ndarray): Sensor measurements. Can be 2D (n_channels, n_times) for a single trial
#                         or 3D (n_trials, n_channels, n_times) for multiple trials.
#     - times (np.ndarray): Time vector corresponding to the signals.
#     - sensor_indices (list or np.ndarray, optional): Indices of sensors to plot. If None, plots all sensors.
#     - save_dir (str or Path, optional): Directory to save the figure.
#     - figure_name (str, optional): Name of the figure file (without extension).
#     - trial_idx (int, optional): Index of the trial being plotted (used for title if y_data is 2D and average_epochs is False).
#     - average_epochs (bool): If True and y_data is 3D, plot the average across trials.
#                             If False and y_data is 3D, raises an error.
#                             If y_data is 2D, this primarily affects the title.
#     """
#     fig, ax = plt.subplots(1, 1, figsize=(12, 6))

#     title_suffix = ""
#     plot_individual_epochs = False # Flag to control plotting individual trials (currently always False)

#     if y_data.ndim == 2: # Input is single trial or already averaged data
#         y_plot = y_data # This is the data to plot (n_channels, n_times)
#         if not average_epochs and trial_idx is not None:
#             title_suffix = f" (Trial {trial_idx+1})"
#         elif average_epochs: # Assume 2D input might be an average if flag is set
#             title_suffix = " (Average across Trials)"
#         # If 2D and not average_epochs and no trial_idx, title is generic
#     elif y_data.ndim == 3: # Input is multi-trial data
#         if average_epochs:
#             y_plot = np.mean(y_data, axis=0) # Calculate average across trials (axis 0) -> shape (n_channels, n_times)
#             title_suffix = " (Average across Trials)"
#             # Do not plot individual epochs if averaging is requested
#             plot_individual_epochs = False
#         else:
#             # If 3D data is passed but averaging is not requested, it's ambiguous.
#             raise ValueError("Input y_data is 3D, but average_epochs is False. "
#                             "Provide 2D data (single trial) or set average_epochs=True.")
#     else:
#         raise ValueError("Input y_data must be 2D or 3D")

#     # Select specific sensors if requested from the data to be plotted (y_plot)
#     if sensor_indices is None:
#         sensor_indices_to_plot = np.arange(y_plot.shape[0]) # Use all channels
#         y_plot_selected = y_plot
#     else:
#         # Ensure indices are valid for the potentially averaged data
#         sensor_indices_to_plot = np.array(sensor_indices)[np.array(sensor_indices) < y_plot.shape[0]]
#         if len(sensor_indices_to_plot) != len(sensor_indices):
#             print("Warning: Some requested sensor_indices are out of bounds for the provided data.")
#         y_plot_selected = y_plot[sensor_indices_to_plot, :]

#     n_plot_sensors = y_plot_selected.shape[0]

#     fig.suptitle(f"Sensor Signals {title_suffix}", fontsize=16)
#     colors = cm.turbo(np.linspace(0, 1, n_plot_sensors))

#     # --- Plotting Logic ---
#     # Plot the main traces (either single trial or trial-averaged)
#     for i in range(n_plot_sensors):
#         actual_sensor_idx = sensor_indices_to_plot[i] # Get original index
#         ax.plot(times, y_plot_selected[i, :], linewidth=1.0, color=colors[i], alpha=0.8, label=f"Ch {actual_sensor_idx}" if n_plot_sensors <= 15 else None)

#     # Optional: Plot individual epoch traces lightly in the background (currently disabled)
#     if plot_individual_epochs and y_data.ndim == 3:
#         y_plot_all_selected = y_data[:, sensor_indices_to_plot, :] # Select sensors from original 3D data
#         for i_trial in range(y_data.shape[0]):
#             for i_ch in range(n_plot_sensors):
#                 ax.plot(times, y_plot_all_selected[i_trial, i_ch, :], linewidth=0.2, color=colors[i_ch], alpha=0.1)


#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel(f"Amplitude ({self.sensor_units})")
#     ax.grid(True, alpha=0.6)
#     ax.set_title(f"{n_plot_sensors} channels")

#     # Update legend
#     if n_plot_sensors <= 15: # Show legend only for fewer channels
#         ax.legend(loc='best', fontsize='small')

#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#     if save_dir and figure_name:
#         output_dir = Path(save_dir)
#         output_dir.mkdir(parents=True, exist_ok=True)
#         save_path = output_dir / f"{figure_name}.png"
#         plt.savefig(save_path, dpi=300)
#     plt.close(fig)

# def plot_active_sources(self, x, times, active_indices, stim_onset, nnz, save_dir=None, figure_name=None, trial_idx=None):
#     """ Plot active sources for a specific trial. """
#     n_cols = 3
#     n_rows = int(np.ceil(nnz / n_cols))
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), constrained_layout=True, sharex=True, sharey=True)
#     title_suffix = f" (Trial {trial_idx+1})" if trial_idx is not None else ""
#     fig.suptitle(f"Active Source Signals{title_suffix}", fontsize=16)
#     axes = axes.flatten()

#     # Handle potential free orientation source data shape (n_sources, n_orient, n_times)
#     # Plot the norm or the first component for simplicity
#     if x.ndim == 3:
#         x_plot = np.linalg.norm(x, axis=1) # Plot magnitude for free orientation
#         # Or plot first component: x_plot = x[:, 0, :]
#     else:
#         x_plot = x

#     for i, src_idx in enumerate(active_indices):
#         axes[i].plot(times, x_plot[src_idx], label=f"Source {src_idx}", linewidth=2)
#         axes[i].axvline(x=stim_onset, linestyle="--", color="gray", label="Stimulus Onset")
#         axes[i].set_xlabel("Time (s)")
#         axes[i].set_ylabel(f"Amplitude ({self.source_units})")
#         axes[i].set_title(f"Active Source {src_idx}")
#         axes[i].legend()
#         axes[i].grid(True)

#     for j in range(i + 1, len(axes)):
#         axes[j].axis("off")

#     if save_dir and figure_name:
#         output_dir = Path(save_dir)
#         output_dir.mkdir(parents=True, exist_ok=True)
#         save_path = output_dir / f"{figure_name}.png"
#         plt.savefig(save_path, dpi=300)
#         print(f"Subplots figure saved to {save_path}")
#     plt.close(fig)



# # ========================== from uncertainty_estimation.py =========================

#     def plot_sorted_posterior_variances(self, top_k=None):
#         """
#         Plot the sorted variances from the covariance matrix, highlighting the top-k variances.
#         """
#         variances = np.diag(self.posterior_cov)
#         sorted_indices = np.argsort(variances)[::-1]
#         sorted_variances = variances[sorted_indices]

#         plt.figure(figsize=(12, 6))
#         bars = plt.bar(range(len(sorted_variances)), sorted_variances, color='skyblue', edgecolor='blue')

#         if top_k is not None:
#             for bar in bars[:top_k]:
#                 bar.set_color('orange')

#         plt.xlabel("Source Index")
#         plt.ylabel("Variance")
#         plt.title(f"Sorted Posterior Variances (Top-{top_k if top_k else len(variances)} Highlighted)")
#         plt.grid(axis='y', linestyle='--', alpha=0.7)
#         plt.tight_layout(rect=[0, 0.05, 1, 0.96])
#         plt.savefig(os.path.join(self.experiment_dir, 'sorted_variances.png'))
#         plt.close()

#     def _plot_confidence_ellipse(self, mean, width, height, angle, ax=None, **kwargs):
#         """
#         Plot a confidence ellipse for given parameters.

#         Parameters:
#         - mean: array-like, shape (2,)
#             The mean of the data in the two dimensions being plotted.
#         - width: float
#             The width of the ellipse (related to variance along the major axis).
#         - height: float
#             The height of the ellipse (related to variance along the minor axis).
#         - angle: float
#             The rotation angle of the ellipse in degrees.
#         - ax: matplotlib.axes.Axes, optional
#             The axis on which to plot the ellipse. If None, creates a new figure.
#         - **kwargs: additional keyword arguments for matplotlib.patches.Ellipse.
#         """
#         if ax is None:
#             fig, ax = plt.subplots()

#         # Add ellipse patch
#         ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
#         ax.add_patch(ellipse)
#         ax.scatter(*mean, color='blue', label='Mean')
        
#         # Set axis labels
#         ax.set_xlabel("Principal Component 1 (Variance in Dim 1)")
#         ax.set_ylabel("Principal Component 2 (Variance in Dim 2)")

#         # Set title
#         ax.set_title("Confidence Ellipse (Width and Height Indicate Variance)")
#         ax.grid()
#         ax.legend()
    
#     def plot_top_relevant_CE_pairs(self, top_k=5, confidence_level=0.95):
#         """
#         Identify the top-k relevant pairs of dimensions (based on covariance magnitude)
#         and plot their confidence ellipses.
#         """
#         mean = self.x_hat[self.active_indices]
#         cov = self.posterior_cov

#         n = len(mean)
#         pairs = list(combinations(range(n), 2))
#         pair_cov_magnitudes = [(pair, np.abs(cov[pair[0], pair[1]])) for pair in pairs]
#         sorted_pairs = sorted(pair_cov_magnitudes, key=lambda x: x[1], reverse=True)
#         top_pairs = [pair for pair, _ in sorted_pairs[:top_k]]

#         n_cols = min(3, top_k)
#         n_rows = (top_k + n_cols - 1) // n_cols
#         fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
#         axes = axes.flatten()

#         for idx, (i, j) in enumerate(top_pairs):
#             mean_ij = mean[[i, j]]
#             cov_ij = cov[np.ix_([i, j], [i, j])]

#             width, height, angle = self._compute_confidence_ellipse(mean_ij, cov_ij, confidence_level)
#             self._plot_confidence_ellipse(mean_ij, width, height, angle, ax=axes[idx], edgecolor='blue', alpha=0.5)

#             axes[idx].set_title(f"Dimensions {i} & {j}")

#         for ax in axes[len(top_pairs):]:
#             fig.delaxes(ax)

#         fig.suptitle("Top Relevant Dimensional Pairs with Confidence Ellipses", fontsize=16)
#         plt.tight_layout(rect=[0, 0.05, 1, 0.96])
#         plt.savefig(os.path.join(self.experiment_dir, 'top_relevant_CE_pairs.png'))
#         plt.close()
        
#     def plot_posterior_covariance_matrix(self):
#         """
#         Plot the posterior covariance matrix.
#         """
#         if self.orientation_type == 'free':
#             # Check if posterior_cov shape is compatible with free orientation slicing
#             n_active_components = self.posterior_cov.shape[0]
#             if n_active_components % 3 != 0:
#                 self.logger.warning(f"Free orientation: posterior_cov shape {self.posterior_cov.shape}, first dimension is not divisible by 3.")
#                 # Fallback to plotting the whole matrix if slicing is not possible
#                 fig, ax = plt.subplots(figsize=(10, 8))
#                 im = ax.imshow(self.posterior_cov, cmap='viridis', aspect='auto')
#                 fig.colorbar(im, ax=ax, label='Covariance Value')
#                 ax.set_title('Posterior Covariance Matrix (Free Orientation - Full)')
#                 ax.set_xlabel('Active Component Index')
#                 ax.set_ylabel('Active Component Index')
#                 plt.tight_layout(rect=[0, 0.05, 1, 0.96])

#             else:
#                 fig, axes = plt.subplots(3, 1, figsize=(10, 18))
#                 orientations = ['X', 'Y', 'Z']
#                 # Determine shared color limits across the subplots
#                 vmin = np.min(self.posterior_cov)
#                 vmax = np.max(self.posterior_cov)

#                 images = [] # Store image objects for colorbar
#                 for i, ax in enumerate(axes):
#                     # Select the block corresponding to the orientation
#                     # This assumes active_indices components are ordered [src0_x, src0_y, src0_z, src1_x, ...]
#                     # which might not be true. A safer plot might show the full matrix.
#                     # Let's plot the diagonal blocks for now, assuming structure.
#                     try:
#                         cov_matrix_block = self.posterior_cov[i::3, i::3]
#                         im = ax.imshow(cov_matrix_block, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
#                         images.append(im)
#                         ax.set_title(f'Diagonal Block - Orientation {orientations[i]}')
#                         ax.set_xlabel('Source Index (within orientation)')
#                         ax.set_ylabel('Source Index (within orientation)')
#                     except IndexError:
#                         self.logger.warning(f"Could not extract block {i}::3 for orientation {orientations[i]}. Skipping subplot.")
#                         ax.set_title(f'Orientation {orientations[i]} - Error')

#                 plt.tight_layout(rect=[0, 0.05, 1, 0.96])

#                 # Add colorbar spanning all axes, using the first image's mappable
#                 if images:
#                     fig.colorbar(images[0], ax=axes.ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.04, label='Covariance Value')


#         else: # Fixed orientation
#             fig, ax = plt.subplots(figsize=(10, 8))
#             im = ax.imshow(self.posterior_cov, cmap='viridis', aspect='auto')
#             plt.colorbar(im, label='Covariance Value')
#             ax.set_title('Posterior Covariance Matrix (Fixed Orientation)')
#             ax.set_xlabel('Active Source Index')
#             ax.set_ylabel('Active Source Index')
#             plt.tight_layout(rect=[0, 0.05, 1, 0.96])

#         try:
#             plt.savefig(os.path.join(self.experiment_dir, 'posterior_covariance_matrix.png'))
#             self.logger.info(f"Posterior covariance matrix plot saved to {self.experiment_dir}/posterior_covariance_matrix.png")
#         except Exception as e:
#             self.logger.error(f"Failed to save posterior covariance matrix plot: {e}")
#         finally:
#             plt.close(fig) 
    
#     def visualize_sorted_covariances(self, top_k=None):
#         """
#         Visualize the sorted magnitudes of covariances for all pairs of dimensions.
#         """
#         sorted_pairs = self._compute_top_covariance_pairs(self.posterior_cov, top_k=top_k)
#         pairs = [f"({i},{j})" for (i, j), _ in sorted_pairs]
#         magnitudes = [magnitude for _, magnitude in sorted_pairs]

#         plt.figure(figsize=(10, 6))
#         plt.bar(pairs, magnitudes, color='skyblue')
#         plt.xlabel('Pairs of Dimensions')
#         plt.ylabel('Covariance Magnitude')
#         plt.title(f"Top-{top_k if top_k else len(magnitudes)} Sorted Covariance Magnitudes")
#         plt.xticks(rotation=45, ha='right')
#         plt.tight_layout(rect=[0, 0.05, 1, 0.96])
#         plt.savefig(os.path.join(self.experiment_dir, 'sorted_covariances.png'))
#         plt.close()


# ============================================================================

# def visualize_leadfield(
#     self,
#     leadfield_matrix: np.ndarray,
#     orientation_type: str = "fixed",
#     save_path: Optional[str] = None,
#     show: bool = False
# ) -> None:
#     """
#     Visualize the leadfield matrix as a heatmap.

#     Parameters
#     ----------
#     leadfield_matrix : np.ndarray
#         The leadfield matrix.
#         - 'fixed': Shape (n_sensors, n_sources).
#         - 'free': Shape (n_sensors, n_sources, 3).
#     orientation_type : str, optional
#         Orientation type ('fixed' or 'free'), by default "fixed".
#     save_path : Optional[str], optional
#         Path to save the figure. If None, not saved, by default None.
#     show : bool, optional
#         If True, display the plot, by default False.

#     Raises
#     ------
#     ValueError
#         If leadfield_matrix is invalid or orientation_type is unsupported.
#     """
#     if leadfield_matrix is None or not isinstance(leadfield_matrix, np.ndarray) or leadfield_matrix.size == 0:
#         self.logger.error("Invalid leadfield matrix provided for visualization.")
#         return

#     fig = None # Initialize fig
#     try:
#         if orientation_type == "fixed":
#             if leadfield_matrix.ndim != 2:
#                  raise ValueError(f"Expected 2D leadfield for fixed orientation, got {leadfield_matrix.ndim}D")
#             fig, ax = plt.subplots(figsize=(10, 8))
#             im = ax.imshow(leadfield_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
#             fig.colorbar(im, ax=ax, label="Amplitude (µV / nAm)", fraction=0.05, pad=0.04)
#             ax.set_title("Leadfield Matrix (Fixed Orientation)")
#             ax.set_xlabel("Sources")
#             ax.set_ylabel("Sensors")
#         elif orientation_type == "free":
#             if leadfield_matrix.ndim != 3 or leadfield_matrix.shape[-1] != 3:
#                  raise ValueError(f"Expected 3D leadfield (..., 3) for free orientation, got shape {leadfield_matrix.shape}")
#             n_orient = leadfield_matrix.shape[-1]
#             fig, axes = plt.subplots(1, n_orient, figsize=(15, 5), sharey=True)
#             if n_orient == 1: axes = [axes] # Ensure axes is iterable
#             orientations = ["X", "Y", "Z"]
#             images = []
#             for i in range(n_orient):
#                 im = axes[i].imshow(leadfield_matrix[:, :, i], aspect='auto', cmap='viridis', interpolation='nearest')
#                 images.append(im)
#                 axes[i].set_title(f"Leadfield Matrix ({orientations[i]})")
#                 axes[i].set_xlabel("Sources")
#             axes[0].set_ylabel("Sensors")
#             fig.colorbar(images[0], ax=axes, location="right", label="Amplitude (µV / nAm)", fraction=0.05, pad=0.04)
#         else:
#             raise ValueError("Invalid orientation type. Must be 'fixed' or 'free'.")

#         plt.tight_layout()

#         if save_path:
#             save_dir = Path(save_path).parent
#             save_dir.mkdir(parents=True, exist_ok=True)
#             plt.savefig(save_path, bbox_inches="tight")
#             self.logger.info(f"Leadfield matrix visualization saved to {save_path}")
#         if show:
#             plt.show()

#     except Exception as e:
#          self.logger.error(f"Failed during leadfield visualization: {e}")
#     finally:
#          if fig:
#              plt.close(fig)

# def visualize_leadfield_distribution(
#     self,
#     leadfield_matrix: np.ndarray,
#     orientation_type: str = "fixed",
#     bins: int = 100,
#     save_path: Optional[str] = None,
#     title: Optional[str] = None,
#     show: bool = False
# ) -> None:
#     """
#     Visualize the distribution of leadfield amplitude values using a histogram.

#     Parameters
#     ----------
#     leadfield_matrix : np.ndarray
#         The leadfield matrix.
#         - 'fixed': Shape (n_sensors, n_sources).
#         - 'free': Shape (n_sensors, n_sources, 3).
#     orientation_type : str, optional
#         Orientation type ('fixed' or 'free'), by default "fixed".
#         This mainly affects the title and interpretation.
#     bins : int, optional
#         Number of bins for the histogram, by default 100.
#     save_path : Optional[str], optional
#         Path to save the figure. If None, not saved, by default None.
#     title : Optional[str], optional
#         Custom title for the plot. If None, a default title is generated.
#     show : bool, optional
#         If True, display the plot, by default False.
#     """
#     if leadfield_matrix is None or not isinstance(leadfield_matrix, np.ndarray) or leadfield_matrix.size == 0:
#         self.logger.error("Invalid leadfield matrix provided for distribution visualization.")
#         return

#     fig = None # Initialize fig
#     try:
#         fig, ax = plt.subplots(figsize=(10, 6))

#         # Flatten the leadfield matrix to get all values for the histogram
#         # For 'free' orientation, this will include values from all X, Y, Z components.
#         leadfield_values_flat = leadfield_matrix.flatten()

#         ax.hist(leadfield_values_flat, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)

#         if title is None:
#             default_title = f"Distribution of Leadfield Amplitudes ({orientation_type.capitalize()} Orientation)"
#             ax.set_title(default_title, fontsize=14)
#         else:
#             ax.set_title(title, fontsize=14)

#         ax.set_xlabel("Leadfield Amplitude (µV / nAm)", fontsize=12)
#         ax.set_ylabel("Frequency", fontsize=12)
#         ax.grid(True, linestyle='--', alpha=0.7)

#         # Add some statistics to the plot
#         mean_val = np.mean(leadfield_values_flat)
#         std_val = np.std(leadfield_values_flat)
#         median_val = np.median(leadfield_values_flat)
#         min_val = np.min(leadfield_values_flat)
#         max_val = np.max(leadfield_values_flat)

#         stats_text = (
#             f"Mean: {mean_val:.2e}\nStd: {std_val:.2e}\nMedian: {median_val:.2e}\n"
#             f"Min: {min_val:.2e}\nMax: {max_val:.2e}\nN Values: {len(leadfield_values_flat)}"
#         )
#         # Position the text box in the upper right corner
#         ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
#                 verticalalignment='top', horizontalalignment='right',
#                 bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))


#         plt.tight_layout()

#         if save_path:
#             save_dir = Path(save_path).parent
#             save_dir.mkdir(parents=True, exist_ok=True)
#             plt.savefig(save_path, bbox_inches="tight")
#             self.logger.info(f"Leadfield distribution visualization saved to {save_path}")
#         if show:
#             plt.show()

#     except Exception as e:
#          self.logger.error(f"Failed during leadfield distribution visualization: {e}")
#     finally:
#          if fig:
#              plt.close(fig)



# def visualize_leadfield_summary(
#     self,
#     leadfield_matrix: np.ndarray,
#     orientation_type: str = "fixed",
#     bins: int = 100,
#     sensor_indices_to_plot: Optional[List[int]] = None,
#     max_sensors_to_plot: int = 10,
#     main_title: Optional[str] = None,
#     save_path: Optional[str] = None,
#     show: bool = False
# ) -> None:
#     """
#     Visualize a summary of the leadfield matrix in a single figure:
#     1. Top: Heatmap of the leadfield (norm for 'free' orientation).
#     2. Bottom-Left: Box plots of leadfield amplitudes for selected sensors.
#     3. Bottom-Right: Rotated histogram of all leadfield amplitudes (marginal to boxplots).

#     Parameters
#     ----------
#     leadfield_matrix : np.ndarray
#         The leadfield matrix.
#         - 'fixed': Shape (n_sensors, n_sources).
#         - 'free': Shape (n_sensors, n_sources, 3).
#     orientation_type : str, optional
#         Orientation type ('fixed' or 'free'), by default "fixed".
#     bins : int, optional
#         Number of bins for the histogram subplot, by default 100.
#     sensor_indices_to_plot : Optional[List[int]], optional
#         Specific list of sensor indices for the box plot. If None, a subset is chosen.
#     max_sensors_to_plot : int, optional
#         Maximum number of sensors for the box plot if sensor_indices_to_plot is None.
#     main_title : Optional[str], optional
#         Overall title for the figure.
#     save_path : Optional[str], optional
#         Path to save the figure.
#     show : bool, optional
#         If True, display the plot.
#     """
#     if leadfield_matrix is None or not isinstance(leadfield_matrix, np.ndarray) or leadfield_matrix.size == 0:
#         self.logger.error("Invalid leadfield matrix provided for summary visualization.")
#         return

#     fig = None
#     try:
#         # Define the layout using GridSpec
#         # Figure will have 2 main rows. The second row is split into 2 columns.
#         # Heatmap takes more vertical space.
#         fig = plt.figure(figsize=(15, 18)) # Adjusted figsize
#         gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1], width_ratios=[3, 1])

#         ax_heatmap = fig.add_subplot(gs[0, :])  # Heatmap spans both columns of the first row
#         ax_boxplot = fig.add_subplot(gs[1, 0])  # Boxplot in the second row, first column
#         ax_hist_y = fig.add_subplot(gs[1, 1], sharey=ax_boxplot) # Rotated histogram, shares y-axis with boxplot

#         if main_title is None:
#             default_main_title = f"Leadfield Matrix Summary ({orientation_type.capitalize()} Orientation)"
#             fig.suptitle(default_main_title, fontsize=18, y=0.99)
#         elif main_title:
#             fig.suptitle(main_title, fontsize=18, y=0.99)

#         # --- Subplot 1: Leadfield Heatmap (ax_heatmap) ---
#         if orientation_type == "fixed":
#             if leadfield_matrix.ndim != 2:
#                 raise ValueError(f"Heatmap: Expected 2D leadfield for fixed, got {leadfield_matrix.ndim}D")
#             lf_to_plot = leadfield_matrix
#             heatmap_title = "Leadfield Matrix (Fixed Orientation)"
#         elif orientation_type == "free":
#             if leadfield_matrix.ndim != 3 or leadfield_matrix.shape[-1] != 3:
#                 raise ValueError(f"Heatmap: Expected 3D leadfield (..., 3) for free, got {leadfield_matrix.shape}")
#             lf_to_plot = np.linalg.norm(leadfield_matrix, axis=-1)
#             heatmap_title = "Leadfield Matrix (Free Orientation - Norm)"
#         else:
#             raise ValueError("Heatmap: Invalid orientation type.")
        
#         im = ax_heatmap.imshow(lf_to_plot, aspect='auto', cmap='viridis', interpolation='nearest')
#         # Add colorbar to the heatmap subplot
#         cbar = fig.colorbar(im, ax=ax_heatmap, label="Amplitude (µV / nAm)", fraction=0.046, pad=0.04, orientation='vertical')
#         ax_heatmap.set_title(heatmap_title, fontsize=14)
#         ax_heatmap.set_xlabel("Sources", fontsize=12)
#         ax_heatmap.set_ylabel("Sensors", fontsize=12)

#         # --- Data for Histogram and Boxplot ---
#         leadfield_values_flat = leadfield_matrix.flatten() # For overall distribution
#         num_total_sensors = leadfield_matrix.shape[0]
#         actual_sensor_indices_to_plot: np.ndarray

#         if sensor_indices_to_plot is None:
#             if num_total_sensors > max_sensors_to_plot:
#                 actual_sensor_indices_to_plot = np.linspace(0, num_total_sensors - 1, max_sensors_to_plot, dtype=int)
#             else:
#                 actual_sensor_indices_to_plot = np.arange(num_total_sensors)
#         else:
#             actual_sensor_indices_to_plot = np.array(sensor_indices_to_plot, dtype=int)
#             if np.any(actual_sensor_indices_to_plot < 0) or np.any(actual_sensor_indices_to_plot >= num_total_sensors):
#                 self.logger.error("Boxplot: Invalid sensor_indices_to_plot.")
#                 ax_boxplot.text(0.5, 0.5, "Error: Invalid sensor indices.", ha='center', va='center', color='red')
#                 actual_sensor_indices_to_plot = np.array([])

#         # --- Subplot 2: Leadfield Sensor Box Plots (ax_boxplot) ---
#         if len(actual_sensor_indices_to_plot) > 0:
#             data_for_boxplot = []
#             labels_for_boxplot = []
#             for sensor_idx in actual_sensor_indices_to_plot:
#                 if orientation_type == "fixed":
#                     sensor_values = leadfield_matrix[sensor_idx, :]
#                 elif orientation_type == "free":
#                     sensor_values_3d = leadfield_matrix[sensor_idx, :, :]
#                     sensor_values = np.linalg.norm(sensor_values_3d, axis=-1)
#                 else:
#                     raise ValueError("Boxplot: Invalid orientation type.")
#                 data_for_boxplot.append(sensor_values)
#                 labels_for_boxplot.append(str(sensor_idx))
            
#             bp = ax_boxplot.boxplot(data_for_boxplot, patch_artist=True, medianprops=dict(color="black", linewidth=1.5), vert=True)
#             try:
#                 colors_list = cm.get_cmap('viridis', len(data_for_boxplot))
#                 for i, patch in enumerate(bp['boxes']):
#                     patch.set_facecolor(colors_list(i / len(data_for_boxplot)))
#             except AttributeError:
#                  self.logger.warning("Boxplot: Could not apply distinct colors.")
            
#             ax_boxplot.set_title("Leadfield Amplitude per Sensor", fontsize=14)
#             ax_boxplot.set_xlabel("Sensor Index", fontsize=12)
#             ax_boxplot.set_ylabel("Leadfield Amplitude (µV / nAm)", fontsize=12)
#             ax_boxplot.set_xticklabels(labels_for_boxplot, rotation=45, ha="right" if len(labels_for_boxplot) > 5 else "center")
#             ax_boxplot.grid(True, linestyle='--', alpha=0.6, axis='y')
#         elif not (np.any(actual_sensor_indices_to_plot < 0) or np.any(actual_sensor_indices_to_plot >= num_total_sensors)):
#             ax_boxplot.text(0.5, 0.5, "No sensors for boxplot.", ha='center', va='center')
#             ax_boxplot.set_xlabel("Sensor Index", fontsize=12)
#             ax_boxplot.set_ylabel("Leadfield Amplitude (µV / nAm)", fontsize=12)


#         # --- Subplot 3: Rotated Histogram (ax_hist_y) ---
#         # This histogram shows the distribution of ALL leadfield values
#         ax_hist_y.hist(leadfield_values_flat, bins=bins, color='skyblue', edgecolor='black', alpha=0.7, orientation='horizontal')
#         ax_hist_y.set_title("Overall Distribution", fontsize=14)
#         ax_hist_y.set_xlabel("Frequency", fontsize=12)
#         # Remove y-tick labels for the histogram as it shares y-axis with boxplot
#         plt.setp(ax_hist_y.get_yticklabels(), visible=False)
#         ax_hist_y.grid(True, linestyle='--', alpha=0.7, axis='x')

#         mean_val = np.mean(leadfield_values_flat)
#         std_val = np.std(leadfield_values_flat)
#         median_val = np.median(leadfield_values_flat)
#         stats_text = (
#             f"Mean: {mean_val:.2e}\nStd: {std_val:.2e}\nMedian: {median_val:.2e}"
#         )
#         # Add stats text to the histogram plot, adjusting position for horizontal orientation
#         ax_hist_y.text(0.95, 0.95, stats_text, transform=ax_hist_y.transAxes, fontsize=9,
#                        verticalalignment='top', horizontalalignment='right',
#                        bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))


#         # Adjust layout
#         gs.tight_layout(fig, rect=[0, 0, 1, 0.96] if main_title else [0,0,1,1]) # Use GridSpec's tight_layout

#         if save_path:
#             save_dir = Path(save_path).parent
#             save_dir.mkdir(parents=True, exist_ok=True)
#             plt.savefig(save_path, bbox_inches="tight", dpi=150) # Added dpi
#             self.logger.info(f"Leadfield summary visualization saved to {save_path}")
#         if show:
#             plt.show()

#     except Exception as e:
#          self.logger.error(f"Failed during leadfield summary visualization: {e}", exc_info=True) # Added exc_info
#     finally:
#          if fig:
#              plt.close(fig)


# def visualize_leadfield_sensor_boxplot(
#     self,
#     leadfield_matrix: np.ndarray,
#     orientation_type: str = "fixed",
#     sensor_indices_to_plot: Optional[List[int]] = None,
#     max_sensors_to_plot: int = 20,
#     save_path: Optional[str] = None,
#     custom_title: Optional[str] = None,
#     show: bool = False
# ) -> None:
#     """
#     Visualize the distribution of leadfield amplitudes for selected sensors using box plots.
#     Each box plot represents one sensor, showing the distribution of its leadfield
#     values across all sources. For 'free' orientation, the norm of the 3 components
#     is used for each source-sensor pair.

#     Parameters
#     ----------
#     leadfield_matrix : np.ndarray
#         The leadfield matrix.
#         - 'fixed': Shape (n_sensors, n_sources).
#         - 'free': Shape (n_sensors, n_sources, 3).
#     orientation_type : str, optional
#         Orientation type ('fixed' or 'free'), by default "fixed".
#     sensor_indices_to_plot : Optional[List[int]], optional
#         Specific list of sensor indices to plot. If None, a subset is chosen
#         based on max_sensors_to_plot, by default None.
#     max_sensors_to_plot : int, optional
#         Maximum number of sensors to create box plots for if sensor_indices_to_plot
#         is None, by default 20.
#     save_path : Optional[str], optional
#         Path to save the figure. If None, not saved, by default None.
#     custom_title : Optional[str], optional
#         Custom title for the plot. If None, a default title is generated.
#     show : bool, optional
#         If True, display the plot, by default False.
#     """
#     if leadfield_matrix is None or not isinstance(leadfield_matrix, np.ndarray) or leadfield_matrix.size == 0:
#         self.logger.error("Invalid leadfield matrix provided for box plot visualization.")
#         return

#     fig = None # Initialize fig
#     try:
#         num_total_sensors = leadfield_matrix.shape[0]

#         if sensor_indices_to_plot is None:
#             if num_total_sensors > max_sensors_to_plot:
#                 # Select evenly spaced sensors
#                 selected_indices = np.linspace(0, num_total_sensors - 1, max_sensors_to_plot, dtype=int)
#                 self.logger.info(f"Plotting box plots for {max_sensors_to_plot} selected sensors out of {num_total_sensors}.")
#             else:
#                 selected_indices = np.arange(num_total_sensors)
#         else:
#             selected_indices = np.array(sensor_indices_to_plot, dtype=int)
#             if np.any(selected_indices < 0) or np.any(selected_indices >= num_total_sensors):
#                 self.logger.error("Invalid sensor_indices_to_plot: indices out of bounds.")
#                 return
        
#         if len(selected_indices) == 0:
#             self.logger.info("No sensors selected for box plot visualization.")
#             return

#         data_for_boxplot = []
#         labels_for_boxplot = []

#         for sensor_idx in selected_indices:
#             if orientation_type == "fixed":
#                 if leadfield_matrix.ndim != 2:
#                     raise ValueError(f"Expected 2D leadfield for fixed orientation, got {leadfield_matrix.ndim}D shape {leadfield_matrix.shape}")
#                 sensor_values = leadfield_matrix[sensor_idx, :]
#             elif orientation_type == "free":
#                 if leadfield_matrix.ndim != 3 or leadfield_matrix.shape[-1] != 3:
#                     raise ValueError(f"Expected 3D leadfield (..., 3) for free orientation, got shape {leadfield_matrix.shape}")
#                 sensor_values_3d = leadfield_matrix[sensor_idx, :, :] # Shape (n_sources, 3)
#                 sensor_values = np.linalg.norm(sensor_values_3d, axis=-1) # Shape (n_sources,)
#             else:
#                 raise ValueError(f"Invalid orientation_type '{orientation_type}'. Choose 'fixed' or 'free'.")
            
#             data_for_boxplot.append(sensor_values)
#             labels_for_boxplot.append(str(sensor_idx))

#         # Adjust figure width based on the number of boxplots, with a max width
#         fig_width = min(max(10, len(selected_indices) * 0.7), 25)
#         fig, ax = plt.subplots(figsize=(fig_width, 7))
        
#         bp = ax.boxplot(data_for_boxplot, patch_artist=True, medianprops=dict(color="black", linewidth=1.5))

#         # Optional: Color the boxes using a colormap
#         # Ensure you have `import matplotlib.cm as cm`
#         try:
#             colors_list = cm.get_cmap('viridis', len(data_for_boxplot))
#             for i, patch in enumerate(bp['boxes']):
#                 patch.set_facecolor(colors_list(i / len(data_for_boxplot))) # Normalize index for colormap
#         except AttributeError: # Fallback if get_cmap with number of colors is not supported (older matplotlib)
#              self.logger.warning("Could not apply distinct colors to boxplots; using default or single color.")


#         if custom_title is None:
#             default_title = f"Leadfield Amplitude Distribution per Sensor ({orientation_type.capitalize()} Orientation)"
#             ax.set_title(default_title, fontsize=14, pad=15)
#         else:
#             ax.set_title(custom_title, fontsize=14, pad=15)

#         ax.set_xlabel("Sensor Index", fontsize=12)
#         ax.set_ylabel("Leadfield Amplitude (µV / nAm)", fontsize=12)
#         ax.set_xticklabels(labels_for_boxplot, rotation=45, ha="right" if len(labels_for_boxplot) > 10 else "center")
#         ax.grid(True, linestyle='--', alpha=0.6, axis='y')

#         plt.tight_layout()

#         if save_path:
#             save_dir = Path(save_path).parent
#             save_dir.mkdir(parents=True, exist_ok=True)
#             plt.savefig(save_path, bbox_inches="tight")
#             self.logger.info(f"Leadfield sensor box plot visualization saved to {save_path}")
#         if show:
#             plt.show()

#     except Exception as e:
#          self.logger.error(f"Failed during leadfield sensor box plot visualization: {e}")
#     finally:
#          if fig:
#              plt.close(fig)
        
        
