import mne
from mne.datasets import sample
from mne.io.constants import FIFF
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_leadfield(
    leadfield_matrix: np.ndarray,
    orientation_type: str = "fixed",
    save_path = None,
    show: bool = True
) -> None:
    """
    Visualize the leadfield matrix as a heatmap.

    Parameters
    ----------
    leadfield_matrix : np.ndarray
        The leadfield matrix.
        - 'fixed': Shape (n_sensors, n_sources).
        - 'free': Shape (n_sensors, n_sources, 3).
    orientation_type : str, optional
        Orientation type ('fixed' or 'free'), by default "fixed".
    save_path : Optional[str], optional
        Path to save the figure. If None, not saved, by default None.
    show : bool, optional
        If True, display the plot, by default False.

    Raises
    ------
    ValueError
        If leadfield_matrix is invalid or orientation_type is unsupported.
    """

    fig = None # Initialize fig

    if orientation_type == "fixed":
        if leadfield_matrix.ndim != 2:
                raise ValueError(f"Expected 2D leadfield for fixed orientation, got {leadfield_matrix.ndim}D")
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(leadfield_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
        fig.colorbar(im, ax=ax, label="Amplitude")
        ax.set_title("Leadfield Matrix (Fixed Orientation)")
        ax.set_xlabel("Sources")
        ax.set_ylabel("Sensors")
    elif orientation_type == "free":
        if leadfield_matrix.ndim != 3 or leadfield_matrix.shape[-1] != 3:
                raise ValueError(f"Expected 3D leadfield (..., 3) for free orientation, got shape {leadfield_matrix.shape}")
        n_orient = leadfield_matrix.shape[-1]
        fig, axes = plt.subplots(1, n_orient, figsize=(15, 5), sharey=True)
        if n_orient == 1: axes = [axes] # Ensure axes is iterable
        orientations = ["X", "Y", "Z"]
        images = []
        for i in range(n_orient):
            im = axes[i].imshow(leadfield_matrix[:, :, i], aspect='auto', cmap='viridis', interpolation='nearest')
            images.append(im)
            axes[i].set_title(f"Leadfield Matrix ({orientations[i]})")
            axes[i].set_xlabel("Sources")
        axes[0].set_ylabel("Sensors")
        fig.colorbar(images[0], ax=axes, location="right", label="Amplitude (Sensitivity)", fraction=0.05, pad=0.04)
    else:
        raise ValueError("Invalid orientation type. Must be 'fixed' or 'free'.")

    plt.tight_layout()

    if save_path:
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()

        # plt.close(fig)
                
                
# import mne
# from mne.datasets import sample

# data_path = sample.data_path()

# # the raw file containing the channel location + types
# sample_dir = data_path / "MEG" / "sample"
# raw_fname = sample_dir / "sample_audvis_raw.fif"
# # The paths to Freesurfer reconstructions
# subjects_dir = data_path / "subjects"
# subject = "sample"

# trans = sample_dir / "sample_audvis_raw-trans.fif"
# info = mne.io.read_info(raw_fname)

# src = mne.setup_source_space(
#     subject, spacing="oct4", add_dist="patch", subjects_dir=subjects_dir
# )

# sphere = (0.0, 0.0, 0.04, 0.09)
# vol_src = mne.setup_volume_source_space(
#     subject,
#     subjects_dir=subjects_dir,
#     sphere=sphere,
#     sphere_units="m",
#     add_interpolator=False,
# )  # just for speed!
# print(vol_src)
# print(vol_src)


# conductivity = (0.3,)  # for single layer
# # conductivity = (0.3, 0.006, 0.3)  # for three layers
# model = mne.make_bem_model(
#     subject="sample", ico=4, conductivity=conductivity, subjects_dir=subjects_dir
# )
# bem = mne.make_bem_solution(model)


# fwd = mne.make_forward_solution(
#     raw_fname,
#     trans=trans,
#     src=src,
#     bem=bem,
#     meg=False,
#     eeg=True,
#     mindist=5.0,
#     n_jobs=None,
#     verbose=True,
# )
# print(fwd)

# leadfield = fwd["sol"]["data"]
# print(f"Leadfield size : {leadfield.shape[0]} sensors x {leadfield.shape[1]} dipoles")



# print("*"*40)
# if fwd["source_ori"] == FIFF.FIFFV_MNE_FREE_ORI:
#     print("Extracted free orientation leadfield (raw shape: %s)", leadfield.shape)
#     n_sensors, n_sources_x_orient = leadfield.shape
#     n_orient = 3
#     if n_sources_x_orient % n_orient != 0:
#         raise ValueError(f"Cannot reshape free orientation leadfield. Shape {leadfield.shape} not divisible by 3.")
#     n_sources = n_sources_x_orient // n_orient
#     leadfield = leadfield.reshape(n_sensors, n_sources, n_orient)
            
            
            
            
            
            
            
import matplotlib.pyplot as plt
import numpy as np

import mne
from mne.datasets import sample
from mne.source_estimate import SourceEstimate
from mne.source_space import compute_distance_to_sensors

print(__doc__)

data_path = sample.data_path()
meg_path = data_path / "MEG" / "sample"
fwd_fname = meg_path / "sample_audvis-meg-eeg-oct-6-fwd.fif"
subjects_dir = data_path / "subjects"

# Read the forward solutions with surface orientation
from mne.io.constants import FIFF 

# ... (assuming fwd_fname is defined as in your context)
# fwd_fname = meg_path / "sample_audvis-meg-eeg-oct-6-fwd.fif"

fwd = mne.read_forward_solution(fwd_fname)
print(f"Original fwd['source_ori']: {fwd['source_ori']}")

# Convert to surface orientation if it's free orientation.
# This modifies fwd in-place.
# If already fixed or surface, it might not change much but ensures surface alignment.
mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, copy=False)
print(f"Converted fwd['source_ori']: {fwd['source_ori']}")

# Pick EEG channels from the (potentially orientation-converted) forward solution
picks_eeg_indices = mne.pick_types(fwd["info"], meg=False, eeg=True, exclude=[]) # Get indices of EEG channels

if len(picks_eeg_indices) == 0:
    print("No EEG channels found in the forward solution.")
    leadfield_eeg = None
else:
    # Create a new forward solution containing only the EEG channels
    fwd_eeg = mne.pick_channels_forward(fwd, include=[fwd["info"]["ch_names"][i] for i in picks_eeg_indices], ordered=True)
    print(f"Initial fwd_eeg['source_ori']: {fwd_eeg['source_ori']}")

    # --- Convert EEG forward solution to FIXED orientation ---
    print("Converting EEG forward solution to fixed orientation...")
    mne.convert_forward_solution(fwd_eeg, surf_ori=False, force_fixed=True, copy=False) # Modifies fwd_eeg in-place
    print(f"Converted fwd_eeg['source_ori'] (should be fixed): {fwd_eeg['source_ori']}")

    # Extract the leadfield data for EEG channels
    # After force_fixed=True, this should be 2D: (n_eeg_channels, n_sources)
    leadfield_eeg = fwd_eeg["sol"]["data"]
    print(f"EEG Leadfield (fixed orientation) shape: {leadfield_eeg.shape}")

    # The reshaping logic for free/surface orientation is no longer needed here
    # as we have forced a fixed orientation.

# The variable 'leadfield_eeg' now holds the EEG-specific leadfield matrix
# in fixed orientation: (n_eeg_channels, n_sources).

# To verify, you can print the number of channels in fwd_eeg info
if leadfield_eeg is not None:
    print(f"Number of channels in fwd_eeg: {fwd_eeg['nchan']}")
    print(f"Number of sources in fwd_eeg: {fwd_eeg['nsource']}")

print(leadfield_eeg.shape)

# --- Inspect Leadfield Matrix Values ---
min_lf = np.min(leadfield_eeg) # Use leadfield_eeg here
max_lf = np.max(leadfield_eeg) # Use leadfield_eeg here
mean_abs_lf = np.mean(np.abs(leadfield_eeg)) # Use leadfield_eeg here
std_lf = np.std(leadfield_eeg) # Use leadfield_eeg here

# -- Check for extreme values
print(f"Leadfield matrix mean: {np.mean(leadfield_eeg)}, std: {std_lf}") # Use leadfield_eeg
print(f"Leadfield matrix min: {min_lf}, max: {max_lf}")
print(f"Leadfield matrix std: {std_lf}")
print(f"Leadfield matrix mean abs: {mean_abs_lf}")
print("*"*40)


visualize_leadfield(
    leadfield_eeg,
    orientation_type="fixed",  # Change to "fixed" if needed
    show=True  # Set to False if you don't want to display the plot
)


print("------------------------")

leadfield = np.load("/Users/orabe/0.braindata/CaliBrain/results/forward/fsaverage-leadfield-fixed.npz")['leadfield']

print(f"Leadfield shape: {leadfield.shape}")


# --- Inspect Leadfield Matrix Values ---
min_lf = np.min(leadfield) # Use leadfield_eeg here
max_lf = np.max(leadfield) # Use leadfield_eeg here
mean_abs_lf = np.mean(np.abs(leadfield)) # Use leadfield_eeg here
std_lf = np.std(leadfield) # Use leadfield_eeg here

# -- Check for extreme values
print(f"Leadfield matrix mean: {np.mean(leadfield)}, std: {std_lf}")
print(f"Leadfield matrix min: {min_lf}, max: {max_lf}")
print(f"Leadfield matrix std: {std_lf}")
print(f"Leadfield matrix mean abs: {mean_abs_lf}")
print("*"*40)

visualize_leadfield(
    leadfield,
    orientation_type="fixed",  # Change to "fixed" if needed
    show=True  # Set to False if you don't want to display the plot
)



print("------------------------")








# leadfield = np.load("/Users/orabe/0.braindata/CaliBrain/BSI-ZOO_forward_data/lead_field_CC120166.npz")
# leadfield = leadfield['lead_field'] 
# print(f"Leadfield shape: {leadfield.shape}")

fwd_fname = "/Users/orabe/Desktop/delme22/BSI-Zoo/bsi_zoo/tests/data/CC120313-fwd.fif"
fwd = mne.read_forward_solution(fwd_fname)
print(f"Original fwd['source_ori']: {fwd['source_ori']}")

for c in fwd['info']['ch_names']:
    print(c)

# Convert to surface orientation if it's free orientation.
# This modifies fwd in-place.
# If already fixed or surface, it might not change much but ensures surface alignment.
mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, copy=False)
print(f"Converted fwd['source_ori']: {fwd['source_ori']}")

# Pick EEG channels from the (potentially orientation-converted) forward solution
picks_eeg_indices = mne.pick_types(fwd["info"], meg=False, eeg=True, exclude=[]) # Get indices of EEG channels

# Create a new forward solution containing only the EEG channels
fwd_eeg = mne.pick_channels_forward(fwd, include=[fwd["info"]["ch_names"][i] for i in picks_eeg_indices], ordered=True)
print(f"Initial fwd_eeg['source_ori']: {fwd_eeg['source_ori']}")

# --- Convert EEG forward solution to FIXED orientation ---
print("Converting EEG forward solution to fixed orientation...")
mne.convert_forward_solution(fwd_eeg, surf_ori=False, force_fixed=True, copy=False) # Modifies fwd_eeg in-place
print(f"Converted fwd_eeg['source_ori'] (should be fixed): {fwd_eeg['source_ori']}")

# Extract the leadfield data for EEG channels
# After force_fixed=True, this should be 2D: (n_eeg_channels, n_sources)
leadfield_eeg = fwd_eeg["sol"]["data"]
print(f"EEG Leadfield (fixed orientation) shape: {leadfield_eeg.shape}")

    # The reshaping logic for free/surface orientation is no longer needed here
    # as we have forced a fixed orientation.

# The variable 'leadfield_eeg' now holds the EEG-specific leadfield matrix
# in fixed orientation: (n_eeg_channels, n_sources).

# To verify, you can print the number of channels in fwd_eeg info

print(f"Number of channels in fwd_eeg: {fwd_eeg['nchan']}")
print(f"Number of sources in fwd_eeg: {fwd_eeg['nsource']}")

print(leadfield_eeg.shape)


# --- Inspect Leadfield Matrix Values ---
min_lf = np.min(leadfield) # Use leadfield_eeg here
max_lf = np.max(leadfield) # Use leadfield_eeg here
mean_abs_lf = np.mean(np.abs(leadfield)) # Use leadfield_eeg here
std_lf = np.std(leadfield) # Use leadfield_eeg here

# -- Check for extreme values
print(f"Leadfield matrix mean: {np.mean(leadfield)}, std: {std_lf}")
print(f"Leadfield matrix min: {min_lf}, max: {max_lf}")
print(f"Leadfield matrix std: {std_lf}")
print(f"Leadfield matrix mean abs: {mean_abs_lf}")
print("*"*40)

visualize_leadfield(
    leadfield,
    orientation_type="fixed",  # Change to "fixed" if needed
    show=True  # Set to False if you don't want to display the plot
)




# --------


fwd_fname = "/Users/orabe/0.braindata/CaliBrain/results/forward/fsaverage-fixed-fwd.fif"
fwd = mne.read_forward_solution(fwd_fname)
print(f"Original fwd['source_ori']: {fwd['source_ori']}")