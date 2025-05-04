import mne
from mne.datasets import sample
from mne.io.constants import FIFF
import numpy as np

data_path = sample.data_path()
sample_dir = data_path / "MEG" / "sample"
fwd_name = sample_dir / "sample_audvis-eeg-oct-6-fwd.fif"

fwd = mne.read_forward_solution(fwd_name)
# fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True)
leadfield = fwd["sol"]["data"]

print("*"*40)
if fwd["source_ori"] == FIFF.FIFFV_MNE_FREE_ORI:
    print("Extracted free orientation leadfield (raw shape: %s)", leadfield.shape)
    n_sensors, n_sources_x_orient = leadfield.shape
    n_orient = 3
    if n_sources_x_orient % n_orient != 0:
        raise ValueError(f"Cannot reshape free orientation leadfield. Shape {leadfield.shape} not divisible by 3.")
    n_sources = n_sources_x_orient // n_orient
    leadfield = leadfield.reshape(n_sensors, n_sources, n_orient)
            
print(leadfield.shape)

# --- Inspect Leadfield Matrix Values ---
min_lf = np.min(leadfield)
max_lf = np.max(leadfield)
mean_abs_lf = np.mean(np.abs(leadfield))
std_lf = np.std(leadfield)

# -- Check for extreme values
print(f"Leadfield matrix mean: {np.mean(leadfield):.1e}, std: {std_lf:.2e}")
print(f"Leadfield matrix min: {min_lf:.1e}, max: {max_lf:.1e}")
print(f"Leadfield matrix std: {std_lf:.1e}")
print(f"Leadfield matrix mean abs: {mean_abs_lf:.1e}")
print("*"*40)