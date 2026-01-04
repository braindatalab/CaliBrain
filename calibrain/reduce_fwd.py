from calibrain.utils import restrict_fwd_to_sources, get_data_path
import mne
import numpy as np
import os
from pathlib import Path

hemi = "rh"
n_keep = 1284

fwd_datapath = get_data_path() / 'fwd'
save_path = Path(get_data_path() / f'{hemi}{n_keep}_fwd')
os.makedirs(save_path, exist_ok=True)

subjects_map = {
    'eeg': ['fsaverage'],
    'mag': ['CC120166', 'CC120264', 'CC120309', 'CC120313'],
}

for coil_name, subjects in subjects_map.items():
    for subject in subjects:
        print(f"\nReducing forward solution for subject: {subject}, coil_name: {coil_name}")

        fwd_path = f"{fwd_datapath}/{subject}-fwd.fif"
        print(f"Loading forward solution from {fwd_path}")

        fwd = mne.read_forward_solution(fwd_path, verbose='error')
        print(f"Number of sources before reduction: {fwd['nsource']}")
        print(f"Number of channels: {len(fwd['info']['chs'])}")
        
        fwd_subset = restrict_fwd_to_sources(
            fwd,
            n_keep=n_keep,
            hemi=hemi,
            seed=42,
        )
        print(f"Number of sources after reduction: {fwd_subset['nsource']}")
        
        out_path = save_path / f"{subject}-fwd.fif"
        mne.write_forward_solution(str(out_path), fwd_subset)
