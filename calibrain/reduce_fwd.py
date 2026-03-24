# from calibrain.utils import restrict_fwd_to_sources, get_data_path
from calibrain.utils import get_data_path
import mne
import numpy as np
import os
from pathlib import Path

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


n_keep_per_hemi = 642

fwd_datapath = get_data_path() / 'fwd'
save_path = Path(get_data_path() / f'{2*n_keep_per_hemi}src_fwd')
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
            n_keep=n_keep_per_hemi,
            seed=42,
        )
        print(f"Number of sources after reduction: {fwd_subset['nsource']}")
        
        out_path = save_path / f"{subject}-fwd.fif"
        mne.write_forward_solution(str(out_path), fwd_subset)
