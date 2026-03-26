from logging import warning
import os
import mne
import numpy as np
from mne.io.constants import FIFF

from calibrain.utils import get_data_path

def _canonicalize_basis_columns(Q: np.ndarray) -> np.ndarray:
    """
    Stabilize basis column signs so that the entry with largest absolute value
    in each column is nonnegative.

    This removes arbitrary SVD sign flips across runs/platforms while leaving
    the spanned subspace unchanged.
    Parameters
    ----------
    Q : array, shape (n, k)
        Basis matrix with k orthonormal columns, where n is the number of orientations and k is the reduced dimensionality (e.g. 2 for MEG free orientation).
    Returns
    -------
    Q_out : array, shape (n, k)
        Basis matrix with signs stabilized.
    """
    if Q.ndim != 2:
        raise ValueError(f"Expected 2D basis matrix. Got {Q.shape}")

    Q_out = np.array(Q, dtype=float, copy=True)
    for j in range(Q_out.shape[1]):
        idx = int(np.argmax(np.abs(Q_out[:, j])))
        if Q_out[idx, j] < 0:
            Q_out[:, j] *= -1.0
    return Q_out

def extract_leadfield(fwd_datapath, subject: str, coil_name: str, orientation_type: str, save_path: str):
    """Extracts the leadfield from a forward solution file for a given subject and channel type. The leadfield is saved as a .npz file. For MEG data, it filters channels by unit type (magnetometers (T) or gradiometers (T/m)). For EEG data, it uses the default unit (V).

    Parameters
    ----------
    fwd_datapath : str
        Path to the directory containing the forward solution files.
    subject : str
        Subject identifier, used to load the specific forward solution file.
    coil_name : str
        Type of channels to filter in the forward solution. Options are 'meg' for magnetometers, 'grad' for gradiometers, or 'eeg' for EEG channels.
    orientation_type : str, optional
        Orientation type of the leadfield. Options are 'fixed' or 'free'. Default is 'fixed'.
    save_path : str
        Path to save the extracted leadfield .npz file.   
    
    Notes
    -----
    `Q_basis` is:
    - identity for free_eeg, as an API-consistent placeholder in the retained 3D basis
    - the local 3x2 SVD basis for free_meg
    """
    fwd_path = f"{fwd_datapath}/{subject}-fwd.fif"
    print(f"Loading forward solution from {fwd_path}")
    
    fwd = mne.read_forward_solution(fwd_path, verbose='error')
    print(f"Number of sources: {fwd['nsource']}")
    print(f"Number of channels: {len(fwd['info']['chs'])}")
            
    if orientation_type == 'fixed':        # EEG, MEG
        if fwd["source_ori"] == FIFF.FIFFV_MNE_FIXED_ORI:
            print("Forward solution orientation is already fixed")
        elif fwd["source_ori"] == FIFF.FIFFV_MNE_FREE_ORI:
            print("Forward solution orientation is free, converting to fixed orientation")
            fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True)
            assert fwd["source_ori"] == FIFF.FIFFV_MNE_FIXED_ORI, "Failed to convert to fixed orientation"
    elif orientation_type == 'free':
        if fwd["source_ori"] == FIFF.FIFFV_MNE_FREE_ORI:
            print("Forward solution orientation is already free")
        elif fwd["source_ori"] == FIFF.FIFFV_MNE_FIXED_ORI:
            print("Forward solution orientation is fixed, converting to free orientation")
            fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=False)
            assert fwd["source_ori"] == FIFF.FIFFV_MNE_FREE_ORI, "Failed to convert to free orientation"
              
    # Filter channels by type
    if coil_name == 'mag':
        mag_channels = [ch['ch_name'] for ch in fwd['info']['chs'] if ch['unit'] == FIFF.FIFF_UNIT_T] # 112 FIFF_UNIT_T for magnetometers (tesla)
        coil_type = FIFF.FIFFV_COIL_VV_MAG_T1 # 3022 (mag)
        
    elif coil_name == 'grad':
        mag_channels = [ch['ch_name'] for ch in fwd['info']['chs'] if ch['unit'] == FIFF.FIFF_UNIT_T_M] # 201 FIFF_UNIT_T_M for gradiometers (tesla per meter)
        coil_type = FIFF.FIFFV_COIL_VV_PLANAR_T1 # 3012 (grad)
        
    elif coil_name == 'eeg': # 1 (FIFFV_COIL_EEG)
        mag_channels = [ch['ch_name'] for ch in fwd['info']['chs'] if ch['unit'] == FIFF.FIFF_UNIT_V] # 107 FIFF_UNIT_V for EEG (volts)
        coil_type = FIFF.FIFFV_COIL_EEG # 1 (eeg)
        
    else:
        raise ValueError(f"Unknown channel type: {coil_name}")

    fwd = fwd.pick_channels(mag_channels)

    if orientation_type == 'fixed':
        Q_basis = None
    
    elif orientation_type == 'free':
        L = fwd['sol']['data']  # (M, 3N)
        N = fwd["nsource"]
        M = L.shape[0]
        L_block = L.reshape(M, N, 3)  # (M, N, 3)

        if coil_name in ['eeg']:
            Q_basis = np.repeat(np.eye(3)[None, :, :], fwd["nsource"], axis=0) # (sources, 3, 3)
            fwd['sol']['data'] = L_block
        
        elif coil_name in ['mag', 'grad']:
            # free MEG: reduce each local block Mx3 -> Mx2 via local SVD
        
            # initialize arrays to store the reduced basis and leadfield  
            Q_basis = np.zeros((fwd["nsource"], 3, 2))
            L_block_final = np.zeros((len(fwd["info"]["chs"]), fwd["nsource"], 2))
            
            for i in range(fwd["nsource"]):
                Li = L_block[:, i, :]          # (M, 3)
                _, _, Vt = np.linalg.svd(Li, full_matrices=False)
                Qi = _canonicalize_basis_columns(Vt[:2, :].T)   # (3, 2)
                Q_basis[i] = Qi
                L_block_final[:, i, :] = Li @ Qi      # (M, 2)

            # flatten L_block_final back to (M, 2N)
            L_flat = L_block_final.reshape(M, -1)  # (M, 2N) Not used
            
            fwd['sol']['data'] = L_block_final

    chs = fwd["info"]["chs"]
    first = chs[0]
    kind = first.get("kind")
    unit = first.get("unit")
    unit_mul = first.get("unit_mul")
    coil_type = first.get("coil_type")

    # Check for consistency across all channels (can be removed later)
    for ch in chs[1:]:
        if (
            kind is not None
            and ch.get("kind") is not None
            and ch["kind"] != kind
        ):
            warning("Info contains mixed channel kinds; using the first one (%s).", kind)
            break
        if (
            unit is not None
            and ch.get("unit") is not None
            and ch["unit"] != unit
        ):
            warning("Info contains mixed channel units; using the first one (%s).", unit)
            break
        if (
            unit_mul is not None
            and ch.get("unit_mul") is not None
            and ch["unit_mul"] != unit_mul
        ):
            warning("Info contains mixed channel unit multipliers; using the first one (%s).", unit_mul)
            break
        if (
            coil_type is not None
            and ch.get("coil_type") is not None
            and ch["coil_type"] != coil_type
        ):
            warning("Info contains mixed coil types; using the first one (%s).", coil_type)
            break
            
            
    # update working directory info to fwd['info']
    with fwd["info"]._unlock():
        fwd["info"]["working_dir"] = str(fwd_datapath)
    
    print(f"Number of channels after picking: {len(fwd['info']['chs'])}")

    if fwd['source_ori'] == FIFF.FIFFV_MNE_FIXED_ORI:
        print(f"Leadfield matrix shape (fixed orientation): {fwd['sol']['data'].shape} (channels, sources)")
    elif fwd['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI:
        print(f"Leadfield matrix shape (free orientation): {fwd['sol']['data'].shape} (channels, sources*3)")
        
    # save the leadfield as npz file
    first_ch = fwd['info']['chs'][0]
    leadfield_data = {
        'subject': subject,
        'leadfield': fwd['sol']['data'],
        'source_ori': fwd['source_ori'],
        'src_coords': fwd['source_rr'],
        'info': fwd['info'],
        'orientation_type': orientation_type,
        'coil_type': coil_type,
        'sensor_kind': first_ch.get('kind'),
        'sensor_units': first_ch.get('unit'),
        'sensor_unitmult': first_ch.get('unit_mul'),
        'Q_basis': Q_basis,  # (sources, 3, 2) for free MEG, identity (3, 3) for free eeg, None for fixed (MEG and EEG).
    }
    file_name = f"{subject}_{orientation_type}_leadfield.npz"
    if save_path is None:
        save_path = get_data_path() / 'leadfield' / file_name
    else:
        save_path = save_path / file_name
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, **leadfield_data)
    print(f"Leadfield saved to {save_path}")

# full sources in fwd
# fwd_datapath = get_data_path() / 'fwd'
# save_path = get_data_path() / 'leadfield' 

# reduced sources in fwd
fwd_datapath = get_data_path() / '1284src_fwd'
save_path = get_data_path() / '1284src_leadfield' 

subjects_map = {
    'eeg': ['fsaverage'],
    'mag': ['CC120166', 'CC120264', 'CC120309', 'CC120313'],
}

for coil_name, subjects in subjects_map.items():
    for subject in subjects:
        print(f"\nExtracting leadfield (fixed & free) for subject: {subject}, coil_name: {coil_name}")
        extract_leadfield(fwd_datapath, subject=subject, coil_name=coil_name, orientation_type='fixed', save_path=save_path)
        extract_leadfield(fwd_datapath, subject=subject, coil_name=coil_name, orientation_type='free', save_path=save_path)