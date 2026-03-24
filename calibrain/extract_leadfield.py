from logging import warning
import os
import mne
import numpy as np
from mne.io.constants import FIFF

from calibrain.utils import get_data_path

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
    
    elif orientation_type == 'free' and coil_name == 'eeg':
        if fwd["source_ori"] == FIFF.FIFFV_MNE_FREE_ORI:
            print("Forward solution orientation is already free")
        elif fwd["source_ori"] == FIFF.FIFFV_MNE_FIXED_ORI:
            print("Forward solution orientation is fixed, converting to free orientation")
            fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=False)
            assert fwd["source_ori"] == FIFF.FIFFV_MNE_FREE_ORI, "Failed to convert to free orientation"
    
    elif orientation_type == 'free' and coil_name in ['mag', 'grad']:
        raise ValueError("Free orientation is not supported for MEG data.")
    
    # Filter channels by type
    if coil_name == 'mag':
        mag_channels = [ch['ch_name'] for ch in fwd['info']['chs'] if ch['unit'] == FIFF.FIFF_UNIT_T] # 112 FIFF_UNIT_T for magnetometers (tesla)
        coil_type = FIFF.FIFFV_COIL_VV_MAG_T1 # 3022
        
    elif coil_name == 'grad':
        mag_channels = [ch['ch_name'] for ch in fwd['info']['chs'] if ch['unit'] == FIFF.FIFF_UNIT_T_M] # 201 FIFF_UNIT_T_M for gradiometers (tesla per meter)
        coil_type = FIFF.FIFFV_COIL_VV_PLANAR_T1 # 3012
        
    elif coil_name == 'eeg': # 1 (FIFFV_COIL_EEG)
        mag_channels = [ch['ch_name'] for ch in fwd['info']['chs'] if ch['unit'] == FIFF.FIFF_UNIT_V] # 107 FIFF_UNIT_V for EEG (volts)
        coil_type = FIFF.FIFFV_COIL_EEG # 1
        
    else:
        raise ValueError(f"Unknown channel type: {coil_name}")

    # slice all channels with unit equal to 112 (FIFF_UNIT_T)
    fwd = fwd.pick_channels(mag_channels)


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
        'info': fwd['info'],
        'orientation_type': orientation_type,
        'coil_type': coil_type,
        'sensor_kind': first_ch.get('kind'),
        'sensor_units': first_ch.get('unit'),
        'sensor_unitmult': first_ch.get('unit_mul'),
    }
    file_name = f"lead_field_{orientation_type}_{subject}.npz"
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