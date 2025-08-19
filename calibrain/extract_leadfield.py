import mne
import numpy as np
from mne.io.constants import FIFF


def extract_leadfield(fwd_datapath, subject: str, channel_type: str, orientation_type: str, save_path: str):
    """Extracts the leadfield from a forward solution file for a given subject and channel type. The leadfield is saved as a .npz file. For MEG data, it filters channels by unit type (magnetometers (T) or gradiometers (T/m)). For EEG data, it uses the default unit (V).

    Parameters
    ----------
    fwd_datapath : str
        Path to the directory containing the forward solution files.
    subject : str
        Subject identifier, used to load the specific forward solution file.
    channel_type : str
        Type of channels to filter in the forward solution. Options are 'meg' for magnetometers, 'grad' for gradiometers, or 'eeg' for EEG channels.
    orientation_type : str, optional
        Orientation type of the leadfield. Options are 'fixed' or 'free'. Default is 'fixed'.
    """
    fwd_path = f"{fwd_datapath}/{subject}-fwd.fif"
    print(f"Loading forward solution from {fwd_path}")
    
    fwd = mne.read_forward_solution(fwd_path)
    print(f"\nNumber of channels before filtering: {len(fwd['info']['chs'])}")

    if orientation_type == 'fixed':        
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
    if channel_type == 'meg':
        mag_channels = [ch['ch_name'] for ch in fwd['info']['chs'] if ch['unit'] == FIFF.FIFF_UNIT_T] # 112 FIFF_UNIT_T for magnetometers (tesla)
    elif channel_type == 'grad':
        mag_channels = [ch['ch_name'] for ch in fwd['info']['chs'] if ch['unit'] == FIFF.FIFF_UNIT_T_M] # 201 FIFF_UNIT_T_M for gradiometers (tesla per meter)
    elif channel_type == 'eeg':
        mag_channels = [ch['ch_name'] for ch in fwd['info']['chs'] if ch['unit'] == FIFF.FIFF_UNIT_V] # 107 FIFF_UNIT_V for EEG (volts)
    else:
        raise ValueError(f"Unknown channel type: {channel_type}")

    print(f"Selecting {channel_type} channels with unit code {FIFF.FIFF_UNIT_T if channel_type == 'meg' else FIFF.FIFF_UNIT_T_M}")

    # slice all channels with unit equal to 112 (FIFF_UNIT_T)
    fwd = fwd.pick_channels(mag_channels)
    print(f"Number of channels after picking: {len(fwd['info']['chs'])}")

    if fwd['source_ori'] == FIFF.FIFFV_MNE_FIXED_ORI:
        print(f"Leadfield matrix shape (fixed orientation): {fwd['sol']['data'].shape} (channels, sources)")
    elif fwd['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI:
        print(f"Leadfield matrix shape (free orientation): {fwd['sol']['data'].shape} (channels, sources*3)")

    # for i in range(len(fwd['info']['chs'])):
    #     print(i, '--',
    #         fwd['info']['chs'][i]['ch_name'], '--',
    #         fwd['info']['chs'][i]['kind'], '--',
    #         fwd['info']['chs'][i]['unit'],
    #     )
    
    # save the leadfield as npz file
    leadfield_data = {
        'leadfield': fwd['sol']['data'],
        'source_ori': fwd['source_ori'],
        'info': fwd['info'],
        'channel_type': channel_type,
        'orientation_type': orientation_type,
        'units': fwd['info']['chs'][0]['unit']
    }
    file_name = f"lead_field_{orientation_type}_{subject}.npz"
    if save_path is None:
        save_path = f"{fwd_datapath}/{file_name}"
    else:
        save_path = f"{save_path}/{file_name}"
    np.savez(save_path, **leadfield_data)
    print(f"Leadfield saved to {save_path}")


fwd_datapath_meg = 'BSI-ZOO_forward_data'
subjects_meg = ['CC120166', 'CC120264', 'CC120309', 'CC120313']

fwd_datapath_eeg = 'results/forward'
subjects_eeg = ['fsaverage']

fwd_datapath = fwd_datapath_eeg
subjects = subjects_eeg
save_path = fwd_datapath_meg

for subject in subjects:
    extract_leadfield(fwd_datapath, subject=subject, channel_type='meg', orientation_type='fixed', save_path=save_path)
    extract_leadfield(fwd_datapath, subject=subject, channel_type='meg', orientation_type='free', save_path=save_path)
