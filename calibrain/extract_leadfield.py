from logging import warning
import os
import mne
import numpy as np
from mne.io.constants import FIFF

from calibrain.utils import get_data_path, get_fwd_dir, get_leadfield_dir, strip_local_paths

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

    `cortical_normals` has shape (sources, 3) and holds the unit cortical normals
    that MNE uses to build the fixed-orientation leadfield (``source_nn`` of the
    fixed-converted forward, i.e. cortical-patch-statistics normals). They are
    expressed in the same 3D source basis as the *free* leadfield columns (before
    any MEG SVD reduction) and follow the same source order. The free-orientation
    source simulation needs them to generate activity along the normal.
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
            # A fixed-only forward does not retain the missing 3D components.
            raise ValueError(
                "Free extraction requires an original free-orientation forward. "
                "A fixed-only forward cannot recover the missing 3D components."
            )
            
              
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

    # Cortical normals in the original (pre-reduction) 3D source basis. The free
    # forward is stored with surf_ori=False, so its columns are in the head x/y/z
    # frame; MNE's fixed conversion yields the matching unit normals in that frame.
    if orientation_type == 'fixed':
        cortical_normals = np.array(fwd['source_nn'], dtype=float, copy=True)
    else:
        # Cortical-normal free simulation requires a surface source space.
        if any(src["type"] != "surf" for src in fwd["src"]):
            raise ValueError(
                "Cortical-normal free simulation requires surface source spaces."
            )
        fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, verbose='error')
        cortical_normals = np.array(fwd_fixed['source_nn'], dtype=float, copy=True)
    if cortical_normals.shape != (fwd['nsource'], 3):
        raise ValueError(f"Unexpected cortical normals shape {cortical_normals.shape}")

    # Check lengths before division to avoid zero or invalid normalization.
    normal_lengths = np.linalg.norm(cortical_normals, axis=1, keepdims=True)
    if not np.all(np.isfinite(normal_lengths)) or np.any(normal_lengths < 1e-12):
        raise ValueError("Each cortical normal must have a finite, nonzero norm.")
    cortical_normals /= np.linalg.norm(cortical_normals, axis=1, keepdims=True)

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
            
            
    # Do not store local paths in the saved file: they would leak user and machine
    # names into every file that is shared.
    strip_local_paths(fwd["info"])
    
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
        'cortical_normals': cortical_normals,  # (sources, 3), unit normals in the free 3D source basis.
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


# Channel type of the example subjects: the template has EEG, the CamCAN data MEG
# magnetometers. Other subjects need an explicit ``coil_name``.
DEFAULT_COILS = {
    "fsaverage": "eeg",
    "CC120166": "mag",
    "CC120264": "mag",
    "CC120309": "mag",
    "CC120313": "mag",
}


def extract_leadfields(subjects=None, data_dir=None, coil_name=None, orientations=("fixed", "free")):
    """Extract the leadfield files of several subjects from their forward solutions.

    This is the step after :func:`calibrain.datasets.fetch_example_data`: it reads
    ``<data_dir>/1284src_fwd/<subject>-fwd.fif`` and writes
    ``<data_dir>/1284src_leadfield/<subject>_<orientation>_leadfield.npz``. The
    leadfield files depend on the version of CaliBrain, so they are created locally
    and not downloaded.

    Parameters
    ----------
    subjects : str | list of str | None
        Subjects to process. By default all subjects that have a forward solution
        (``*-fwd.fif``) in ``<data_dir>/1284src_fwd``.
    data_dir : str | Path | None
        Data directory. By default the CaliBrain data directory
        (see :func:`calibrain.utils.get_data_path`).
    coil_name : {"eeg", "mag", "grad"} | None
        Channel type. By default it is taken from ``DEFAULT_COILS`` (``"eeg"`` for
        ``fsaverage``, ``"mag"`` for the CamCAN subjects); other subjects need it.
    orientations : sequence of {"fixed", "free"}
        Orientation types to extract.

    Returns
    -------
    saved : list of pathlib.Path
        The leadfield files that were written.

    Raises
    ------
    FileNotFoundError
        If there is no forward solution to process.
    ValueError
        If the channel type of a subject is unknown and ``coil_name`` is not given.
    """
    fwd_dir = get_fwd_dir(data_dir)
    save_dir = get_leadfield_dir(data_dir)

    if subjects is None:
        subjects = sorted(path.name[: -len("-fwd.fif")] for path in fwd_dir.glob("*-fwd.fif"))
        if not subjects:
            raise FileNotFoundError(
                f"No forward solutions (*-fwd.fif) found in {fwd_dir}. Download the example "
                "data with calibrain.fetch_example_data() and reduce it with "
                "calibrain.reduce_forward_solutions()."
            )
    elif isinstance(subjects, str):
        subjects = [subjects]

    saved = []
    for subject in subjects:
        coil = coil_name or DEFAULT_COILS.get(subject)
        if coil is None:
            raise ValueError(
                f"The channel type of subject {subject!r} is unknown. Pass coil_name='eeg', 'mag' or 'grad'."
            )
        if not (fwd_dir / f"{subject}-fwd.fif").is_file():
            raise FileNotFoundError(f"No forward solution for subject {subject!r} in {fwd_dir}.")
        for orientation in orientations:
            extract_leadfield(fwd_dir, subject, coil, orientation, save_dir)
            saved.append(save_dir / f"{subject}_{orientation}_leadfield.npz")
    return saved


def main(argv=None):
    """Extract the leadfield files from the forward solutions in the data directory.

    Run with ``python -m calibrain.extract_leadfield``. By default it processes every
    subject that has a forward solution in ``<data>/1284src_fwd`` and writes
    ``<data>/1284src_leadfield``.
    """
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", help="Data directory (default: the CaliBrain default).")
    parser.add_argument("--subjects", nargs="+", help="Subjects to process (default: all with a forward solution).")
    parser.add_argument("--coil", choices=["eeg", "mag", "grad"], help="Channel type (default: from the subject).")
    args = parser.parse_args(argv)

    saved = extract_leadfields(subjects=args.subjects, data_dir=args.data_dir, coil_name=args.coil)
    print("\nWritten leadfield files:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
