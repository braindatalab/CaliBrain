"""Tests for calibrain.extract_leadfield.

Most tests extract leadfields from the small (1284-source) forward solutions in
``<data>/1284src_fwd``. They are skipped when those files are not available
(e.g. on a fresh clone or in CI) and always write to a temporary directory, never
to the real leadfield directory.
"""
import copy
from pathlib import Path

import mne
import numpy as np
import pytest
from mne.io.constants import FIFF

from calibrain.extract_leadfield import extract_leadfield, extract_leadfields, main

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FWD_DIR = _DATA_DIR / "1284src_fwd"

needs_fwd = pytest.mark.skipif(
    not (FWD_DIR / "fsaverage-fwd.fif").exists(),
    reason="1284src forward solutions not available",
)
needs_meg_fwd = pytest.mark.skipif(
    not (FWD_DIR / "CC120166-fwd.fif").exists(),
    reason="1284src MEG forward solution not available",
)


def _extract(tmp_path, subject, coil, orientation, fwd_dir=FWD_DIR):
    extract_leadfield(fwd_dir, subject, coil, orientation, tmp_path)
    with np.load(tmp_path / f"{subject}_{orientation}_leadfield.npz", allow_pickle=True) as f:
        return {k: f[k] for k in f.files if k != "info"}


def test_importing_the_module_does_not_run_the_batch_extraction():
    import calibrain.extract_leadfield as module

    assert not hasattr(module, "save_path")
    assert not hasattr(module, "subjects_map")


@needs_fwd
def test_free_eeg_extraction_saves_unit_normals_and_identity_basis(tmp_path):
    out = _extract(tmp_path, "fsaverage", "eeg", "free")

    n_sources = out["leadfield"].shape[1]
    assert out["leadfield"].shape[2] == 3
    assert out["cortical_normals"].shape == (n_sources, 3)
    np.testing.assert_allclose(np.linalg.norm(out["cortical_normals"], axis=1), 1.0)
    np.testing.assert_array_equal(out["Q_basis"], np.repeat(np.eye(3)[None], n_sources, axis=0))


@needs_fwd
def test_saved_leadfield_does_not_contain_a_local_path(tmp_path):
    extract_leadfield(FWD_DIR, "fsaverage", "eeg", "free", tmp_path)

    with np.load(tmp_path / "fsaverage_free_leadfield.npz", allow_pickle=True) as f:
        assert f["info"].item()["working_dir"] == ""
    # The forward directory must not appear anywhere in the saved file either.
    assert str(FWD_DIR).encode() not in (tmp_path / "fsaverage_free_leadfield.npz").read_bytes()


@needs_fwd
def test_fixed_eeg_extraction_saves_unit_normals_and_no_basis(tmp_path):
    out = _extract(tmp_path, "fsaverage", "eeg", "fixed")

    n_sources = out["leadfield"].shape[1]
    assert out["leadfield"].ndim == 2
    assert out["cortical_normals"].shape == (n_sources, 3)
    np.testing.assert_allclose(np.linalg.norm(out["cortical_normals"], axis=1), 1.0)
    assert out["Q_basis"].item() is None


@needs_fwd
def test_free_eeg_leadfield_along_normals_matches_fixed_leadfield(tmp_path):
    free = _extract(tmp_path / "free", "fsaverage", "eeg", "free")
    fixed = _extract(tmp_path / "fixed", "fsaverage", "eeg", "fixed")

    # L_free @ n must reproduce the fixed leadfield, so the stored normals
    # are consistent with the leadfield columns.
    along_normal = np.einsum("mnk,nk->mn", free["leadfield"], free["cortical_normals"])
    np.testing.assert_allclose(along_normal, fixed["leadfield"], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(free["cortical_normals"], fixed["cortical_normals"], atol=1e-9)


@needs_meg_fwd
@pytest.mark.parametrize("coil", ["mag", "grad"])
def test_free_meg_extraction_reduces_to_two_orthonormal_components(tmp_path, coil):
    out = _extract(tmp_path, "CC120166", coil, "free")

    n_sources = out["leadfield"].shape[1]
    assert out["leadfield"].shape[2] == 2
    assert out["Q_basis"].shape == (n_sources, 3, 2)
    gram = np.einsum("nki,nkj->nij", out["Q_basis"], out["Q_basis"])
    # The forward matrix is stored in float32, so orthonormality holds to ~1e-7.
    np.testing.assert_allclose(gram, np.broadcast_to(np.eye(2), gram.shape), atol=1e-6)
    assert out["cortical_normals"].shape == (n_sources, 3)


@needs_fwd
def test_free_extraction_from_fixed_only_forward_raises(tmp_path):
    free_fwd = mne.read_forward_solution(FWD_DIR / "fsaverage-fwd.fif", verbose="error")
    fixed_only = mne.convert_forward_solution(free_fwd, surf_ori=True, force_fixed=True, verbose="error")
    fixed_only = copy.deepcopy(fixed_only)
    # Drop MNE's memory of the original free matrix: a genuinely fixed-only forward.
    fixed_only["_orig_sol"] = fixed_only["sol"]["data"].copy()
    fixed_only["_orig_source_ori"] = FIFF.FIFFV_MNE_FIXED_ORI
    fixed_only["_orig_sol_names"] = list(fixed_only["sol"]["row_names"])
    fwd_dir = tmp_path / "fwd"
    fwd_dir.mkdir()
    mne.write_forward_solution(fwd_dir / "fixedonly-fwd.fif", fixed_only, verbose="error")

    with pytest.raises(ValueError, match="original free-orientation forward"):
        extract_leadfield(fwd_dir, "fixedonly", "eeg", "free", tmp_path)
    assert not (tmp_path / "fixedonly_free_leadfield.npz").exists()


@needs_fwd
def test_free_extraction_requires_surface_source_spaces(tmp_path, monkeypatch):
    real_read = mne.read_forward_solution

    def read_with_volume_source_space(*args, **kwargs):
        fwd = real_read(*args, **kwargs)
        fwd["src"][0]["type"] = "vol"
        return fwd

    monkeypatch.setattr(mne, "read_forward_solution", read_with_volume_source_space)

    with pytest.raises(ValueError, match="surface source spaces"):
        extract_leadfield(FWD_DIR, "fsaverage", "eeg", "free", tmp_path)


@needs_fwd
@pytest.mark.parametrize("orientation", ["fixed", "free"])
def test_zero_cortical_normal_raises(tmp_path, monkeypatch, orientation):
    real_convert = mne.convert_forward_solution

    def convert_with_zero_normal(*args, **kwargs):
        fwd = real_convert(*args, **kwargs)
        if kwargs.get("force_fixed"):
            fwd["source_nn"][0] = 0.0
        return fwd

    monkeypatch.setattr(mne, "convert_forward_solution", convert_with_zero_normal)

    with pytest.raises(ValueError, match="finite, nonzero"):
        extract_leadfield(FWD_DIR, "fsaverage", "eeg", orientation, tmp_path)


def test_importing_reduce_fwd_does_not_run_the_batch_reduction():
    import calibrain.reduce_fwd as module

    assert callable(module.restrict_fwd_to_sources)
    assert not hasattr(module, "subjects_map")
    assert not hasattr(module, "save_path")


def _data_dir_with_forward(tmp_path, subject="fsaverage"):
    """A data directory that holds only the forward solution of ``subject``."""
    fwd_dir = tmp_path / "1284src_fwd"
    fwd_dir.mkdir(parents=True)
    (fwd_dir / f"{subject}-fwd.fif").symlink_to(FWD_DIR / f"{subject}-fwd.fif")
    return tmp_path


@needs_fwd
def test_extract_leadfields_processes_the_forwards_that_exist(tmp_path):
    data_dir = _data_dir_with_forward(tmp_path)

    saved = extract_leadfields(data_dir=data_dir)

    assert saved == [
        data_dir / "1284src_leadfield" / "fsaverage_fixed_leadfield.npz",
        data_dir / "1284src_leadfield" / "fsaverage_free_leadfield.npz",
    ]
    assert all(path.is_file() for path in saved)


@needs_fwd
def test_extract_leadfields_can_extract_one_orientation(tmp_path):
    data_dir = _data_dir_with_forward(tmp_path)

    saved = extract_leadfields("fsaverage", data_dir=data_dir, orientations=("free",))

    assert [path.name for path in saved] == ["fsaverage_free_leadfield.npz"]


@needs_fwd
def test_command_line_extracts_the_leadfields(tmp_path, capsys):
    data_dir = _data_dir_with_forward(tmp_path)

    main(["--data-dir", str(data_dir), "--subjects", "fsaverage"])

    assert (data_dir / "1284src_leadfield" / "fsaverage_free_leadfield.npz").is_file()
    assert "Written leadfield files" in capsys.readouterr().out


def test_extract_leadfields_without_a_forward_points_to_the_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_example_data"):
        extract_leadfields(data_dir=tmp_path)


def test_extract_leadfields_for_a_missing_forward_raises(tmp_path):
    (tmp_path / "1284src_fwd").mkdir()

    with pytest.raises(FileNotFoundError, match="fsaverage"):
        extract_leadfields("fsaverage", data_dir=tmp_path)


def test_extract_leadfields_needs_the_channel_type_of_an_unknown_subject(tmp_path):
    fwd_dir = tmp_path / "1284src_fwd"
    fwd_dir.mkdir()
    (fwd_dir / "mysubject-fwd.fif").write_bytes(b"")

    with pytest.raises(ValueError, match="coil_name"):
        extract_leadfields("mysubject", data_dir=tmp_path)
