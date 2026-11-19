"""Tests for the geometry (cortical normals, Q_basis) handled by LeadfieldBuilder.

Tests that need real forward solutions use the small (1284-source) files in
``<data>/1284src_fwd``. They are skipped when those files are not available and
write only to temporary directories.
"""
from pathlib import Path

import mne
import numpy as np
import pytest
from mne.io.constants import FIFF

from calibrain.leadfield_builder import LeadfieldBuilder

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FWD_DIR = _DATA_DIR / "1284src_fwd"

needs_eeg_fwd = pytest.mark.skipif(
    not (FWD_DIR / "fsaverage-fwd.fif").exists(), reason="1284src EEG forward not available"
)
needs_meg_fwd = pytest.mark.skipif(
    not (FWD_DIR / "CC120166-fwd.fif").exists(), reason="1284src MEG forward not available"
)


def _config(tmp_path, **leadfield_cfg):
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "subjects").mkdir(parents=True, exist_ok=True)
    return {
        "data": {
            "subject": "sub",
            "data_path": str(tmp_path / "data"),
            "subjects_dir": str(tmp_path / "subjects"),
            "save_path": str(tmp_path / "saved"),  # does not exist yet on purpose
        },
        "leadfield": leadfield_cfg,
    }


def _builder(tmp_path, **leadfield_cfg):
    return LeadfieldBuilder(config=_config(tmp_path, **leadfield_cfg), leadfield_dir=tmp_path)


def _read_fwd(subject, coil):
    fwd = mne.read_forward_solution(FWD_DIR / f"{subject}-fwd.fif", verbose="error")
    unit = {"eeg": FIFF.FIFF_UNIT_V, "mag": FIFF.FIFF_UNIT_T}[coil]
    return fwd.pick_channels([ch["ch_name"] for ch in fwd["info"]["chs"] if ch["unit"] == unit])


def test_builder_with_missing_save_path_creates_it(tmp_path):
    builder = _builder(tmp_path)

    assert (tmp_path / "saved").is_dir()
    assert builder.subject == "sub"


@needs_eeg_fwd
def test_free_eeg_leadfield_gets_unit_normals_and_identity_basis(tmp_path):
    builder = _builder(tmp_path)

    leadfield = builder.handle_leadfield(_read_fwd("fsaverage", "eeg"))

    n_sources = leadfield.shape[1]
    assert leadfield.shape[2] == 3
    assert builder.cortical_normals.shape == (n_sources, 3)
    np.testing.assert_allclose(np.linalg.norm(builder.cortical_normals, axis=1), 1.0)
    np.testing.assert_array_equal(builder.Q_basis, np.repeat(np.eye(3)[None], n_sources, axis=0))
    assert builder.src_coords.shape == (n_sources, 3)


@needs_eeg_fwd
def test_fixed_leadfield_has_no_normals_or_basis(tmp_path):
    builder = _builder(tmp_path)
    fwd = mne.convert_forward_solution(
        _read_fwd("fsaverage", "eeg"), surf_ori=True, force_fixed=True, verbose="error"
    )

    leadfield = builder.handle_leadfield(fwd)

    assert leadfield.ndim == 2
    assert builder.cortical_normals is None
    assert builder.Q_basis is None


@needs_meg_fwd
def test_free_meg_leadfield_is_reduced_to_two_orthonormal_components(tmp_path):
    builder = _builder(tmp_path)

    leadfield = builder.handle_leadfield(_read_fwd("CC120166", "mag"))

    n_sources = leadfield.shape[1]
    assert leadfield.shape[2] == 2
    assert builder.Q_basis.shape == (n_sources, 3, 2)
    gram = np.einsum("nki,nkj->nij", builder.Q_basis, builder.Q_basis)
    # The forward matrix is stored in float32, so orthonormality holds to ~1e-7.
    np.testing.assert_allclose(gram, np.broadcast_to(np.eye(2), gram.shape), atol=1e-6)


@needs_eeg_fwd
@needs_meg_fwd
@pytest.mark.parametrize("subject,coil", [("fsaverage", "eeg"), ("CC120166", "mag")])
def test_builder_geometry_matches_extract_leadfield(tmp_path, subject, coil):
    # Both ways of producing a free leadfield must agree on normals and basis.
    from calibrain.extract_leadfield import extract_leadfield

    builder = _builder(tmp_path / "builder")
    builder.handle_leadfield(_read_fwd(subject, coil))

    extract_leadfield(FWD_DIR, subject, coil, "free", tmp_path)
    with np.load(tmp_path / f"{subject}_free_leadfield.npz", allow_pickle=True) as ref:
        np.testing.assert_allclose(builder.cortical_normals, ref["cortical_normals"], atol=1e-9)
        np.testing.assert_allclose(builder.Q_basis, ref["Q_basis"], atol=1e-6)


@needs_meg_fwd
def test_saved_leadfield_round_trips_geometry_and_units(tmp_path):
    first = _builder(tmp_path, save=True)
    leadfield = first.handle_leadfield(_read_fwd("CC120166", "mag"))
    saved = tmp_path / "saved" / "sub-leadfield-free.npz"
    assert saved.exists()

    second = _builder(tmp_path, fname=str(saved))
    loaded = second.handle_leadfield()

    # Units are restored before scaling, so the cached leadfield is not rescaled.
    np.testing.assert_array_equal(loaded, leadfield)
    np.testing.assert_array_equal(second.cortical_normals, first.cortical_normals)
    np.testing.assert_array_equal(second.Q_basis, first.Q_basis)
    np.testing.assert_array_equal(second.src_coords, first.src_coords)
    assert second.sensor_units == first.sensor_units == FIFF.FIFF_UNIT_T
    assert second.sensor_unitmult == first.sensor_unitmult


@needs_eeg_fwd
def test_free_leadfield_requires_surface_source_spaces(tmp_path):
    fwd = _read_fwd("fsaverage", "eeg")
    fwd["src"][0]["type"] = "vol"

    with pytest.raises(ValueError, match="surface source spaces"):
        _builder(tmp_path).handle_leadfield(fwd)


@needs_eeg_fwd
def test_loaded_forward_is_converted_to_the_requested_orientation(tmp_path):
    config = _config(tmp_path)
    config["forward_solution"] = {
        "fname": str(FWD_DIR / "fsaverage-fwd.fif"),
        "orientation_type": "fixed",
    }
    builder = LeadfieldBuilder(config=config, leadfield_dir=tmp_path)

    fwd = builder.handle_forward_solution(info=None, src=None, bem=None)

    assert fwd["source_ori"] == FIFF.FIFFV_MNE_FIXED_ORI
