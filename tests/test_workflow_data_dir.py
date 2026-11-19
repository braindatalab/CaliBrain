"""The workflows read the forward solutions and leadfields from a configurable data folder."""
import numpy as np
import pytest

from calibrain.workflows import calibration, data_generation


@pytest.fixture(autouse=True)
def empty_coordinate_cache():
    calibration._COORD_CACHE.clear()
    yield
    calibration._COORD_CACHE.clear()


def test_calibration_reads_the_forward_solution_from_the_given_data_dir(tmp_path, monkeypatch):
    fwd_file = tmp_path / "1284src_fwd" / "S1-fwd.fif"
    fwd_file.parent.mkdir()
    fwd_file.touch()
    read = []

    def fake_read_forward_solution(path, verbose=None):
        read.append(path)
        rr = np.arange(12, dtype=float).reshape(4, 3)
        return {"src": [{"rr": rr, "vertno": np.array([0, 1])}, {"rr": rr, "vertno": np.array([2])}]}

    monkeypatch.setattr(calibration.mne, "read_forward_solution", fake_read_forward_solution)

    coords = calibration._load_subject_coords("S1", tmp_path)

    assert read == [str(fwd_file)]
    assert coords.shape == (3, 3)


def test_calibration_reports_the_missing_forward_solution_with_its_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="1284src_fwd"):
        calibration._load_subject_coords("S1", tmp_path)


def test_coordinates_of_different_data_folders_are_not_mixed_up(tmp_path, monkeypatch):
    values = {}
    for name, offset in (("a", 0.0), ("b", 100.0)):
        fwd_file = tmp_path / name / "1284src_fwd" / "S1-fwd.fif"
        fwd_file.parent.mkdir(parents=True)
        fwd_file.touch()
        values[str(fwd_file)] = offset

    def fake_read_forward_solution(path, verbose=None):
        rr = np.full((1, 3), values[path])
        return {"src": [{"rr": rr, "vertno": np.array([0])}]}

    monkeypatch.setattr(calibration.mne, "read_forward_solution", fake_read_forward_solution)

    assert calibration._load_subject_coords("S1", tmp_path / "a")[0, 0] == 0.0
    assert calibration._load_subject_coords("S1", tmp_path / "b")[0, 0] == 100.0


def test_data_generation_looks_for_leadfields_in_the_given_data_dir(tmp_path, monkeypatch):
    seen = {}

    class RecordingLeadfieldBuilder:
        def __init__(self, leadfield_dir, logger=None):
            seen["leadfield_dir"] = leadfield_dir

    monkeypatch.setattr(data_generation, "LeadfieldBuilder", RecordingLeadfieldBuilder)

    # no estimators: the run stops right after the leadfield builder was created
    with pytest.raises(ValueError, match="at least one estimator"):
        data_generation.run_data_generation({"data_dir": tmp_path, "estimators": []})

    assert seen["leadfield_dir"] == tmp_path / "1284src_leadfield"
