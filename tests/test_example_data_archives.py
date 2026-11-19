import importlib.util
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

from calibrain.utils import strip_local_paths

ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = ROOT / "data"

needs_fwd = pytest.mark.skipif(
    not (_DATA_DIR / "1284src_fwd" / "fsaverage-fwd.fif").exists(),
    reason="1284src forward solutions not available",
)


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "build_example_data_archives", ROOT / "scripts" / "build_example_data_archives.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_strip_local_paths_keeps_only_file_names():
    info = {
        "working_dir": "/home/someone/project",
        "mri_file": "/Users/someone/env/fsaverage-trans.fif",
        "meas_file": "C:\\Users\\someone\\data\\raw.fif",
        "command_line": "make_forward_solution(instance of Info, /Users/someone/env/fsaverage-trans.fif, None)",
    }

    strip_local_paths(info)

    assert info["working_dir"] == ""
    assert info["mri_file"] == "fsaverage-trans.fif"
    assert info["meas_file"] == "raw.fif"
    assert info["command_line"] == "make_forward_solution(instance of Info, fsaverage-trans.fif, None)"


def test_strip_local_paths_leaves_other_fields_alone():
    info = {"working_dir": "/home/someone", "description": "/keep/this/text", "nchan": 64}

    strip_local_paths(info)

    assert info["description"] == "/keep/this/text"
    assert info["nchan"] == 64


def test_arrays_match_allows_rounding_noise_but_not_real_differences():
    script = _load_script()
    values = np.array([0.1, 0.2, 0.3])

    assert script.arrays_match(values, values + 2.2e-16)      # rounding noise of another library version
    assert not script.arrays_match(values, values + 1e-6)     # a real difference
    assert not script.arrays_match(values, values[:2])        # different shape
    assert script.arrays_match(np.array([1, 2]), np.array([1, 2]))
    assert not script.arrays_match(np.array([1, 2]), np.array([1, 3]))


def test_find_leaks_reports_personal_paths(tmp_path):
    script = _load_script()
    leaky = tmp_path / "leaky.bin"
    leaky.write_bytes(b"data /home/someone/project more data")
    clean = tmp_path / "clean.bin"
    clean.write_bytes(b"data fsaverage-trans.fif more data")

    assert "/home/" in script.find_leaks(leaky)
    assert script.find_leaks(clean) == []


def test_zip_is_reproducible_for_identical_files(tmp_path):
    script = _load_script()
    (tmp_path / "a").mkdir()
    first = tmp_path / "a" / "data.txt"
    first.write_text("same content")
    os.utime(first, (1_000_000_000, 1_000_000_000))

    script.write_zip(tmp_path / "one.zip", [first], tmp_path / "a")
    os.utime(first, (1_700_000_000, 1_700_000_000))  # different modification time
    script.write_zip(tmp_path / "two.zip", [first], tmp_path / "a")

    assert (tmp_path / "one.zip").read_bytes() == (tmp_path / "two.zip").read_bytes()


@needs_fwd
def test_fsaverage_archive_is_complete_and_free_of_personal_information(tmp_path):
    script = _load_script()

    manifest = script.build_archives(_DATA_DIR, tmp_path, ["fsaverage"])

    archive = tmp_path / "calibrain-example-data-fsaverage.zip"
    entry = manifest[archive.name]
    assert entry["sha256"] == script.sha256(archive)
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest
    with zipfile.ZipFile(archive) as zf:
        assert sorted(zf.namelist()) == ["1284src_fwd/fsaverage-fwd.fif"]
        for name in zf.namelist():
            content = zf.read(name)
            assert not any(pattern in content for pattern in script.forbidden_patterns()), name



@pytest.mark.parametrize("dataset,subject,coil", [("fsaverage", "fsaverage", "eeg"), ("camcan", "CC120166", "mag")])
def test_leadfields_extracted_from_the_archive_equal_those_from_the_original(tmp_path, dataset, subject, coil):
    # the two steps of a user: unpack the downloaded forward, then extract the leadfields
    from calibrain.extract_leadfield import extract_leadfield, extract_leadfields

    script = _load_script()
    if not all((_DATA_DIR / "1284src_fwd" / f"{s}-fwd.fif").exists() for s in script.DATASETS[dataset]):
        pytest.skip(f"forward solutions of {dataset} not available")
    script.build_archives(_DATA_DIR, tmp_path / "out", [dataset])
    user_dir = tmp_path / "user"
    with zipfile.ZipFile(tmp_path / "out" / f"calibrain-example-data-{dataset}.zip") as zf:
        zf.extractall(user_dir)
    extract_leadfields(data_dir=user_dir)

    for orientation in ("fixed", "free"):
        extract_leadfield(_DATA_DIR / "1284src_fwd", subject, coil, orientation, tmp_path / "original")
        name = f"{subject}_{orientation}_leadfield.npz"
        with np.load(user_dir / "1284src_leadfield" / name, allow_pickle=True) as new, \
                np.load(tmp_path / "original" / name, allow_pickle=True) as old:
            for key in new.files:
                if key != "info":
                    assert script.arrays_match(new[key], old[key]), (name, key)


def test_the_builder_rejects_an_unknown_dataset(tmp_path):
    script = _load_script()

    with pytest.raises(SystemExit):
        script.main(["--output", str(tmp_path), "--datasets", "does-not-exist"])
