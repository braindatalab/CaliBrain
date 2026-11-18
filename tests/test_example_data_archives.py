import importlib.util
import json
import os
import zipfile
from pathlib import Path

import pytest

from calibrain.utils import strip_local_paths

ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = Path(os.environ.get("CALIBRAIN_DATA", ROOT / "data"))

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
        assert sorted(zf.namelist()) == [
            "1284src_fwd/fsaverage-fwd.fif",
            "1284src_leadfield/fsaverage_fixed_leadfield.npz",
            "1284src_leadfield/fsaverage_free_leadfield.npz",
        ]
        for name in zf.namelist():
            content = zf.read(name)
            assert not any(pattern in content for pattern in script.forbidden_patterns()), name
