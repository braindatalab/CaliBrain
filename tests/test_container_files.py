"""The files that build the container image agree with each other."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTAINER = ROOT / "container"


def _base_image(path):
    match = re.search(r"^(?:FROM|From:)\s+(python:\S+)", path.read_text(), re.MULTILINE)
    assert match, path
    return match.group(1)


def test_dockerfile_and_definition_file_use_the_same_base_image():
    assert _base_image(CONTAINER / "Dockerfile") == _base_image(CONTAINER / "calibrain.def")


def test_constraints_are_made_for_the_python_of_the_base_image():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import make_container_constraints
    finally:
        sys.path.pop(0)
    major, minor = make_container_constraints.IMAGE_PYTHON
    assert _base_image(CONTAINER / "Dockerfile").startswith(f"python:{major}.{minor}")


def test_both_build_files_install_through_the_same_script():
    for name in ("Dockerfile", "calibrain.def"):
        assert "install_calibrain.sh" in (CONTAINER / name).read_text(), name


def test_constraints_pin_everything_and_do_not_pin_calibrain_itself():
    lines = [x for x in (CONTAINER / "constraints.txt").read_text().splitlines() if x]
    assert lines
    assert all("==" in line for line in lines)
    names = {line.split("==")[0].lower() for line in lines}
    assert "calibrain" not in names
    for required in ("numpy", "scipy", "mne", "notebook"):
        assert required in names, required


def test_wheel_folder_is_tracked():
    assert (CONTAINER / "wheels" / ".gitkeep").exists()
