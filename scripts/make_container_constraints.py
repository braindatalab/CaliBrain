"""Write container/constraints.txt: the package versions the container image installs.

The file pins every package that ``pip install calibrain notebook`` brings in, so the
same release always gives the same image. Run it before a release, in any environment
with the Python version of the image (``IMAGE_PYTHON``), and commit the result::

    python scripts/make_container_constraints.py --version 1.1.1
    python scripts/make_container_constraints.py --wheel dist/calibrain-1.1.1-py3-none-any.whl

``--version`` installs the release from PyPI, ``--wheel`` a local build.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

IMAGE_PYTHON = (3, 13)  # the tag of the base image in container/Dockerfile and calibrain.def
OUTPUT = Path(__file__).resolve().parents[1] / "container" / "constraints.txt"


def make_constraints(target: str) -> list[str]:
    """Install ``target`` and Jupyter in a fresh virtual environment and list the versions."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, "-m", "venv", tmp], check=True)
        python = str(Path(tmp) / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python"))
        subprocess.run([python, "-m", "pip", "install", "--quiet", target, "notebook"], check=True)
        frozen = subprocess.run(
            [python, "-m", "pip", "freeze", "--exclude-editable"],
            check=True, capture_output=True, text=True,
        ).stdout.splitlines()
    return sorted(
        (line for line in frozen if line and not line.lower().startswith("calibrain")),
        key=str.lower,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--version", help="release number on PyPI, for example 1.1.1")
    group.add_argument("--wheel", help="path of a calibrain wheel")
    parser.add_argument("--output", default=str(OUTPUT), help="file to write")
    args = parser.parse_args(argv)

    if sys.version_info[:2] != IMAGE_PYTHON:
        print(
            f"The image uses Python {IMAGE_PYTHON[0]}.{IMAGE_PYTHON[1]}, "
            f"this is {sys.version_info[0]}.{sys.version_info[1]}. The versions can differ.",
            file=sys.stderr,
        )
        return 1

    target = f"calibrain=={args.version}" if args.version else str(Path(args.wheel).resolve())
    lines = make_constraints(target)
    Path(args.output).write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} pinned packages to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
