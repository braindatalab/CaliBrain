"""Build the example-data archives that are uploaded to Zenodo.

The example data are not distributed with the package. This script prepares the
files for a Zenodo dataset record:

1. It cleans each forward solution (the local paths stored by MNE are removed,
   because a published file cannot be edited afterwards).
2. It regenerates the leadfield files from the cleaned forward solutions with
   ``calibrain.extract_leadfield``.
3. It checks that the regenerated leadfields equal the existing ones, and that no
   personal path or user name remains in any file.
4. It writes one zip archive per dataset and ``manifest.json`` with the size and
   SHA-256 checksum of every archive. The checksums go into the download registry.

Usage::

    python scripts/build_example_data_archives.py --output /path/to/output
    python scripts/build_example_data_archives.py --output /path/to/output --include-camcan

By default only the ``fsaverage`` template data is built. The CamCAN subjects are
derived from CamCAN participants, so only include them if the CamCAN terms allow
sharing derived data. Upload the archives of ONE run: the forward files contain a
creation time, so the checksums differ between runs.
"""
import argparse
import contextlib
import getpass
import hashlib
import io
import json
import tempfile
import zipfile
from pathlib import Path

import mne
import numpy as np

from calibrain.extract_leadfield import extract_leadfield
from calibrain.utils import get_data_path, strip_local_paths

# dataset name -> {channel type: subjects}
DATASETS = {
    "fsaverage": {"eeg": ["fsaverage"]},
    "camcan": {"mag": ["CC120166", "CC120264", "CC120309", "CC120313"]},
}
ORIENTATIONS = ("fixed", "free")
FWD_DIR_NAME = "1284src_fwd"
LEADFIELD_DIR_NAME = "1284src_leadfield"
ARCHIVE_NAME = "calibrain-example-data-{name}.zip"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)  # fixed, so that the zip itself is reproducible


def forbidden_patterns():
    """Byte strings that must not appear in a file that is published."""
    patterns = [b"/home/", b"/Users/", b"/datasabzi", b"C:\\Users"]
    with contextlib.suppress(Exception):
        patterns.append(getpass.getuser().encode())
    return patterns


def find_leaks(path):
    """Return the forbidden patterns that occur in the file ``path``."""
    content = Path(path).read_bytes()
    return [p.decode(errors="replace") for p in forbidden_patterns() if p in content]


def clean_forward(source, target):
    """Copy a forward solution and remove the local directory stored in its info."""
    fwd = mne.read_forward_solution(source, verbose="error")
    strip_local_paths(fwd["info"])
    mne.write_forward_solution(target, fwd, overwrite=True, verbose="error")


def build_dataset_files(name, data_dir, staging):
    """Create the cleaned files of dataset ``name`` in ``staging``; return their paths."""
    fwd_dir = staging / FWD_DIR_NAME
    leadfield_dir = staging / LEADFIELD_DIR_NAME
    fwd_dir.mkdir(parents=True)
    leadfield_dir.mkdir(parents=True)
    for coil, subjects in DATASETS[name].items():
        for subject in subjects:
            clean_forward(Path(data_dir) / FWD_DIR_NAME / f"{subject}-fwd.fif", fwd_dir / f"{subject}-fwd.fif")
            for orientation in ORIENTATIONS:
                with contextlib.redirect_stdout(io.StringIO()):
                    extract_leadfield(fwd_dir, subject, coil, orientation, leadfield_dir)
    return sorted(p for p in staging.rglob("*") if p.is_file())


def verify_against_existing(files, staging, data_dir):
    """Check that regenerated leadfields equal the existing ones; return the mismatches."""
    mismatches = []
    for path in files:
        if path.suffix != ".npz":
            continue
        existing = Path(data_dir) / path.relative_to(staging)
        if not existing.exists():
            continue
        with np.load(path, allow_pickle=True) as new, np.load(existing, allow_pickle=True) as old:
            for key in new.files:
                if key == "info":
                    continue
                a, b = new[key], old[key]
                same = a.shape == b.shape and (
                    a.tolist() == b.tolist() if a.dtype == object else np.array_equal(a, b)
                )
                if not same:
                    mismatches.append(f"{path.name}:{key}")
    return mismatches


def write_zip(archive, files, staging):
    """Write ``files`` to a zip with fixed timestamps and a stable member order."""
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(files):
            info = zipfile.ZipInfo(path.relative_to(staging).as_posix(), date_time=ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, path.read_bytes())


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_archives(data_dir, output, names):
    """Build the archives ``names`` into ``output``; return the manifest."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in names:
        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp)
            files = build_dataset_files(name, data_dir, staging)
            leaks = {p.name: find_leaks(p) for p in files if find_leaks(p)}
            if leaks:
                raise RuntimeError(f"Personal information found in {name}: {leaks}")
            mismatches = verify_against_existing(files, staging, Path(data_dir))
            if mismatches:
                raise RuntimeError(f"Regenerated leadfields differ from the existing ones: {mismatches}")
            archive = output / ARCHIVE_NAME.format(name=name)
            write_zip(archive, files, staging)
            manifest[archive.name] = {
                "dataset": name,
                "size_bytes": archive.stat().st_size,
                "sha256": sha256(archive),
                "files": [p.relative_to(staging).as_posix() for p in files],
            }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", required=True, help="Directory for the archives and manifest.json.")
    parser.add_argument("--data-dir", help="Data directory (default: CALIBRAIN_DATA or <repository>/data).")
    parser.add_argument("--include-camcan", action="store_true",
                        help="Also build the CamCAN subjects (only if their terms allow sharing).")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir) if args.data_dir else get_data_path()
    names = ["fsaverage"] + (["camcan"] if args.include_camcan else [])
    manifest = build_archives(data_dir, args.output, names)
    for archive, entry in manifest.items():
        print(f"{archive}  {entry['size_bytes'] / 1e6:.1f} MB  sha256:{entry['sha256']}")
        for file in entry["files"]:
            print(f"    {file}")
    print(f"\nWrote {Path(args.output) / 'manifest.json'}")


if __name__ == "__main__":
    main()
