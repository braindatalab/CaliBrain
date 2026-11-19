"""Build the example-data archives that are uploaded to the CaliBrain-data release.

The example data are not distributed with the package. This script prepares the
files for a release of https://github.com/braindatalab/CaliBrain-data:

1. It cleans each forward solution (the local paths stored by MNE are removed, so
   that no user or machine name is published).
2. It checks that the cleaned forward solution has the same numbers as the original
   one (up to floating-point rounding), and that no personal path or user name
   remains in any file.
3. It writes one zip archive per dataset and ``manifest.json`` with the size and
   SHA-256 checksum of every archive. The checksums go into ``calibrain/datasets.py``.

Only the forward solutions are published. The leadfield files depend on the version
of CaliBrain and are created locally from them with
``python -m calibrain.extract_leadfield``.

Usage::

    python scripts/build_example_data_archives.py --output /path/to/output
    python scripts/build_example_data_archives.py --output /path/to/output --include-camcan

By default only the ``fsaverage`` template data is built. The CamCAN subjects are
derived from CamCAN participants, so only include them if the CamCAN terms allow
sharing derived data. Build in the same environment each time (the checksum depends
on the MNE and NumPy versions) and take the checksum from the file you upload.
"""
import argparse
import contextlib
import getpass
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import mne
import numpy as np

from calibrain.utils import get_data_path, strip_local_paths

# dataset name -> subjects (one forward solution each)
DATASETS = {
    "fsaverage": ["fsaverage"],
    "camcan": ["CC120166", "CC120264", "CC120309", "CC120313"],
}
FWD_DIR_NAME = "1284src_fwd"
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
    """Copy a forward solution and remove the local paths stored in its info."""
    fwd = mne.read_forward_solution(source, verbose="error")
    strip_local_paths(fwd["info"])
    mne.write_forward_solution(target, fwd, overwrite=True, verbose="error")


def build_dataset_files(name, data_dir, staging):
    """Create the cleaned files of dataset ``name`` in ``staging``; return their paths."""
    fwd_dir = staging / FWD_DIR_NAME
    fwd_dir.mkdir(parents=True)
    for subject in DATASETS[name]:
        clean_forward(Path(data_dir) / FWD_DIR_NAME / f"{subject}-fwd.fif", fwd_dir / f"{subject}-fwd.fif")
    return sorted(p for p in staging.rglob("*") if p.is_file())


def arrays_match(a, b, atol=1e-12):
    """Compare two arrays; floating-point values may differ by rounding noise.

    The last binary digit of computed values depends on the MNE and NumPy versions,
    so floats are compared with a tolerance. Everything else must be equal.
    """
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype == object or b.dtype == object:
        return a.tolist() == b.tolist()
    if np.issubdtype(a.dtype, np.floating) or np.issubdtype(b.dtype, np.floating):
        return bool(np.allclose(a, b, rtol=0, atol=atol, equal_nan=True))
    return bool(np.array_equal(a, b))


def verify_against_existing(files, staging, data_dir):
    """Check that the cleaned forwards equal the originals; return the mismatches."""
    mismatches = []
    for path in files:
        original = Path(data_dir) / path.relative_to(staging)
        new = mne.read_forward_solution(path, verbose="error")
        old = mne.read_forward_solution(original, verbose="error")
        checks = {
            "gain matrix": arrays_match(new["sol"]["data"], old["sol"]["data"]),
            "source positions": arrays_match(new["source_rr"], old["source_rr"]),
            "source normals": arrays_match(new["source_nn"], old["source_nn"]),
            "channel names": new["info"]["ch_names"] == old["info"]["ch_names"],
        }
        mismatches += [f"{path.name}: {what}" for what, same in checks.items() if not same]
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
                raise RuntimeError(f"The cleaned forward solutions differ from the originals: {mismatches}")
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
