"""Download the CaliBrain example data.

The example forward solutions are not part of the package. They are published as
release assets of https://github.com/braindatalab/CaliBrain-data and downloaded on
demand with :func:`fetch_example_data`. Every archive is verified against a SHA-256
checksum that is stored in this module, so a given version of CaliBrain always gets
exactly the data it was tested with.

Only the forward solutions are downloaded. The leadfield files depend on the version
of CaliBrain, so they are created from the forward solutions in a separate step, with
:func:`calibrain.extract_leadfield.extract_leadfields`.
"""
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Optional, Union

import pooch

from calibrain.utils import get_data_path

__all__ = ["fetch_example_data"]

#: Download location of the archives (``{version}`` is the release of
#: https://github.com/braindatalab/CaliBrain-data that holds the dataset).
DATA_URL = "https://github.com/braindatalab/CaliBrain-data/releases/download/{version}/"

# dataset name -> release of CaliBrain-data, archive file name, SHA-256 checksum, and
# the files it provides. Every dataset has its own release, and a release is never changed.
_DATASETS = {
    "fsaverage": {
        "version": "example-data-fsaverage-1",
        "archive": "calibrain-example-data-fsaverage.zip",
        "sha256": "0b2d57df59b22aff4541ea33ed75faa36290709034823275823cc892580efcce",
        "files": ("1284src_fwd/fsaverage-fwd.fif",),
    },
    # Derived from four CamCAN participants; see the Dataset Notice in the documentation.
    "camcan": {
        "version": "example-data-camcan-1",
        "archive": "calibrain-example-data-camcan.zip",
        "sha256": "6e19da512390614f865d246204db585447a9e8447dd0b1c3a8e7172e6d9cd124",
        "files": (
            "1284src_fwd/CC120166-fwd.fif",
            "1284src_fwd/CC120264-fwd.fif",
            "1284src_fwd/CC120309-fwd.fif",
            "1284src_fwd/CC120313-fwd.fif",
        ),
    },
}


def fetch_example_data(
    dataset: str = "fsaverage",
    data_dir: Optional[Union[str, Path]] = None,
    force: bool = False,
) -> Path:
    """Download the example forward solution into the data directory, once.

    The archive is downloaded from the CaliBrain-data repository, checked against its
    SHA-256 checksum, and unpacked into ``<data_dir>/1284src_fwd``. Nothing is
    downloaded if the files already exist. This function only downloads the forward
    solution; create the leadfield files from it in a separate step with
    :func:`calibrain.extract_leadfield.extract_leadfields`.

    Parameters
    ----------
    dataset : {"fsaverage", "camcan"}, default "fsaverage"
        The dataset to fetch: ``"fsaverage"`` (the template forward solution) or
        ``"camcan"`` (the forward solutions of four CamCAN participants, which are
        subject to the CamCAN terms; see the Dataset Notice in the documentation).
    data_dir : str | Path | None
        Directory to unpack the data into. By default the CaliBrain data directory
        is used: ``<repository>/data`` in a source checkout, or a per-user directory
        for an installed package (see :func:`calibrain.utils.get_data_path`).
    force : bool, default False
        Download again and overwrite existing files.

    Returns
    -------
    data_dir : pathlib.Path
        The data directory that now contains the files.

    Raises
    ------
    ValueError
        If ``dataset`` is unknown, if the downloaded file does not match its
        checksum, or if the archive is not as expected.
    """
    if dataset not in _DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}. Available datasets: {sorted(_DATASETS)}.")
    entry = _DATASETS[dataset]
    data_dir = get_data_path(data_dir)

    if not force and all((data_dir / name).is_file() for name in entry["files"]):
        return data_dir

    download_dir = data_dir / ".downloads"
    archive = Path(
        pooch.retrieve(
            url=DATA_URL.format(version=entry["version"]) + entry["archive"],
            known_hash=f"sha256:{entry['sha256']}",
            fname=entry["archive"],
            path=download_dir,
            progressbar=False,
        )
    )
    try:
        _extract(archive, data_dir, entry["files"])
    finally:
        archive.unlink(missing_ok=True)
        try:
            download_dir.rmdir()
        except OSError:  # not empty, or already removed
            pass
    return data_dir


def _extract(archive, data_dir, expected_files):
    """Unpack ``archive`` into ``data_dir`` after checking that it is safe and complete."""
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        for name in names:
            posix, windows = PurePosixPath(name), PureWindowsPath(name)
            if posix.is_absolute() or windows.is_absolute() or ".." in posix.parts or ".." in windows.parts:
                raise ValueError(f"Unsafe path in the archive: {name!r}")
        missing = [name for name in expected_files if name not in names]
        if missing:
            raise ValueError(f"The archive does not contain the expected files: {missing}")
        zf.extractall(data_dir)
