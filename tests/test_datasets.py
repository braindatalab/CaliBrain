import hashlib
import http.server
import os
import threading
import zipfile
from pathlib import Path

import pytest

from calibrain import datasets
from calibrain.datasets import fetch_example_data

FILES = {"1284src_fwd/fsaverage-fwd.fif": b"forward solution"}
ARCHIVE = "calibrain-example-data-fsaverage.zip"
VERSION = "release-fsaverage"
CAMCAN_FILES = {f"1284src_fwd/{subject}-fwd.fif": subject.encode() for subject in ("CC1", "CC2")}
CAMCAN_ARCHIVE = "calibrain-example-data-camcan.zip"
CAMCAN_VERSION = "release-camcan"


def _entry(version, archive, sha256, files):
    return {"version": version, "archive": archive, "sha256": sha256, "files": tuple(files)}


def _make_zip(path, files):
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _Server:
    """A local web server for one directory, that counts the requests it gets."""

    def __init__(self, directory):
        self.requests = []
        outer = self

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(directory), **kwargs)

            def do_GET(self):
                outer.requests.append(self.path)
                super().do_GET()

            def log_message(self, *args):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def published(tmp_path, monkeypatch):
    """Serve fake archives of two datasets (each in its own release) and point to them."""
    served = tmp_path / "served"
    registry = {}
    for name, version, archive, files in (
        ("fsaverage", VERSION, ARCHIVE, FILES),
        ("camcan", CAMCAN_VERSION, CAMCAN_ARCHIVE, CAMCAN_FILES),
    ):
        (served / version).mkdir(parents=True)
        sha256 = _make_zip(served / version / archive, files)
        registry[name] = _entry(version, archive, sha256, files)
    server = _Server(served)
    monkeypatch.setattr(datasets, "DATA_URL", server.url + "{version}/")
    monkeypatch.setattr(datasets, "_DATASETS", registry)
    yield server
    server.close()


def test_downloads_verifies_and_unpacks_the_archive(published, tmp_path):
    data_dir = fetch_example_data(data_dir=tmp_path / "data")

    assert data_dir == tmp_path / "data"
    for name, content in FILES.items():
        assert (data_dir / name).read_bytes() == content
    # the downloaded archive is not kept, and no leadfield files are created
    assert not (data_dir / ".downloads").exists()
    assert not (data_dir / "1284src_leadfield").exists()
    assert len(published.requests) == 1


def test_does_not_download_again_when_the_files_exist(published, tmp_path):
    fetch_example_data(data_dir=tmp_path / "data")
    published.requests.clear()

    fetch_example_data(data_dir=tmp_path / "data")

    assert published.requests == []


def test_force_downloads_again_and_overwrites(published, tmp_path):
    data_dir = fetch_example_data(data_dir=tmp_path / "data")
    (data_dir / "1284src_fwd/fsaverage-fwd.fif").write_bytes(b"changed locally")

    fetch_example_data(data_dir=data_dir, force=True)

    assert (data_dir / "1284src_fwd/fsaverage-fwd.fif").read_bytes() == FILES["1284src_fwd/fsaverage-fwd.fif"]


def test_does_not_overwrite_existing_files_without_force(published, tmp_path):
    data_dir = tmp_path / "data"
    for name in FILES:
        (data_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (data_dir / name).write_bytes(b"my own file")

    fetch_example_data(data_dir=data_dir)

    assert published.requests == []
    assert (data_dir / "1284src_fwd/fsaverage-fwd.fif").read_bytes() == b"my own file"


def test_rejects_a_file_with_the_wrong_checksum(published, tmp_path, monkeypatch):
    monkeypatch.setitem(datasets._DATASETS["fsaverage"], "sha256", "0" * 64)

    with pytest.raises(ValueError, match="hash"):
        fetch_example_data(data_dir=tmp_path / "data")

    assert not (tmp_path / "data" / "1284src_fwd").exists()


def test_rejects_an_archive_with_an_unsafe_path(tmp_path, monkeypatch):
    served = tmp_path / "served" / VERSION
    served.mkdir(parents=True)
    sha256 = _make_zip(served / ARCHIVE, {**FILES, "../outside.txt": b"escaped"})
    server = _Server(tmp_path / "served")
    monkeypatch.setattr(datasets, "DATA_URL", server.url + "{version}/")
    monkeypatch.setattr(datasets, "_DATASETS", {"fsaverage": _entry(VERSION, ARCHIVE, sha256, FILES)})
    try:
        with pytest.raises(ValueError, match="Unsafe path"):
            fetch_example_data(data_dir=tmp_path / "data")
    finally:
        server.close()

    assert not (tmp_path / "outside.txt").exists()
    assert not (tmp_path / "data" / "1284src_fwd").exists()


def test_rejects_an_archive_that_misses_a_file(tmp_path, monkeypatch):
    served = tmp_path / "served" / VERSION
    served.mkdir(parents=True)
    sha256 = _make_zip(served / ARCHIVE, FILES)
    server = _Server(tmp_path / "served")
    monkeypatch.setattr(datasets, "DATA_URL", server.url + "{version}/")
    # the registry expects one more file than the archive holds
    monkeypatch.setattr(datasets, "_DATASETS", {
        "fsaverage": _entry(VERSION, ARCHIVE, sha256, (*FILES, "1284src_fwd/other-fwd.fif")),
    })
    try:
        with pytest.raises(ValueError, match="expected files"):
            fetch_example_data(data_dir=tmp_path / "data")
    finally:
        server.close()


def test_each_dataset_is_downloaded_from_its_own_release(published, tmp_path):
    fetch_example_data("camcan", data_dir=tmp_path / "data")

    assert published.requests == [f"/{CAMCAN_VERSION}/{CAMCAN_ARCHIVE}"]
    for name, content in CAMCAN_FILES.items():
        assert (tmp_path / "data" / name).read_bytes() == content
    assert not (tmp_path / "data" / "1284src_fwd" / "fsaverage-fwd.fif").exists()


def test_the_registry_is_well_formed():
    import re

    from calibrain.extract_leadfield import DEFAULT_COILS

    assert {"fsaverage", "camcan"} <= set(datasets._DATASETS)
    for name, entry in datasets._DATASETS.items():
        assert entry["version"] and entry["archive"].endswith(".zip"), name
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), name
        assert entry["files"], name
        for file in entry["files"]:
            match = re.fullmatch(r"1284src_fwd/(.+)-fwd\.fif", file)
            assert match, (name, file)
            # the extraction step must know the channel type of every downloaded subject
            assert match.group(1) in DEFAULT_COILS, (name, file)


def test_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        fetch_example_data("does-not-exist")


def test_uses_the_calibrain_data_directory_by_default(published, tmp_path, monkeypatch):
    monkeypatch.setenv("CALIBRAIN_DATA", str(tmp_path / "from_env"))

    data_dir = fetch_example_data()

    assert data_dir == tmp_path / "from_env"
    assert (data_dir / "1284src_fwd/fsaverage-fwd.fif").exists()


def test_default_data_directory_of_an_installed_package_is_a_user_directory(tmp_path, monkeypatch):
    from calibrain import utils

    # a layout like site-packages: no pyproject.toml next to the package
    fake_module = tmp_path / "site-packages" / "calibrain" / "utils.py"
    fake_module.parent.mkdir(parents=True)
    monkeypatch.setattr(utils, "__file__", str(fake_module))
    monkeypatch.delenv("CALIBRAIN_DATA", raising=False)
    monkeypatch.setattr("pooch.os_cache", lambda project: tmp_path / "user_cache" / project)

    assert utils.get_data_path() == tmp_path / "user_cache" / "calibrain"
    assert not (tmp_path / "site-packages" / "data").exists()


def test_default_data_directory_of_a_source_checkout_is_the_repository_data(tmp_path, monkeypatch):
    from calibrain import utils

    fake_module = tmp_path / "repo" / "calibrain" / "utils.py"
    fake_module.parent.mkdir(parents=True)
    (tmp_path / "repo" / "pyproject.toml").write_text("")
    monkeypatch.setattr(utils, "__file__", str(fake_module))
    monkeypatch.delenv("CALIBRAIN_DATA", raising=False)

    assert utils.get_data_path() == tmp_path / "repo" / "data"


@pytest.mark.skipif(not os.environ.get("CALIBRAIN_TEST_NETWORK"), reason="set CALIBRAIN_TEST_NETWORK=1 to test the real download")
@pytest.mark.parametrize("dataset", sorted(datasets._DATASETS))
def test_real_download_matches_the_published_checksum(tmp_path, dataset):
    data_dir = fetch_example_data(dataset, data_dir=tmp_path)

    for name in datasets._DATASETS[dataset]["files"]:
        assert (data_dir / name).is_file()
