from pathlib import Path

import numpy as np
import pytest

from calibrain import reduce_fwd
from calibrain.reduce_fwd import reduce_forward_solutions

ROOT = Path(__file__).resolve().parent.parent
FULL_FWD = ROOT / "data" / "fwd" / "fsaverage-fwd.fif"
REDUCED_FWD = ROOT / "data" / "1284src_fwd" / "fsaverage-fwd.fif"


@pytest.fixture
def stubbed_mne(monkeypatch):
    """Replace reading, reducing and writing, and record what was asked."""
    calls = {"restrict": [], "read": []}

    class FakeForward(dict):
        pass

    def read(path, verbose=None):
        calls["read"].append(Path(path).name)
        return FakeForward(nsource=5124)

    def restrict(fwd, n_keep, seed):
        calls["restrict"].append((n_keep, seed))
        return FakeForward(nsource=2 * n_keep)

    def write(path, fwd, overwrite=False, verbose=None):
        Path(path).write_text("reduced")

    monkeypatch.setattr(reduce_fwd.mne, "read_forward_solution", read)
    monkeypatch.setattr(reduce_fwd.mne, "write_forward_solution", write)
    monkeypatch.setattr(reduce_fwd, "restrict_fwd_to_sources", restrict)
    return calls


def _full_forwards(data_dir, *subjects):
    (data_dir / "fwd").mkdir(parents=True)
    for subject in subjects:
        (data_dir / "fwd" / f"{subject}-fwd.fif").write_text("full")


def test_reduces_all_subjects_of_the_full_folder_by_default(tmp_path, stubbed_mne):
    _full_forwards(tmp_path, "b", "a")

    written = reduce_forward_solutions(data_dir=tmp_path)

    assert written == [tmp_path / "1284src_fwd" / "a-fwd.fif", tmp_path / "1284src_fwd" / "b-fwd.fif"]
    assert all(path.is_file() for path in written)
    assert stubbed_mne["restrict"] == [(642, 42), (642, 42)]  # 642 per hemisphere, seed 42


def test_the_seed_and_the_subjects_can_be_chosen(tmp_path, stubbed_mne):
    _full_forwards(tmp_path, "a", "b")

    written = reduce_forward_solutions("b", data_dir=tmp_path, seed=7)

    assert written == [tmp_path / "1284src_fwd" / "b-fwd.fif"]
    assert stubbed_mne["read"] == ["b-fwd.fif"]
    assert stubbed_mne["restrict"] == [(642, 7)]


def test_existing_reduced_forwards_are_skipped_unless_forced(tmp_path, stubbed_mne):
    _full_forwards(tmp_path, "a")
    reduced = tmp_path / "1284src_fwd" / "a-fwd.fif"
    reduced.parent.mkdir()
    reduced.write_text("keep me")

    assert reduce_forward_solutions(data_dir=tmp_path) == []
    assert reduced.read_text() == "keep me"

    assert reduce_forward_solutions(data_dir=tmp_path, force=True) == [reduced]
    assert reduced.read_text() == "reduced"


def test_missing_forward_solutions_raise(tmp_path, stubbed_mne):
    (tmp_path / "fwd").mkdir()
    with pytest.raises(FileNotFoundError, match=r"No forward solutions \(\*-fwd.fif\)"):
        reduce_forward_solutions(data_dir=tmp_path)

    _full_forwards(tmp_path / "other", "a")
    with pytest.raises(FileNotFoundError, match="subject 'missing'"):
        reduce_forward_solutions("missing", data_dir=tmp_path / "other")


def test_command_line_passes_its_options(tmp_path, stubbed_mne):
    _full_forwards(tmp_path, "a")

    reduce_fwd.main(["--data-dir", str(tmp_path), "--subjects", "a", "--seed", "3"])

    assert stubbed_mne["restrict"] == [(642, 3)]
    assert (tmp_path / "1284src_fwd" / "a-fwd.fif").is_file()


@pytest.mark.skipif(
    not (FULL_FWD.exists() and REDUCED_FWD.exists()),
    reason="needs the full and the reduced fsaverage forward solutions in data/",
)
def test_reduction_reproduces_the_example_forward_solution(tmp_path):
    import mne

    (tmp_path / "fwd").mkdir()
    (tmp_path / "fwd" / "fsaverage-fwd.fif").symlink_to(FULL_FWD)

    (written,) = reduce_forward_solutions(data_dir=tmp_path)

    new = mne.read_forward_solution(written, verbose="error")
    reference = mne.read_forward_solution(REDUCED_FWD, verbose="error")
    assert new["nsource"] == 1284
    assert all(np.array_equal(a["vertno"], b["vertno"]) for a, b in zip(new["src"], reference["src"]))
    assert np.allclose(new["sol"]["data"], reference["sol"]["data"], rtol=0, atol=1e-12)
