import re
from pathlib import Path

import calibrain

ROOT = Path(__file__).resolve().parent.parent


def test_version_is_defined_in_one_place_and_matches_citation():
    assert re.fullmatch(r"\d+\.\d+\.\d+", calibrain.__version__)

    # pyproject.toml reads the version from calibrain.__version__.
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert 'dynamic = ["version"]' in pyproject
    assert 'version = {attr = "calibrain.__version__"}' in pyproject

    # CITATION.cff must be bumped together with the package version.
    citation = re.search(r'^version:\s*"?([^"\n]+)"?', (ROOT / "CITATION.cff").read_text(), re.M)
    assert citation and citation.group(1) == calibrain.__version__


def test_repository_root_is_not_a_package():
    assert not (ROOT / "__init__.py").exists()


def test_3d_plot_without_the_viz_extra_logs_an_install_hint(monkeypatch, caplog):
    import importlib.util

    from calibrain.visualization import Visualizer

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a, **k: None if name == "pyvistaqt" else real_find_spec(name, *a, **k)
    )

    Visualizer().plot_stc_3d_brain(
        ERP_config={}, x_one_trial=None, x_hat_one_trial=None, orientations=None,
        source_units=None, source_unitmult=None, sample_idx=0, subject="s", fwd_path="p", show=True,
    )

    assert 'calibrain[viz]' in caplog.text
