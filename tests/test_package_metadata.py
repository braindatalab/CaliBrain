import json
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


def test_version_switcher_lists_only_read_the_docs_addresses_and_the_current_release():
    entries = json.loads((ROOT / "docs" / "source" / "_static" / "switcher.json").read_text())

    assert isinstance(entries, list)
    preferred = [entry for entry in entries if entry.get("preferred")]
    for entry in entries:
        assert {"name", "version", "url"} <= set(entry), entry
        if entry not in preferred:
            # one path segment after the host: no /en/ prefix, and no other site such as GitHub Pages
            assert entry["url"] == f"https://calibrain.readthedocs.io/{entry['version']}/", entry
    versions = [entry["version"] for entry in entries]
    assert len(versions) == len(set(versions))
    assert "dev" in versions
    # a release must be added to the switcher together with the version bump
    assert f"v{calibrain.__version__}" in versions
    # The "stable" entry is the preferred one. The theme compares its version with the version
    # of each page, so it must be the current release number: this drives the old-version banner.
    assert len(preferred) == 1
    assert preferred[0]["url"] == "https://calibrain.readthedocs.io/stable/"
    assert preferred[0]["version"] == calibrain.__version__


def test_documentation_loads_its_version_list_from_the_stable_address():
    # Released documentation keeps this address forever, so it must not change.
    conf = (ROOT / "docs" / "source" / "conf.py").read_text()

    match = re.search(r'^switcher_json_url = "([^"]+)"', conf, re.M)
    assert match and match.group(1) == "https://calibrain.readthedocs.io/stable/_static/switcher.json"
