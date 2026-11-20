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


CANONICAL_TITLE = (
    "CaliBrain: A Python toolbox for uncertainty quantification, calibration, and "
    "benchmarking in Bayesian M/EEG inverse source imaging"
)
AUTHORS = ["Mohammad Orabe", "Srikantan Nagarajan", "Stefan Haufe", "Ismail Huseynov"]


def test_title_and_description_are_the_same_everywhere():
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert f'description = "{CANONICAL_TITLE}"' in pyproject
    assert f'title: "{CANONICAL_TITLE}"' in (ROOT / "CITATION.cff").read_text()
    assert f"{CANONICAL_TITLE}.\n" in (ROOT / "README.md").read_text()
    assert f"{CANONICAL_TITLE}.\n" in (ROOT / "docs/source/index.rst").read_text()
    abstract = CANONICAL_TITLE.replace("CaliBrain: A", "CaliBrain is a") + "."
    assert f'abstract: "{abstract}"' in (ROOT / "CITATION.cff").read_text()
    assert f":description: {abstract}" in (ROOT / "docs/source/index.rst").read_text()


def test_authors_are_listed_in_the_same_order_everywhere():
    pyproject = (ROOT / "pyproject.toml").read_text()
    names = re.findall(r'\{name = "([^"]+)"', pyproject.split("authors = [", 1)[1].split("]", 1)[0])
    assert names == AUTHORS

    citation = (ROOT / "CITATION.cff").read_text()
    family = re.findall(r'family-names: "([^"]+)"', citation)
    assert family == [name.split()[-1] for name in AUTHORS]

    license_line = (ROOT / "LICENSE").read_text().splitlines()[2]
    assert license_line == "Copyright (c) 2025, " + ", ".join(AUTHORS)
    assert f"author = '{', '.join(AUTHORS)}'" in (ROOT / "docs/source/conf.py").read_text()


def test_citation_uses_the_zenodo_concept_doi_that_always_points_to_the_latest_version():
    for path in ("CITATION.cff", "README.md", "docs/source/index.rst", "docs/source/cite/index.rst"):
        text = (ROOT / path).read_text()
        assert "10.5281/zenodo.20683761" in text, path
        assert "zenodo.21261767" not in text, path  # the DOI of v1.0.3 only


def test_zenodo_metadata_agrees_with_the_citation_file():
    # Zenodo uses .zenodo.json instead of CITATION.cff when it exists, so they must agree.
    import yaml

    zenodo = json.loads((ROOT / ".zenodo.json").read_text())
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text())

    assert zenodo["title"] == citation["title"] == CANONICAL_TITLE
    assert zenodo["description"].startswith("<p>" + citation["abstract"] + "</p>")
    assert zenodo["keywords"] == citation["keywords"]
    assert zenodo["license"].lower() == citation["license"].lower()  # Zenodo spells the id in lowercase
    assert zenodo["upload_type"] == "software"
    expected = [
        {"name": f"{a['family-names']}, {a['given-names']}", "orcid": a["orcid"].rsplit("/", 1)[-1]}
        for a in citation["authors"]
    ]
    assert zenodo["creators"] == expected


def test_readme_images_use_absolute_urls_so_that_they_show_on_pypi():
    readme = (ROOT / "README.md").read_text()
    sources = re.findall(r'<img[^>]*\ssrc="([^"]+)"', readme)
    assert sources
    assert all(src.startswith("https://") for src in sources), sources
