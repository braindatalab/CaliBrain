"""Check the documentation version switcher against the live Read the Docs site.

Every released version of the documentation loads its list of versions from a
``switcher.json`` file whose address is stored in the version's own HTML. Old
releases cannot be changed, so if that file moves or disappears, their version
menu silently stops working. This script finds such problems:

1. every entry of the current ``switcher.json`` (the file in the repository and
   the one that is live) must exist on Read the Docs and must not lead to
   another site (for example the old GitHub Pages site);
2. for every listed version, the page must load, the ``switcher.json`` address
   stored in it must exist, and the list it returns must contain that version;
3. the old GitHub Pages site must be unpublished or only redirect to Read the Docs.

Usage::

    python scripts/check_version_switcher.py

It needs network access and exits with 1 if a problem is found. Run it after
publishing a release, and after changing the redirects of the Read the Docs project.
"""
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urljoin, urlparse

DOCS_HOST = "calibrain.readthedocs.io"
OLD_PAGES_URL = "https://braindatalab.github.io/CaliBrain/"
LIVE_JSON_URL = f"https://{DOCS_HOST}/stable/_static/switcher.json"
LOCAL_JSON = Path(__file__).resolve().parent.parent / "docs" / "source" / "_static" / "switcher.json"
USER_AGENT = "calibrain-docs-check"
JSON_URL_IN_PAGE = re.compile(r"json_url\s*=\s*'([^']*)'")
VERSION_MATCH_IN_PAGE = re.compile(r"theme_switcher_version_match\s*[:=]\s*['\"]([^'\"]*)['\"]")


def fetch(url, timeout=30):
    """Return ``(status, final_url, text)``; ``status`` is 0 if the host cannot be reached."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.geturl(), response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return error.code, url, ""
    except (urllib.error.URLError, TimeoutError) as error:
        return 0, url, str(error)


def entry_problems(entries):
    """Problems of the switcher entries themselves, as ``(url, problem)`` pairs."""
    problems = []
    for entry in entries:
        url = entry["url"]
        status, final_url, _ = fetch(url)
        if status != 200:
            problems.append((url, f"HTTP {status}"))
        elif urlparse(final_url).netloc != DOCS_HOST:
            problems.append((url, f"leads to {final_url}"))
    return problems


def page_problem(entry):
    """Problem with the version menu of one documentation version, or ``None``."""
    status, page_url, html = fetch(entry["url"])
    if status != 200:
        return f"page returns HTTP {status}"
    match = JSON_URL_IN_PAGE.search(html)
    if match is None:
        return None  # no pydata version menu on this page (very old builds use Read the Docs' own menu)
    json_url = urljoin(page_url, match.group(1))  # may be relative to the page
    status, _, body = fetch(json_url)
    if status != 200:
        return f"its version list ({json_url}) returns HTTP {status}"
    try:
        items = json.loads(body)
    except ValueError:
        return f"its version list ({json_url}) is not valid JSON"
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        return f"its version list ({json_url}) is not a JSON list of versions"
    other_sites = sorted({urlparse(item.get("url", "")).netloc for item in items} - {DOCS_HOST})
    if other_sites:
        return f"its version list ({json_url}) links to {', '.join(other_sites)}"
    # the version the page believes it is (it highlights the entry with this version)
    found = VERSION_MATCH_IN_PAGE.search(html)
    page_version = found.group(1) if found else entry["version"]
    if page_version not in {item.get("version") for item in items}:
        return f"its version list ({json_url}) does not contain {page_version}"
    return None


def main():
    failures = 0
    print(f"switcher.json in the repository: {LOCAL_JSON.relative_to(LOCAL_JSON.parents[3])}")
    entries = json.loads(LOCAL_JSON.read_text())
    preferred = [entry["version"] for entry in entries if entry.get("preferred")]
    if len(preferred) != 1:
        print(f"   exactly one entry must be marked preferred (the old-version banner needs it); found {preferred}")
        failures += 1

    print("\n1. entries of the switcher (must exist on Read the Docs):")
    for url, problem in entry_problems(entries) or [("all entries", "ok")]:
        print(f"   {url:<58} {problem}")
        failures += problem != "ok"

    status, _, body = fetch(LIVE_JSON_URL)
    if status != 200:
        print(f"\n   live list {LIVE_JSON_URL}: HTTP {status}")
        failures += 1
    elif json.loads(body) != entries:
        print("\n   the live list differs from the one in the repository (not published yet, or stale)")

    print("\n2. the version menu of every documented version:")
    for entry in entries:
        problem = page_problem(entry)
        print(f"   {entry['version']:<8} {'ok' if problem is None else 'PROBLEM: ' + problem}")
        failures += problem is not None

    print("\n3. the old GitHub Pages site:")
    status, _, html = fetch(OLD_PAGES_URL)
    if status == 200 and DOCS_HOST not in html:
        print(f"   {OLD_PAGES_URL} is still live and serves old documentation (unpublish it or make it redirect)")
        failures += 1
    else:
        print(f"   {OLD_PAGES_URL} " + ("is unpublished" if status != 200 else "redirects to Read the Docs"))

    print("\nresult:", "no problems found" if not failures else f"{failures} problem(s) found")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
