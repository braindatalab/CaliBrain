import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from sphinx_gallery.sorting import FileNameSortKey

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import calibrain

project = 'CaliBrain'
copyright = '2026, Mohammad Orabe'
author = 'Mohammad Orabe, Ismail Huseynov, Srikantan Nagarajan, Stefan Haufe'
release = calibrain.__version__
version = release
documentation_version = os.environ.get(
    "READTHEDOCS_VERSION",
    os.environ.get("DOCS_VERSION", "latest"),
)
switcher_json_url = "https://calibrain.readthedocs.io/en/latest/_static/switcher.json"

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
    'sphinx_autodoc_typehints',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx_gallery.gen_gallery',
    'matplotlib.sphinxext.plot_directive',
]

autosummary_generate = True
exclude_patterns = []
suppress_warnings = ['config.cache']

sphinx_gallery_conf = {
    'examples_dirs': ['../../tutorials'],
    'gallery_dirs': ['auto_tutorials'],
    'filename_pattern': r'.*\.py$',
    'ignore_pattern': r'__init__\.py',
    'within_subsection_order': FileNameSortKey,
    'plot_gallery': True,
    'download_all_examples': True,
    'image_scrapers': ('matplotlib',),
    'thumbnail_size': (220, 160),
    'abort_on_example_error': True,
    'show_memory': False,
    'show_signature': False,
}

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "navigation_with_keys": True,
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["version-switcher", "navbar-icon-links", "theme-switcher"],

    "announcement": f"You are viewing CaliBrain v{release} documentation",
    "switcher": {
        "json_url": switcher_json_url,
        "version_match": documentation_version,
    },
    
    "show_nav_level": 1,
    "navigation_depth": 3,
    "show_toc_level": 3,
    "collapse_navigation": False,
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
    "logo": {
        "image_light": "_static/caliBrain.png",
        "image_dark": "_static/caliBrain.png",
        "text": "CaliBrain",
    },
    "show_version_warning_banner": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/braindatalab/CaliBrain",
            "icon": "fab fa-github",
        },
    ],
    "use_edit_page_button": False,
    "navbar_persistent": ["search-button"],
    "header_links_before_dropdown": 5,
}

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

source_suffix = {'.rst': 'restructuredtext'}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

master_doc = 'index'

# Tell Sphinx where static files are located
html_static_path = ['_static']
html_css_files = ['custom.css']
templates_path = ['_templates']
html_sidebars = {
    "**": ["sidebar-nav-bs.html"],
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'mne': ('https://mne.tools/stable/', None),
}

html_meta = {
    "google-site-verification": "U4j0wnWaoGzIzJ5_ThCD3I8YLlSSrcOoIeECjL2nPuk",
}

html_title = "CaliBrain Documentation"
html_short_title = "CaliBrain"
