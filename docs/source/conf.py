# filepath: /Users/orabe/0.braindata/CaliBrain/docs/source/conf.py
import os
import sys
from pathlib import Path
import calibrain

# sys.path.insert(0, os.path.abspath('../..')) # Point to project root relative to conf.py
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ... Project information ...
project = 'CaliBrain'
copyright = '2025, Mohammad Orabe, Ismail Huseynov'
author = 'Mohammad Orabe, Ismail Huseynov'

print(f"DEBUG: calibrain.__version__ = {calibrain.__version__}")

release = calibrain.__version__

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',      # Core library to generate documentation from docstrings
    'sphinx.ext.napoleon',     # Support for Google and NumPy style docstrings
    'sphinx.ext.viewcode',     # Add links to source code
    'sphinx_copybutton',       # Enable "Copy" buttons for code blocks
    'sphinx_autodoc_typehints',# Automatically document typehints
    'myst_parser',             # If you want to write docs in Markdown
    'sphinx.ext.intersphinx',  # Link to other projects' documentation
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
]
autosummary_generate = True  # Turn on sphinx.ext.autosummary

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "navigation_with_keys": True,
    "navbar_align": "content",
    "navbar_start": ["navbar-logo", "version-switcher"],
    "navbar_end": ["navbar-icon-links","theme-switcher"],

    # Custom version display
    "announcement": f"📚 You are viewing CaliBrain v{release} documentation",
    # "show_version_warning_banner": True,
    # VERSION CONFIGURATION
    "switcher": {
        # "json_url": "https://braindatalab.github.io/CaliBrain/_static/switcher.json",
        "json_url": "_static/switcher.json",  # Use local path for testing
        "version_match": f"v{release}",  # Current version (0.1.1)
    },
    
    # SIDEBAR CONFIGURATION - These are key for PyData theme
    "show_nav_level": 3,                    # Show navigation levels in sidebar
    "navigation_depth": 4,                  # Depth of navigation tree
    "show_toc_level": 3,                    # Show table of contents levels
    "collapse_navigation": False,           # Keep navigation expanded
    
    # SIDEBAR VISIBILITY - This is crucial
    "secondary_sidebar_items": ["page-toc", "sourcelink"],  # Show sidebar items
    # "primary_sidebar_end": ["sidebar-ethical-ads"],         # End of primary sidebar
    
    # LOGO AND TITLE CONFIGURATION
    "logo": {
        "image_light": "_static/caliBrain.png",
        "image_dark": "_static/caliBrain.png",
        "text": f"CaliBrain",  # Show version in logo text
    },
    
    # Show version in sidebar
    "show_version_warning_banner": True,

    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/braindatalab/CaliBrain",
            "icon": "fab fa-github",
        },
    ],
    
    # # Add external links to navigation
    # "external_links": [
    #     {
    #         "name": "Changelog",
    #         "url": "https://github.com/braindatalab/CaliBrain/blob/main/CHANGELOG.md"
    #     }
    # ],
      
    # Additional PyData theme options
    "use_edit_page_button": False,

    # IMPORTANT: Enable sidebar
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

source_suffix = {
    '.rst': None,
    '.md': 'myst_parser',
}

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

# # Optional: Reduce padding around logo if needed
# html_theme_options = {
#     # "logo_only": True,  # Only show logo, no project name
#     # "display_version": True,  
# }

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'mne': ('https://mne.tools/stable/', None),
}

html_title = "CaliBrain Documentation"
html_short_title = "CaliBrain"

# open build/html/index.html