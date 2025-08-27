# filepath: /Users/orabe/0.braindata/CaliBrain/docs/source/conf.py
import os
import sys
from pathlib import Path
import calibrain
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# from sphinx_gallery.sorting import ExplicitOrder

# sys.path.insert(0, os.path.abspath('../..')) # Point to project root relative to conf.py
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ADD tutorial path for Sphinx Gallery
# sys.path.insert(0, os.path.abspath('../../tutorials'))

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
    'sphinx.ext.autosummary',  # Generate summary tables for modules
    'sphinx.ext.mathjax',      # For rendering math equations
    'sphinx_gallery.gen_gallery',          # For auto-generating tutorial gallery
    'matplotlib.sphinxext.plot_directive', # For matplotlib plots
]

autosummary_generate = True  # Turn on sphinx.ext.autosummary

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "navigation_with_keys": True,
    "navbar_align": "content",
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["version-switcher", "navbar-icon-links","theme-switcher"],

    # Custom version display
    "announcement": f"📚 You are viewing CaliBrain v{release} documentation",
    
    # VERSION CONFIGURATION
    "switcher": {
        # "json_url": "https://braindatalab.github.io/CaliBrain/_static/switcher.json",
        "json_url": "_static/switcher.json",  # Use local path for testing
        "version_match": f"v{release}",  # Current version
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

# ADD Sphinx Gallery configuration
sphinx_gallery_conf = {
    # Handle multiple source directories
    'examples_dirs': ['../../tutorials', '../../examples'],   # Multiple source dirs
    'gallery_dirs': ['auto_tutorials', 'auto_examples'],      # Corresponding output dirs

    # Process all Python files (not just overview.py)
    'filename_pattern': '/.*.py$',                            # All .py files
    'ignore_pattern': '__init__.py',       # Ignore these patterns
    
    # Gallery settings
    'plot_gallery': True,  # Generate plots
    'download_all_examples': True,       # Don't create download zip
    'within_subsection_order': 'FileNameSortKey',             # Sort by filename
    'first_notebook_cell': '# This tutorial is available for download as a Jupyter notebook',
    'remove_config_comments': True,
    'show_memory': True,             
    'matplotlib_animations': True,        # Support animated plots
    
    # Subsection configuration
    # 'subsection_order': ExplicitOrder(['../../tutorials', '../../examples']),
    'expected_failing_examples': [],      # List any expected failures

    # Add these to handle thumbnail issues
    'image_scrapers': ('matplotlib',),
    'reset_modules': ('matplotlib', 'seaborn'),
    'thumbnail_size': (200, 200),
    'abort_on_example_error': False,  # Don't stop on errors
    'default_thumb_file': None,  # Use default thumbnail
    'promote_jupyter_magic': True,  # Handle notebook compatibility

    # Add this to create dummy thumbnails if plots fail
    'min_reported_time': 0,                                 # Report all execution times
    'show_signature': False,        
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