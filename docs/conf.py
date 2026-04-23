# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from setuptools_scm import get_version

# -- Project information -----------------------------------------------------

project = "2p-preprocess"
copyright = "2024, Znamenskiy lab"
author = "Znamenskiy lab"

# Version is derived from git tags via setuptools_scm.
# `release` is the full version string (e.g. "1.2.3.dev4+gabcdef");
# `version` is the short X.Y form shown in the sidebar.
try:
    release = get_version(root="..", relative_to=__file__)
except Exception:
    release = "unknown"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",           # Markdown support
    "autoapi.extension",     # Auto-generate API docs without importing the package
    "sphinx.ext.napoleon",   # Google / NumPy docstring styles
    "sphinx.ext.viewcode",   # Add links to source code
    "sphinx.ext.intersphinx",  # Cross-link to numpy, python docs
    "sphinx_copybutton",     # Copy button on code blocks
]

# -- MyST configuration ------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",     # ::: fences as an alternative to ```
    "deflist",         # Definition lists
]
myst_heading_anchors = 3  # Auto-generate anchors for headings up to h3

# -- AutoAPI configuration ---------------------------------------------------

autoapi_dirs = ["../twop_preprocess"]
autoapi_type = "python"
autoapi_add_toctree_entry = True
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_python_class_content = "both"
# Suppress "nothing to document" warnings for modules with no docstrings
autoapi_keep_files = False
suppress_warnings = ["autoapi.python_import_resolution"]

# -- Napoleon configuration --------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False  # project standard is Google style
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_title = "2p-preprocess"

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/znamlab/2p-preprocess",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#0077b6",
        "color-brand-content": "#0077b6",
    },
    "dark_css_variables": {
        "color-brand-primary": "#48cae4",
        "color-brand-content": "#48cae4",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/znamlab/2p-preprocess",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0"
                    viewBox="0 0 16 16">
                  <path fill-rule="evenodd"
                    d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17
                    .55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94
                    -.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87
                    2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59
                    .82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27
                    2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08
                    2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73
                    .54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0
                    16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]

# -- Other options -----------------------------------------------------------

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = []
