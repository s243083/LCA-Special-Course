"""Sphinx configuration for WINPACT documentation."""
from __future__ import annotations

import os
import sys
from datetime import datetime

# Anchor on conf.py's location, NOT on CWD — sphinx-build is invoked from
# different working directories locally (docs/) vs. CI (repo root), and
# os.path.abspath("..") resolves against CWD. Using __file__ makes the
# import path stable in both cases.
_DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_DOCS_DIR, "..")))

project = "WINPACT"
author = "DTU Wind and Energy Systems – Integrity and Reliability Section"
copyright = f"{datetime.now().year}, {author}"

try:
    from importlib.metadata import version as _pkg_version
    release = _pkg_version("winpact")
except Exception:
    release = "0.0.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "myst_parser",
    "nbsphinx",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_autodoc_typehints",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_mock_imports = [
    # Only mock packages that are NOT installed in the docs build env.
    # Mocking an installed package (e.g. pyarrow) can break other packages
    # that read its real attributes (pandas reads pa.__version__).
    "maesopt",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
    "substitution",
    "tasklist",
]

nbsphinx_execute = "never"  # flip to "auto" once demo inputs are wired
nbsphinx_allow_errors = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pywake": ("https://topfarm.pages.windenergy.dtu.dk/PyWake/", None),
}

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "WINPACT"
html_logo = None
html_favicon = None
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "titles_only": False,
}

todo_include_todos = True
