# Configuration file for the Sphinx documentation builder.
#
# This configuration is intentionally minimal and focused on
# automatic documentation generation from docstrings.

import os
import sys

# Add project root to PYTHONPATH so Sphinx can import the modules
sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------

project = "Titanic Survival Analysis"
copyright = "2026, Rayane Benbakir, Oussama Khabouch"
author = "Rayane Benbakir, Oussama Khabouch"
release = "1.4"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",    # Automatic documentation from docstrings
    "sphinx.ext.napoleon",   # Support for Google / NumPy style docstrings
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
