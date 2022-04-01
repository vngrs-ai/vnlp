# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

# mock deps with system level requirements.
# autodoc_mock_imports = ["tensorflow"]

# -- Project information

project = 'VNLP'
author = 'Meliksah Turker'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.autosectionlabel'
]

# Make sure the target is unique
autosectionlabel_prefix_document = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'TODO/*']

source_suffix = [".rst", ".md"]

# -- Options for HTML output
# Stanford Theme
#import sphinx_theme
#html_theme = 'stanford_theme'
#html_theme_path = [sphinx_theme.get_html_theme_path('stanford-theme')]

# sphinx readthedocs theme
html_theme = 'sphinx_rtd_theme'

html_logo = '_static/vnlp_white.png'

# Below html_theme_options config depends on the theme.
# For Stanford theme:
# https://sphinx-rtd-theme.readthedocs.io/en/stable/configuring.html
html_theme_options = {
    'logo_only': True,
    'display_version': True
}

# -- Options for EPUB output
epub_show_urls = 'footnote'
