# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
from types import ModuleType
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('_ext'))

# Create a fake collections module
collections = ModuleType('collections')
sys.modules['collections'] = collections

# Import the real collections
import collections as real_collections

# Add everything from the real collections to our fake one
for attr in dir(real_collections):
    setattr(collections, attr, getattr(real_collections, attr))

# Add Callable from collections.abc
from collections.abc import Callable
collections.Callable = Callable

project = 'Visualize Training'
copyright = '2024, shreyans jain'
author = 'shreyans jain'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['myst_parser', "sphinx.ext.autodoc",'collection_patch','sphinxcontrib.napoleon','sphinx.ext.doctest']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]
