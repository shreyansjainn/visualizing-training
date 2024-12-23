# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
import types
from collections import abc
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath('../'))

# class PatchedCollections(types.ModuleType):
#     def __getattr__(self, name):
#         if name == 'Callable':
#             return abc.Callable
#         if name == 'deque':
#             return abc.deque
#         return getattr(abc, name)

# sys.modules['collections'] = PatchedCollections('collections')

project = 'Visualizing Training'
copyright = '2024, shreyans jain'
author = 'shreyans jain'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['myst_parser', "sphinx.ext.autodoc",'sphinx.ext.napoleon','sphinx.ext.doctest']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]
