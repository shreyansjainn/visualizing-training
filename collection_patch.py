
from sphinx.application import Sphinx
from collections import abc

def patch_napoleon(app):
    import sphinxcontrib.napoleon.docstring as napoleon_docstring
    if not hasattr(napoleon_docstring, 'Callable'):
        napoleon_docstring.Callable = abc.Callable

def setup(app: Sphinx):
    app.connect('builder-inited', patch_napoleon)
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }