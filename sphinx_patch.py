# save this as sphinx_patch.py in your repository
import sys
from types import ModuleType

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

# Now import napoleon, which will use our patched collections
import sphinxcontrib.napoleon