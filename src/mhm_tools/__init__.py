"""Tools to pre- and post-process data for and from mHM.

.. toctree::
   :hidden:

   self

Subpackages
===========

Built-in processing and tool functions.

.. autosummary::
   :toctree: api
   :caption: Subpackages

   common
   post
   pre
"""

try:
    from ._version import __version__
except ModuleNotFoundError:  # pragma: no cover
    # package is not installed
    __version__ = "not_available"

from . import common, post, pre

__all__ = ["__version__"]
__all__ += ["common", "post", "pre"]
