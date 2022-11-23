"""
Tools to pre- and post-process data for and from mHM.

.. toctree::
   :hidden:

   self
"""

try:
    from ._version import __version__
except ModuleNotFoundError:  # pragma: no cover
    # package is not installed
    __version__ = "0.0.0.dev0"


__all__ = ["__version__"]
