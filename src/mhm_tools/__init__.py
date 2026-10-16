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

   ~mhm_tools.common
   ~mhm_tools.post
   ~mhm_tools.pre
"""

try:
    from ._version import __version__
except ModuleNotFoundError:  # pragma: no cover
    # package is not installed
    __version__ = "not_available"

import importlib

# the subpackages load xarray, dask and co., so they are imported on first access
_SUBPACKAGES = ("common", "post", "pre")

__all__ = ["__version__"]
__all__ += list(_SUBPACKAGES)


def __getattr__(name):
    """Import a subpackage on first access.

    Args:
        name (str): Attribute name.

    Returns
    -------
        The subpackage module.
    """
    if name in _SUBPACKAGES:
        return importlib.import_module(f".{name}", __name__)
    error_msg = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(error_msg)


def __dir__():
    """List the package attributes including the not yet imported subpackages."""
    return sorted(set(globals().keys()) | set(__all__))
