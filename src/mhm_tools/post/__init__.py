"""Post processing routines for mHM.

Bankfull discharge
==================

.. autosummary::
   :toctree:

    bankfull_discharge

Notes
-----
This module lazily imports submodules and attributes to keep import time low.
"""

import importlib

_MODULE_EXPORTS = {
    "bankfull": "bankfull",
}

_ATTR_EXPORTS = {
    "bankfull_discharge": ("bankfull", "bankfull_discharge"),
    "evaludate_grdc_data": ("GRDC_validation", "evaludate_grdc_data"),
    "gridded_data_evaluation": ("gridded_data_evaluation", "gridded_data_evaluation"),
    "Hydrograph": ("hydrograph", "Hydrograph"),
}

__all__ = [
    "Hydrograph",
    "bankfull",
    "bankfull_discharge",
    "evaludate_grdc_data",
    "gridded_data_evaluation",
]


def __getattr__(name):
    if name in _MODULE_EXPORTS:
        module = importlib.import_module(f".{_MODULE_EXPORTS[name]}", __name__)
        globals()[name] = module
        return module
    if name in _ATTR_EXPORTS:
        module_name, attr_name = _ATTR_EXPORTS[name]
        module = importlib.import_module(f".{module_name}", __name__)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    error_msg = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(error_msg)


def __dir__():
    return sorted(set(globals().keys()) | set(__all__))
