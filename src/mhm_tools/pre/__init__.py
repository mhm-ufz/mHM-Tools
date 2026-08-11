"""Pre processing routines for mHM.

Files
=====

.. autosummary::
   :toctree:

   ~mhm_tools.pre.catchment
   ~mhm_tools.pre.create_id_gauges
   ~mhm_tools.pre.create_mhm_restart_file
   ~mhm_tools.pre.create_mhm_restart_from_setup
   ~mhm_tools.pre.crop_mhm_setup
   ~mhm_tools.pre.fill_nearest
   ~mhm_tools.pre.format_geology
   ~mhm_tools.pre.format_lc_data
   ~mhm_tools.pre.format_soil
   ~mhm_tools.pre.landcover_ascii_to_nc
   ~mhm_tools.pre.latlon
   ~mhm_tools.pre.link_folder_tree
   ~mhm_tools.pre.merge
   ~mhm_tools.pre.pet_calc
   ~mhm_tools.pre.prepare_mhm_forcings
   ~mhm_tools.pre.rasterize_map
   ~mhm_tools.pre.regrid
   ~mhm_tools.pre.subdomain_masks
"""

import importlib

from .format_lc_data import format_lc_data, format_lc_periods

_MODULE_EXPORTS = {
    "catchment": "catchment",
    "create_mhm_restart_from_setup": "create_mhm_restart_from_setup",
    "create_mhm_restart_file": "create_mhm_restart_file",
    "fill_nearest": "fill_nearest",
    "format_geology": "format_geology",
    "format_soil": "format_soil",
    "latlon": "latlon",
    "rasterize_map": "rasterize_map",
    "subdomain_masks": "subdomain_masks",
}

_ATTR_EXPORTS = {
    "create_catchment": ("catchment", "create_catchment"),
    "merge_catchment": ("catchment", "merge_catchment"),
    "create_id_gauges": ("create_id_gauges", "create_id_gauges"),
    "Grid": ("create_mhm_restart_file", "Grid"),
    "LatLon": ("create_mhm_restart_file", "LatLon"),
    "MHMRestartFile": ("create_mhm_restart_file", "MHMRestartFile"),
    "MHMRunner": ("create_mhm_restart_from_setup", "MHMRunner"),
    "MorphFiles": ("create_mhm_restart_file", "MorphFiles"),
    "crop_mhm_setup": ("crop_mhm_setup", "crop_mhm_setup"),
    "create_latlon": ("latlon", "create_latlon"),
    "xy_to_latlon": ("latlon", "xy_to_latlon"),
    "link_folder_tree": ("link_folder_tree", "link_folder_tree"),
    "create_subdomain_masks": ("subdomain_masks", "create_subdomain_masks"),
    "fill_nearest": ("fill_nearest", "fill_dataarray_with_nearest"),
    "format_geology_data": ("format_geology", "format_geology_data"),
    "write_geology_classdefinition": (
        "format_geology",
        "write_geology_classdefinition",
    ),
    "format_soil_data": ("format_soil", "format_soil_data"),
    "format_soil_horizons": ("format_soil", "format_soil_horizons"),
    "write_soil_classdefinition": (
        "format_soil",
        "write_soil_classdefinition",
    ),
    "rasterize_map_data": ("rasterize_map", "rasterize_map_data"),
}

__all__ = [
    "Grid",
    "LatLon",
    "MHMRestartFile",
    "MHMRunner",
    "MorphFiles",
    "catchment",
    "create_catchment",
    "create_id_gauges",
    "create_latlon",
    "create_mhm_restart_file",
    "create_mhm_restart_from_setup",
    "create_subdomain_masks",
    "crop_mhm_setup",
    "fill_nearest",
    "format_geology",
    "format_geology_data",
    "format_lc_data",
    "format_lc_periods",
    "format_soil",
    "format_soil_data",
    "format_soil_horizons",
    "latlon",
    "link_folder_tree",
    "merge_catchment",
    "rasterize_map",
    "rasterize_map_data",
    "subdomain_masks",
    "write_geology_classdefinition",
    "write_soil_classdefinition",
    "xy_to_latlon",
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
