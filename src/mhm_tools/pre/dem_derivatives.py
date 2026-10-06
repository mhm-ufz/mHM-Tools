"""Derive filled DEM, slope, aspect, flow accumulation and flow direction.

The derivatives come from :mod:`pyflwdir`, except aspect, which pyflwdir does
not provide and which is computed with the Horn gradient. Grid coordinates and
the CRS variable come from :mod:`mhm_tools.pre.latlon`, so the output carries
the same lat/lon convention as the rest of an mHM setup.

Authors
-------
- Sanjeev Bashyal
"""

from pathlib import Path

import numpy as np
import xarray as xr

from mhm_tools.common.crs_handler import resolve_crs
from mhm_tools.common.file_handler import (
    get_nodata,
    get_raster_data,
    write_xarray_to_ascii,
    write_xarray_to_geotiff,
    write_xarray_to_netcdf,
)
from mhm_tools.pre.latlon import xy_to_latlon

__all__ = ["DERIVATIVE_ATTRS", "OUTPUT_EXTENSIONS", "create_dem_derivatives"]

NO_DATA = -9999.0
OUTPUT_EXTENSIONS = ("nc", "asc", "tif")

#: Output variable -> (dtype, attributes). ``dem`` is the input as read.
DERIVATIVE_ATTRS = {
    "dem": ("f8", {"long_name": "elevation", "units": "m"}),
    "dem_filled": ("f8", {"long_name": "depression filled elevation", "units": "m"}),
    "slope": ("f8", {"long_name": "slope", "units": "%"}),
    "aspect": ("f8", {"long_name": "aspect", "units": "degree"}),
    "facc": ("i4", {"long_name": "flow accumulation", "units": "1"}),
    "fdir": ("i4", {"long_name": "d8 flow direction", "units": "1"}),
}


def create_dem_derivatives(
    input_file,
    output_path,
    output_extension="nc",
    crs=None,
    var_name=None,
    compression=None,
):
    """Write the DEM derivatives for ``input_file`` into ``output_path``.

    ``nc`` writes one file holding every derivative plus the grid coordinates
    and a ``crs`` variable; ``asc`` and ``tif`` write one file per derivative.
    Returns the paths written.
    """
    extension = _validate_extension(output_extension)
    dem = get_raster_data(input_file, var_name=var_name, crs=crs)
    try:
        resolved_crs = resolve_crs(dem, crs, required=True)
        derivatives = compute_dem_derivatives(dem, resolved_crs)
        grid = _grid_coordinates(dem, resolved_crs)
    finally:
        dem.close()

    output_path = Path(output_path)
    if extension == "nc":
        return (
            _write_netcdf(
                derivatives, grid, resolved_crs, output_path, compression=compression
            ),
        )
    return _write_per_layer(derivatives, grid, resolved_crs, output_path, extension)


def compute_dem_derivatives(dem, crs):
    """Return the six derivative arrays for one 2-D DEM."""
    import pyflwdir

    elevation = np.asarray(dem.values, dtype="float64")
    invalid = ~np.isfinite(elevation)
    nodata = get_nodata(dem)
    if nodata is not None and np.isfinite(nodata):
        invalid |= np.isclose(elevation, float(nodata))
    elevation = np.where(invalid, NO_DATA, elevation)

    transform = dem.rio.transform()
    transform_array = np.asarray(tuple(transform), dtype=np.float64)  # affine v3 fix
    latlon = bool(crs.is_geographic)
    filled, _ = pyflwdir.dem.fill_depressions(elevation, nodata=NO_DATA)
    flw = pyflwdir.from_dem(
        data=filled, nodata=NO_DATA, transform=transform, latlon=latlon
    )
    # Built one at a time and released as we go: a country-scale DEM makes each
    # of these arrays hundreds of megabytes.
    derivatives = {"dem": _masked(elevation, invalid, "f8")}
    del elevation
    slope = pyflwdir.dem.slope(
        filled, nodata=NO_DATA, latlon=latlon, transform=transform_array
    )
    slope *= 100.0
    derivatives["slope"] = _masked(slope, invalid, "f8")
    del slope
    derivatives["aspect"] = _masked(
        horn_aspect(filled, transform, NO_DATA), invalid, "f8"
    )
    derivatives["dem_filled"] = _masked(filled, invalid, "f8")
    del filled
    derivatives["facc"] = _masked(flw.upstream_area(unit="cell"), invalid, "i4")
    derivatives["fdir"] = _masked(flw.to_array(), invalid, "i4")
    return derivatives


def horn_aspect(elevation, transform, nodata=NO_DATA):
    """Return aspect in degrees clockwise from north, by Horn's method.

    pyflwdir has no aspect, and Horn's 3x3 kernel is what GDAL uses, so the
    result matches the convention the rest of an mHM setup expects.
    """
    values = np.where(np.isclose(elevation, nodata), np.nan, elevation)
    padded = np.pad(values, 1, mode="edge")
    del values
    # Horn weights the four side neighbours twice as heavily as the corners.
    dz_dx = (
        (padded[:-2, 2:] + 2 * padded[1:-1, 2:] + padded[2:, 2:])
        - (padded[:-2, :-2] + 2 * padded[1:-1, :-2] + padded[2:, :-2])
    ) / (8.0 * abs(transform.a))
    dz_dy = (
        (padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:])
        - (padded[:-2, :-2] + 2 * padded[:-2, 1:-1] + padded[:-2, 2:])
    ) / (8.0 * abs(transform.e))
    del padded
    # A flat cell has no aspect; report nodata rather than an arbitrary bearing.
    flat = (dz_dx == 0) & (dz_dy == 0)
    # Compass bearing: 0 at north, increasing clockwise.
    aspect = (90.0 - np.degrees(np.arctan2(dz_dy, -dz_dx))) % 360.0
    del dz_dx, dz_dy
    aspect[flat | ~np.isfinite(aspect)] = nodata
    return aspect


def _masked(values, invalid, dtype):
    """Return ``values`` in its output dtype with the DEM's invalid cells blanked.

    Masking in place keeps one array per derivative rather than the two a
    ``np.where`` would leave alive at once.
    """
    out = np.asarray(values).astype(dtype, copy=True)
    out[invalid] = NO_DATA
    return out


def _validate_extension(output_extension):
    extension = str(output_extension or "nc").strip().lower().lstrip(".")
    if extension not in OUTPUT_EXTENSIONS:
        msg = (
            f"Unsupported output extension {output_extension!r}. "
            f"Choose one of: {', '.join(OUTPUT_EXTENSIONS)}."
        )
        raise ValueError(msg)
    return extension


def _grid_coordinates(dem, crs):
    """Return the coordinate variables describing the DEM grid."""
    y_name, x_name = dem.dims
    x = np.asarray(dem[x_name].values, dtype="float64")
    y = np.asarray(dem[y_name].values, dtype="float64")
    x_grid, y_grid = np.meshgrid(x, y)
    projection = None if crs.is_geographic else crs.to_wkt()
    lons, lats = xy_to_latlon(x_grid, y_grid, projection)
    return {
        "x": x,
        "y": y,
        "lon": np.asarray(lons, dtype="float64"),
        "lat": np.asarray(lats, dtype="float64"),
        "projected": not crs.is_geographic,
    }


def _write_netcdf(derivatives, grid, crs, output_path, compression=None):
    """Write every derivative plus grid and CRS into one NetCDF file."""
    output = _resolve_output(output_path, "dem_derivatives.nc")
    coords = {"y": grid["y"], "x": grid["x"]}
    data_vars = {
        name: (("y", "x"), derivatives[name], dict(attrs))
        for name, (_dtype, attrs) in DERIVATIVE_ATTRS.items()
    }
    data_vars["lon"] = (
        ("y", "x"),
        grid["lon"],
        {
            "long_name": "longitude",
            "standard_name": "longitude",
            "units": "degrees_east",
        },
    )
    data_vars["lat"] = (
        ("y", "x"),
        grid["lat"],
        {
            "long_name": "latitude",
            "standard_name": "latitude",
            "units": "degrees_north",
        },
    )
    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    wkt = crs.to_wkt()
    ds["crs"] = xr.DataArray(0, attrs={"crs_wkt": wkt, "spatial_ref": wkt})
    for name in DERIVATIVE_ATTRS:
        ds[name].attrs.update({"coordinates": "lat lon", "grid_mapping": "crs"})
    axis_units = "m" if grid["projected"] else "degrees"
    ds["x"].attrs.update({"axis": "X", "units": axis_units})
    ds["y"].attrs.update({"axis": "Y", "units": axis_units})
    ds.attrs.update({"Conventions": "CF-1.8", "source": "mhm-tools"})
    encoding = {
        name: {"_FillValue": NO_DATA if dtype == "f8" else int(NO_DATA)}
        for name, (dtype, _attrs) in DERIVATIVE_ATTRS.items()
    }
    write_xarray_to_netcdf(ds, output, encoding=encoding, compression=compression)
    return output


def _write_per_layer(derivatives, grid, crs, output_path, extension):
    """Write one file per derivative, as ASCII grids or GeoTIFFs."""
    folder = Path(output_path)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for name, (_dtype, attrs) in DERIVATIVE_ATTRS.items():
        # The coordinates must travel with the array: without them the writers
        # invent a 0..n index grid and flip the row order.
        data = xr.DataArray(
            derivatives[name],
            dims=("y", "x"),
            coords={"y": grid["y"], "x": grid["x"]},
            name=name,
            attrs={**attrs, "nodata_value": NO_DATA, "_FillValue": NO_DATA},
        )
        target = folder / f"{name}.{extension}"
        if extension == "asc":
            write_xarray_to_ascii(data, target, nodata_value=NO_DATA, crs=crs)
        else:
            write_xarray_to_geotiff(data, target, nodata_value=NO_DATA, crs=crs)
        written.append(target)
    return tuple(written)


def _resolve_output(output_path, default_name):
    """Return the output file, treating a directory or bare name sensibly."""
    output_path = Path(output_path)
    if output_path.suffix:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path / default_name
