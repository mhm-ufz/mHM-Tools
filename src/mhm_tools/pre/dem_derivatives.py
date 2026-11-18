"""Derive filled DEM, slope, aspect, flow accumulation and flow direction.

The derivatives come from :mod:`pyflwdir`, except aspect, which pyflwdir does
not provide and which :mod:`mhm_tools.common.topography` computes with the Horn
gradient, as it does the slope. Flow accumulation can
also be derived from an existing flow direction grid, so it follows the same
network as the fdir of an mHM setup, also after upscaling. Grid coordinates and
the CRS variable come from :mod:`mhm_tools.pre.latlon`, so the output carries
the same lat/lon convention as the rest of an mHM setup.

Authors
-------
- Sanjeev Bashyal
"""

from pathlib import Path

import numpy as np
import xarray as xr

from mhm_tools.common.constants import (
    DEM_DERIVATIVE_EXTENSIONS,
    FACC_UNITS,
    FDIR_CODES,
    FDIR_FILLVALUE,
    NO_DATA,
)
from mhm_tools.common.crs_handler import resolve_crs
from mhm_tools.common.file_handler import (
    get_nodata,
    get_raster_data,
    write_xarray_to_ascii,
    write_xarray_to_geotiff,
    write_xarray_to_netcdf,
)
from mhm_tools.common.categorical_raster import _format_categories
from mhm_tools.common.topography import calculate_aspect, calculate_slope
from mhm_tools.pre.latlon import xy_to_latlon

__all__ = [
    "DERIVATIVE_ATTRS",
    "FACC_UNITS",
    "OUTPUT_EXTENSIONS",
    "create_dem_derivatives",
    "create_facc_from_fdir",
]

# kept under its former name, the CLI reads it from common.constants
OUTPUT_EXTENSIONS = DEM_DERIVATIVE_EXTENSIONS

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
    one_file_per_variable=False,
    facc_unit="cell",
    missing_aspect=90.0,
):
    """Write the DEM derivatives for ``input_file`` into ``output_path``.

    ``nc`` writes one file holding every derivative plus the grid coordinates
    and a ``crs`` variable, or with ``one_file_per_variable`` one such file per
    derivative named after it; ``asc`` and ``tif`` always write one file per
    derivative. ``facc_unit`` gives facc in upstream cells or km2, and
    ``missing_aspect`` the aspect of flat cells (NaN or ``NO_DATA`` leaves them
    missing). Returns the paths written.
    """
    extension = _validate_extension(output_extension)
    variable_attrs = create_variable_attrs(facc_unit)
    # flow directions and aspect assume that the first row is the northernmost
    dem = get_raster_data(
        input_file, var_name=var_name, crs=crs, force_decending_y=True
    )
    try:
        resolved_crs = resolve_crs(dem, crs, required=True)
        derivatives = compute_dem_derivatives(
            dem, resolved_crs, facc_unit=facc_unit, missing_aspect=missing_aspect
        )
        grid = _grid_coordinates(dem, resolved_crs)
    finally:
        dem.close()

    output_path = Path(output_path)
    if extension == "nc":
        return _write_netcdf(
            derivatives,
            grid,
            resolved_crs,
            output_path,
            variable_attrs,
            compression=compression,
            one_file_per_variable=one_file_per_variable,
        )
    return _write_per_layer(
        derivatives, grid, resolved_crs, output_path, extension, variable_attrs
    )


def create_facc_from_fdir(
    input_file,
    output_path,
    output_extension="nc",
    crs=None,
    var_name=None,
    compression=None,
    ftype="d8",
    facc_unit="cell",
):
    """Write the flow accumulation of a flow direction grid into ``output_path``.

    Args:
        input_file (str or Path): Flow direction raster (.nc, .asc, .tif).
        output_path (str or Path): Output file for ``nc``, or directory.
        output_extension (str): One of ``OUTPUT_EXTENSIONS``.
        crs (str): CRS to use when the input has none.
        var_name (str): Flow direction variable, e.g. ``flwdir`` or ``fdir``.
        compression (NetcdfCompression): NetCDF compression, or None.
        ftype (str): Flow direction type, one of ``FDIR_CODES``.
        facc_unit (str): ``cell`` for upstream cells or ``km2``.

    Returns
    -------
        Tuple of the written paths: ``facc.nc`` (with grid and CRS), or
        ``facc.asc``/``facc.tif``.
    """
    extension = _validate_extension(output_extension)
    variable_attrs = create_variable_attrs(facc_unit)
    # the flow direction codes assume that the first row is the northernmost
    fdir = get_raster_data(
        input_file, var_name=var_name, crs=crs, force_decending_y=True
    )
    try:
        resolved_crs = resolve_crs(fdir, crs, required=True)
        facc = compute_facc_from_fdir(
            fdir, resolved_crs, ftype=ftype, facc_unit=facc_unit
        )
        grid = _grid_coordinates(fdir, resolved_crs)
    finally:
        fdir.close()

    output_path = Path(output_path)
    if extension == "nc":
        return _write_netcdf(
            {"facc": facc},
            grid,
            resolved_crs,
            output_path,
            variable_attrs,
            compression=compression,
            default_name="facc.nc",
        )
    return _write_per_layer(
        {"facc": facc}, grid, resolved_crs, output_path, extension, variable_attrs
    )


def create_variable_attrs(facc_unit="cell"):
    """Create the output dtype and attributes of every derivative for a facc unit.

    Args:
        facc_unit (str): ``cell`` for upstream cells or ``km2``.

    Returns
    -------
        dict: Variable name -> (dtype, attributes), like ``DERIVATIVE_ATTRS``.
    """
    if facc_unit not in FACC_UNITS:
        msg = (
            f"Unsupported facc unit {facc_unit!r}. Choose one of: "
            f"{', '.join(FACC_UNITS)}."
        )
        raise ValueError(msg)
    variable_attrs = dict(DERIVATIVE_ATTRS)
    if facc_unit == "km2":
        variable_attrs["facc"] = (
            "f8",
            {"long_name": "flow accumulation", "units": "km2"},
        )
    return variable_attrs


def compute_dem_derivatives(dem, crs, facc_unit="cell", missing_aspect=90.0):
    """Return the six derivative arrays for one 2-D DEM.

    facc is in ``facc_unit``, flat cells get the aspect ``missing_aspect``.
    """
    import pyflwdir

    elevation, invalid = get_elevation_values(dem)
    # recalculated, a transform cached in the file may describe the unflipped rows
    transform = dem.rio.transform(recalc=True)
    latlon = bool(crs.is_geographic)
    filled, _ = pyflwdir.dem.fill_depressions(elevation, nodata=NO_DATA)
    flw = pyflwdir.from_dem(
        data=filled, nodata=NO_DATA, transform=transform, latlon=latlon
    )
    # Built one at a time and released as we go: a country-scale DEM makes each
    # of these arrays hundreds of megabytes.
    derivatives = {"dem": _masked(elevation, invalid, "f8")}
    del elevation
    derivatives["slope"] = _masked(
        calculate_slope(filled, transform, latlon), invalid, "f8"
    )
    derivatives["aspect"] = _masked(
        calculate_aspect(filled, transform, latlon, missing_aspect=missing_aspect),
        invalid,
        "f8",
    )
    derivatives["dem_filled"] = _masked(filled, invalid, "f8")
    del filled
    facc_dtype = create_variable_attrs(facc_unit)["facc"][0]
    derivatives["facc"] = _masked(
        flw.upstream_area(unit=facc_unit), invalid, facc_dtype
    )
    derivatives["fdir"] = _masked(flw.to_array(), invalid, "i4")
    return derivatives


def compute_facc_from_fdir(fdir, crs, ftype="d8", facc_unit="cell"):
    """Calculate the flow accumulation of a 2-D flow direction grid.

    Args:
        fdir (xarray.DataArray): Flow direction grid with a rio transform.
        crs (rasterio.crs.CRS): CRS of the grid.
        ftype (str): Flow direction type, one of ``FDIR_CODES``.
        facc_unit (str): ``cell`` for upstream cells or ``km2``.

    Returns
    -------
        numpy.ndarray: Flow accumulation, ``NO_DATA`` where fdir has no data.
    """
    import pyflwdir

    if ftype not in FDIR_CODES:
        msg = f"Unsupported ftype {ftype!r}. Choose one of: {', '.join(FDIR_CODES)}."
        raise ValueError(msg)
    facc_dtype = create_variable_attrs(facc_unit)["facc"][0]
    codes, invalid = create_fdir_codes(fdir.values, ftype, get_nodata(fdir))
    flw = pyflwdir.from_array(
        codes,
        ftype=ftype,
        transform=fdir.rio.transform(recalc=True),
        latlon=bool(crs.is_geographic),
        check_ftype=False,
    )
    return _masked(flw.upstream_area(unit=facc_unit), invalid, facc_dtype)


def get_elevation_values(dem):
    """Get the elevation of a DEM with ``NO_DATA`` in its missing cells.

    Args:
        dem (xarray.DataArray): 2-D DEM; NaN and its declared nodata value mark
            missing cells.

    Returns
    -------
        tuple: The elevation as a float64 numpy.ndarray and the boolean mask of
        the missing cells.
    """
    elevation = np.asarray(dem.values, dtype="float64")
    invalid = ~np.isfinite(elevation)
    nodata = get_nodata(dem)
    if nodata is not None and np.isfinite(nodata):
        invalid |= np.isclose(elevation, float(nodata))
    return np.where(invalid, NO_DATA, elevation), invalid


def create_fdir_codes(values, ftype="d8", nodata=None):
    """Create pyflwdir flow direction codes from the values of a flow direction grid.

    Args:
        values (numpy.ndarray): Flow direction values; NaN, the fill value of the
            flow direction type and ``nodata`` mark missing cells.
        ftype (str): Flow direction type, one of ``FDIR_CODES``.
        nodata (float): Declared nodata value of the grid, or None.

    Returns
    -------
        tuple: The codes as uint8 numpy.ndarray with ``FDIR_FILLVALUE[ftype]`` in
        the missing cells, and the boolean mask of the missing cells.
    """
    values = np.asarray(values, dtype="float64")
    # xarray turns a decoded integer _FillValue into NaN, other files keep it
    invalid = ~np.isfinite(values) | (values == FDIR_FILLVALUE[ftype])
    if nodata is not None and np.isfinite(nodata):
        invalid |= values == float(nodata)
    unknown = ~invalid & ~np.isin(values, FDIR_CODES[ftype])
    if np.any(unknown):
        msg = (
            f"The flow direction grid holds values that are no valid {ftype} codes "
            f"nor its nodata value: {_format_categories(np.unique(values[unknown]))}. "
            "Check --ftype and the declared nodata value of the input."
        )
        raise ValueError(msg)
    codes = np.where(invalid, FDIR_FILLVALUE[ftype], values).astype(np.uint8)
    return codes, invalid


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


def _write_netcdf(
    derivatives,
    grid,
    crs,
    output_path,
    variable_attrs,
    compression=None,
    one_file_per_variable=False,
    default_name="dem_derivatives.nc",
):
    """Write the derivatives plus grid and CRS into NetCDF.

    Args:
        derivatives (dict): Derivative arrays by variable name.
        grid (dict): Grid coordinates from ``_grid_coordinates``.
        crs (rasterio.crs.CRS): CRS of the input grid.
        output_path (Path): Output file, or directory for one file per variable.
        variable_attrs (dict): Variable name -> (dtype, attributes).
        compression (NetcdfCompression): Compression settings, or None.
        one_file_per_variable (bool): Write ``<variable>.nc`` per derivative
            into the ``output_path`` directory instead of one combined file.
        default_name (str): File name used when ``output_path`` is a directory.

    Returns
    -------
        Tuple of the written paths.
    """
    names = [name for name in variable_attrs if name in derivatives]
    coords = {"y": grid["y"], "x": grid["x"]}
    data_vars = {
        name: (("y", "x"), derivatives[name], dict(variable_attrs[name][1]))
        for name in names
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
    for name in names:
        ds[name].attrs.update({"coordinates": "lat lon", "grid_mapping": "crs"})
    axis_units = "m" if grid["projected"] else "degrees"
    ds["x"].attrs.update({"axis": "X", "units": axis_units})
    ds["y"].attrs.update({"axis": "Y", "units": axis_units})
    ds.attrs.update({"Conventions": "CF-1.8", "source": "mhm-tools"})
    encoding = {
        name: {
            "_FillValue": NO_DATA if variable_attrs[name][0] == "f8" else int(NO_DATA)
        }
        for name in names
    }
    if not one_file_per_variable:
        output = _resolve_output(output_path, default_name)
        write_xarray_to_netcdf(ds, output, encoding=encoding, compression=compression)
        return (output,)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in names:
        # every file keeps the grid and CRS, so each one is usable on its own
        output = output_dir / f"{name}.nc"
        write_xarray_to_netcdf(
            ds[[name, "lat", "lon", "crs"]],
            output,
            encoding={name: encoding[name]},
            compression=compression,
        )
        written.append(output)
    return tuple(written)


def _write_per_layer(derivatives, grid, crs, output_path, extension, variable_attrs):
    """Write one file per derivative, as ASCII grids or GeoTIFFs.

    Args:
        derivatives (dict): Derivative arrays by variable name.
        grid (dict): Grid coordinates from ``_grid_coordinates``.
        crs (rasterio.crs.CRS): CRS of the input grid.
        output_path (Path): Output directory.
        extension (str): ``asc`` or ``tif``.
        variable_attrs (dict): Variable name -> (dtype, attributes).

    Returns
    -------
        Tuple of the written paths.
    """
    folder = Path(output_path)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for name, (_dtype, attrs) in variable_attrs.items():
        if name not in derivatives:
            continue
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
