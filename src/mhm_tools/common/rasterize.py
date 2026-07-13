"""Common rasterization and raster-alignment helpers."""

from pathlib import Path

import numpy as np

from mhm_tools.common.constants import NO_DATA


def _safe_raster_dtype(dtype, nodata):
    """Return a GDAL-compatible numeric dtype that can represent ``nodata``."""
    dtype = np.dtype(dtype)
    if dtype.kind not in "iuf":
        msg = f"Raster data must have a real numeric dtype; found {dtype}."
        raise TypeError(msg)
    if dtype == np.dtype("float16"):
        dtype = np.dtype("float32")
    if nodata is None:
        return dtype

    try:
        number = float(nodata)
    except (TypeError, ValueError) as exc:
        msg = f"Raster nodata must be numeric; found {nodata!r}."
        raise TypeError(msg) from exc

    if dtype.kind in "iu":
        info = np.iinfo(dtype)
        representable = (
            np.isfinite(number)
            and number.is_integer()
            and info.min <= number <= info.max
        )
    else:
        info = np.finfo(dtype)
        representable = not np.isfinite(number) or abs(number) <= info.max
    if representable:
        return dtype

    promoted = np.promote_types(dtype, np.min_scalar_type(nodata))
    if promoted == np.dtype("float16"):
        promoted = np.dtype("float32")
    if promoted.kind not in "iuf" or promoted.itemsize > 8:
        msg = f"No supported raster dtype can safely represent {dtype} and {nodata!r}."
        raise ValueError(msg)
    if dtype.kind in "iu" and promoted.kind == "f":
        msg = f"No integer raster dtype can safely represent {dtype} and {nodata!r}."
        raise ValueError(msg)
    return promoted


def _reference_profile(reference_file, dtype, nodata):
    """Read a reference raster and return its grid plus an output profile."""
    import rasterio

    reference_path = Path(reference_file)
    if not reference_path.is_file():
        msg = f"Reference raster does not exist: {reference_path}"
        raise FileNotFoundError(msg)

    with rasterio.open(reference_path) as reference:
        if reference.width < 1 or reference.height < 1:
            msg = f"Reference raster has an empty grid: {reference_path}"
            raise ValueError(msg)
        if reference.crs is None:
            msg = (
                f"Reference raster has no coordinate reference system: {reference_path}"
            )
            raise ValueError(msg)
        transform_values = np.asarray(tuple(reference.transform)[:6], dtype=float)
        if (
            not np.all(np.isfinite(transform_values))
            or reference.transform.is_degenerate
        ):
            msg = f"Reference raster has an invalid affine transform: {reference_path}"
            raise ValueError(msg)

        shape = (reference.height, reference.width)
        crs = reference.crs
        transform = reference.transform
        profile = reference.profile.copy()

    profile.update(
        driver="GTiff",
        height=shape[0],
        width=shape[1],
        count=1,
        dtype=np.dtype(dtype).name,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="deflate",
    )
    # These source-specific creation options can be invalid for a single-band output.
    profile.pop("photometric", None)
    profile.pop("interleave", None)
    return shape, crs, transform, profile


def write_array_to_reference_geotiff(
    data, reference_file, output_file, nodata=NO_DATA
) -> Path:
    """Write a two-dimensional array on the exact grid of a reference raster.

    Parameters
    ----------
    data : array-like
        Two-dimensional numeric raster values.
    reference_file : path-like
        Raster providing the output CRS, transform, and dimensions.
    output_file : path-like
        Destination GeoTIFF.
    nodata : number, default -9999
        Output nodata value.

    Returns
    -------
    pathlib.Path
        Path to the created GeoTIFF.
    """
    import rasterio

    output_path = Path(output_file)
    reference_path = Path(reference_file)
    if output_path.suffix.lower() not in {".tif", ".tiff"}:
        msg = f"Output file must be a GeoTIFF: {output_path}"
        raise ValueError(msg)
    if output_path.resolve() == reference_path.resolve():
        msg = "Output file must differ from the reference raster."
        raise ValueError(msg)

    values = np.asanyarray(data)
    if values.ndim != 2:
        msg = f"Raster data must be two-dimensional; found shape {values.shape}."
        raise ValueError(msg)
    dtype = _safe_raster_dtype(values.dtype, nodata)
    shape, _crs, _transform, profile = _reference_profile(reference_path, dtype, nodata)
    if values.shape != shape:
        msg = f"Raster data shape {values.shape} does not match reference grid {shape}."
        raise ValueError(msg)

    if np.ma.isMaskedArray(values):
        if nodata is None:
            msg = "Masked raster data require a nodata value."
            raise ValueError(msg)
        values = values.filled(nodata)
    values = np.asarray(values, dtype=dtype)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as output:
        output.write(values, 1)
    return output_path


def align_raster_to_reference(input_file, reference_file, output_file) -> Path:
    """Align a single-band raster to a reference grid using nearest neighbor.

    Parameters
    ----------
    input_file : path-like
        Single-band source raster.
    reference_file : path-like
        Raster providing the exact destination grid.
    output_file : path-like
        Destination GeoTIFF.

    Returns
    -------
    pathlib.Path
        Path to the aligned GeoTIFF.
    """
    import rasterio
    from rasterio.warp import Resampling, reproject

    input_path = Path(input_file)
    reference_path = Path(reference_file)
    output_path = Path(output_file)
    if not input_path.is_file():
        msg = f"Input raster does not exist: {input_path}"
        raise FileNotFoundError(msg)
    if output_path.resolve() in {input_path.resolve(), reference_path.resolve()}:
        msg = "Output file must differ from the input and reference rasters."
        raise ValueError(msg)

    shape, reference_crs, reference_transform, _profile = _reference_profile(
        reference_path, np.int32, int(NO_DATA)
    )
    with rasterio.open(input_path) as source:
        if source.count != 1:
            msg = f"Input raster must contain exactly one band: {input_path}"
            raise ValueError(msg)
        if source.crs is None:
            msg = f"Input raster has no coordinate reference system: {input_path}"
            raise ValueError(msg)

        nodata = source.nodata if source.nodata is not None else int(NO_DATA)
        dtype = _safe_raster_dtype(source.dtypes[0], nodata)
        destination = np.full(shape, nodata, dtype=dtype)
        reproject(
            source=rasterio.band(source, 1),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=reference_transform,
            dst_crs=reference_crs,
            dst_nodata=nodata,
            resampling=Resampling.nearest,
        )

    return write_array_to_reference_geotiff(
        destination, reference_path, output_path, nodata=nodata
    )


def rasterize_vector(frame, mapping_field, out_shape, transform):
    """Rasterize a numeric vector attribute onto an existing grid.

    Parameters
    ----------
    frame : geopandas.GeoDataFrame
        Vector features in the target grid's coordinate reference system.
    mapping_field : str
        Attribute whose values are burned into the raster.
    out_shape : tuple of int
        Raster shape as ``(height, width)``.
    transform : affine.Affine
        Pixel-to-coordinate transform of the target grid.

    Returns
    -------
    numpy.ndarray
        Rasterized values as a two-dimensional ``int32`` array.
    """
    from rasterio.features import rasterize

    if frame.empty:
        msg = "The input vector contains no features."
        raise ValueError(msg)
    if mapping_field not in frame.columns:
        msg = f"Mapping field {mapping_field!r} is not present in the input vector."
        raise ValueError(msg)

    try:
        numeric_values = np.asarray(frame[mapping_field], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        msg = f"Mapping field {mapping_field!r} must contain numeric values."
        raise ValueError(msg) from exc

    if not np.all(np.isfinite(numeric_values)):
        msg = f"Mapping field {mapping_field!r} contains non-finite values."
        raise ValueError(msg)
    rounded_values = np.rint(numeric_values)
    if not np.array_equal(numeric_values, rounded_values):
        msg = f"Mapping field {mapping_field!r} must contain integral values."
        raise ValueError(msg)

    int32 = np.iinfo(np.int32)
    if np.any((rounded_values < int32.min) | (rounded_values > int32.max)):
        msg = f"Mapping field {mapping_field!r} contains values outside int32 range."
        raise ValueError(msg)

    nodata = int(NO_DATA)
    values = rounded_values.astype(np.int32)
    if np.any(values == nodata):
        msg = f"Mapping field {mapping_field!r} contains the nodata value {nodata}."
        raise ValueError(msg)

    return rasterize(
        zip(frame.geometry, values),
        out_shape=out_shape,
        transform=transform,
        fill=nodata,
        dtype="int32",
        all_touched=False,
    )
