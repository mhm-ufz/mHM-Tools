"""Common vector rasterization helper."""

import numpy as np

from mhm_tools.common.constants import NO_DATA


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
