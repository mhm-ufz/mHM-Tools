"""
Fill missing NetCDF cells from nearest valid neighbours.

This interpolator should not be used on global scale as it will be slow.

The module fills spatial gaps in gridded NetCDF variables by copying values
from the nearest valid source cell. For variables with a ``time`` dimension,
the nearest source cells are selected from the first time slice and then reused
for all time steps, preserving temporal variability at the selected source
locations.

An optional mask can reserve cells that must remain outside the filled domain.
Those masked cells are written with the configured fill value instead of being
nearest-neighbour filled.

Authors
- Simon Lüdke
- Sebastian Müller
"""

import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from scipy.spatial import KDTree

from mhm_tools.common.logger import ErrorLogger, log_arguments
from mhm_tools.common.xarray_utils import get_single_data_var

logger = logging.getLogger(__name__)


def _coordinate_mesh(coords, names):
    """Return coordinate position arrays for spatial nearest-neighbour lookup.

    Parameters
    ----------
    coords : xarray.core.coordinates.DataArrayCoordinates
        Coordinate mapping from the target data array.
    names : tuple[str, ...]
        Dimension names used to construct the KD-tree positions.

    Returns
    -------
    list[numpy.ndarray] or tuple[numpy.ndarray, ...]
        Coordinate arrays aligned with the dimensions in ``names``.
    """
    generate_mesh = all(name in coords and name in coords[name].dims for name in names)
    positions = [coords[name].data for name in names]
    return np.meshgrid(*positions, indexing="ij") if generate_mesh else positions


def nearest_indices(input_positions, target_positions):
    """Return nearest source indices for target coordinate positions.

    Parameters
    ----------
    input_positions : numpy.ndarray
        Two-dimensional array of valid source positions with shape
        ``(n_source, n_dimensions)``.
    target_positions : numpy.ndarray
        Two-dimensional array of target positions with shape
        ``(n_target, n_dimensions)``.

    Returns
    -------
    numpy.ndarray
        Indices into ``input_positions`` for the nearest source cell of each
        target position.
    """
    tree = KDTree(input_positions)
    return tree.query(target_positions)[1]


def _valid_mask(array, along_time, missing_value):
    """Build a boolean mask for cells that can be used as fill sources.

    Parameters
    ----------
    array : xarray.DataArray
        Data array whose finite non-missing cells are potential source cells.
    along_time : bool
        If true, derive source validity from the first time slice only.
    missing_value : float or None
        Explicit missing-value marker to exclude in addition to NaN.

    Returns
    -------
    numpy.ndarray
        Boolean mask with true values at valid source cells.
    """
    values = array[{"time": 0}].data if along_time else array.data
    valid = ~np.isnan(values)
    if missing_value is not None and not np.isnan(missing_value):
        valid &= ~np.isclose(values, missing_value)
    return valid


def fill_dataarray_with_nearest(
    array,
    along_time=None,
    missing_value=None,
    mask=None,
    fill_value=-9999.0,
    source_file=None,
):
    """Fill missing values in a data array with nearest valid neighbours.

    Parameters
    ----------
    array : xarray.DataArray
        Data array modified in place.
    along_time : bool, optional
        Whether to treat all non-time dimensions as the fill domain and copy
        the selected source-cell time series to each target cell. Defaults to
        true when ``array`` has a ``time`` dimension.
    missing_value : float, optional
        Missing-value marker to fill. NaNs are always treated as missing.
    mask : numpy.ndarray, optional
        Boolean spatial mask where true cells are excluded from filling and set
        to ``fill_value``.
    fill_value : float, default -9999.0
        Value written to masked cells and to target cells when no valid source
        cells exist.
    source_file : str or pathlib.Path, optional
        File path used only for diagnostic log messages.

    Returns
    -------
    int
        Number of unmasked cells selected for nearest-neighbour filling.
    """
    coord_names = array.dims
    along_time = "time" in coord_names if along_time is None else bool(along_time)
    if along_time:
        coord_names = tuple(name for name in coord_names if name != "time")

    valid = _valid_mask(array, along_time=along_time, missing_value=missing_value)
    target = ~valid
    if mask is not None:
        target &= ~mask

    if not np.any(target):
        logger.debug(f"No cells require nearest-neighbour filling for {array.name}.")
        return 0

    filled_count = int(np.sum(target))
    valid_count = int(np.sum(valid))
    if valid_count == 0:
        context = f" in {source_file}" if source_file is not None else ""
        logger.warning(
            f"Cannot nearest-neighbour fill {filled_count} cells in variable "
            f"{array.name}{context}: no valid source cells are available. Setting "
            f"target cells to {fill_value}. dims={array.dims} shape={array.shape} "
            f"missing_value={missing_value} mask={mask is not None}"
        )
        values = array.transpose("time", ...).data if along_time else array.data
        if along_time:
            values[:, target] = fill_value
            if mask is not None:
                values[:, mask] = fill_value
        else:
            values[target] = fill_value
            if mask is not None:
                values[mask] = fill_value
        return 0

    positions = _coordinate_mesh(array.coords, coord_names)
    valid_positions = np.array([position[valid] for position in positions]).T
    target_positions = np.array([position[target] for position in positions]).T
    target_indices = nearest_indices(valid_positions, target_positions)

    values = array.transpose("time", ...).data if along_time else array.data
    if along_time:
        values[:, target] = values[:, valid][:, target_indices]
        if mask is not None:
            values[:, mask] = fill_value
    else:
        values[target] = values[valid][target_indices]
        if mask is not None:
            values[mask] = fill_value

    source_context = f" from {source_file}" if source_file is not None else ""
    logger.info(
        f"Filled {filled_count} cells in {array.name}{source_context} using "
        f"{valid_count} valid source cells."
    )
    return filled_count


def read_mask(mask_file, mask_var):
    """Read a fixed output mask from a NetCDF variable.

    Parameters
    ----------
    mask_file : str or pathlib.Path or None
        NetCDF file containing the mask variable. If omitted, no mask is used.
    mask_var : str or None
        Variable whose NaN cells define masked output cells. If omitted, no
        mask is used.

    Returns
    -------
    numpy.ndarray or None
        Boolean array with true values where output cells must remain masked,
        or ``None`` when masking is disabled.
    """
    if mask_file is None or mask_var is None:
        return None

    mask_file = Path(mask_file)
    with xr.open_dataset(mask_file, engine="netcdf4") as dataset:
        if mask_var is None:
            mask_var = get_single_data_var(dataset)
        if mask_var not in dataset:
            msg = f"Variable {mask_var!r} not found in mask file {mask_file}."
            with ErrorLogger(logger):
                raise KeyError(msg)
        logger.info(f"Reading mask from variable {mask_var} in {mask_file}.")
        data_array = dataset[mask_var].load()

    values = (
        data_array.transpose("time", ...).data[0, ...]
        if "time" in data_array.dims
        else data_array.data
    )
    return np.isnan(values)


def _missing_value(data_array):
    """Return the missing-value marker configured on a data array.

    Parameters
    ----------
    data_array : xarray.DataArray
        Data array whose encoding and attributes are inspected.

    Returns
    -------
    float
        Missing-value marker, falling back to NaN when no marker is present.
    """
    return data_array.encoding.get(
        "missing_value",
        data_array.attrs.get(
            "missing_value",
            data_array.encoding.get(
                "_FillValue", data_array.attrs.get("_FillValue", np.nan)
            ),
        ),
    )


def _prepare_coordinate(data_array):
    """Remove fill-value metadata from a coordinate data array.

    Parameters
    ----------
    data_array : xarray.DataArray
        Coordinate data array modified in place.
    """
    data_array.attrs.pop("_FillValue", None)
    data_array.attrs.pop("missing_value", None)


def _output_encoding(dataset):
    """Build NetCDF encoding for coordinate variables in filled output.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset whose coordinate encodings are generated.

    Returns
    -------
    dict[str, dict]
        Encoding mapping passed to ``Dataset.to_netcdf``.
    """
    return {
        name: dict(_FillValue=None, **({"dtype": "i4"} if name == "time" else {}))
        for name in dataset.coords
    }


def fill_one_file(input_file, input_dir, fill_value, mask, output_dir):
    """Fill all data variables in one NetCDF file and write the result.

    Parameters
    ----------
    input_file : pathlib.Path
        NetCDF file to read.
    input_dir : pathlib.Path
        Base input directory used for relative log messages.
    fill_value : float
        Fill value written to output metadata and masked cells.
    mask : numpy.ndarray or None
        Optional fixed output mask shared by all data variables.
    output_dir : pathlib.Path
        Directory where the filled file is written.

    Returns
    -------
    pathlib.Path
        Path to the written output file.
    """
    logger.info(f"Filling {input_file.relative_to(input_dir)}.")
    with xr.open_dataset(input_file, engine="netcdf4", mask_and_scale=False) as ds:
        dataset = ds.load()

    for var_name in list(dataset.data_vars) + list(dataset.coords):
        data_array = dataset[var_name]
        if var_name in dataset.coords:
            _prepare_coordinate(data_array)
            continue

        missing_value = float(_missing_value(data_array))
        data_array.attrs["_FillValue"] = fill_value
        data_array.attrs["missing_value"] = fill_value
        try:
            fill_dataarray_with_nearest(
                data_array,
                missing_value=missing_value,
                mask=mask,
                fill_value=fill_value,
                source_file=input_file,
            )
        except Exception as exc:
            msg = (
                f"Failed to nearest-neighbour fill variable {var_name!r} "
                f"in {input_file}. dims={data_array.dims} "
                f"shape={data_array.shape} missing_value={missing_value}"
            )
            with ErrorLogger(logger):
                raise RuntimeError(msg) from exc

    output_file = output_dir / input_file.name
    with tempfile.TemporaryDirectory(dir=output_dir) as tmp_dir:
        tmp_file = Path(tmp_dir) / "tmp.nc"
        dataset.to_netcdf(tmp_file, encoding=_output_encoding(dataset))
        shutil.move(tmp_file, output_file)
    return output_file


@log_arguments()
def fill_nearest(
    input_dir,
    fname="precipitation_*.nc",
    output_dir="output_dir",
    mask_file=None,
    mask_var=None,
    fill_value=-9999.0,
    n_cpus=1,
):
    """Fill missing values in matching NetCDF files with nearest neighbours.

    Parameters
    ----------
    input_dir : str or pathlib.Path
        Directory searched for input files.
    fname : str, default "precipitation_*.nc"
        Glob pattern selecting input NetCDF files.
    output_dir : str or pathlib.Path, default "output_dir"
        Directory where filled files are written.
    mask_file : str or pathlib.Path, optional
        Optional NetCDF file containing the fixed output mask.
    mask_var : str, optional
        Variable in ``mask_file`` whose NaNs define masked output cells.
    fill_value : float, default -9999.0
        Fill value written to missing metadata and masked cells.
    n_cpus : int, default 1
        Number of worker processes used for file-level parallelism.

    Returns
    -------
    list[pathlib.Path]
        Output files written by the fill operation.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob(fname))
    if not input_files:
        msg = f"No files match {input_dir / fname}."
        with ErrorLogger(logger):
            raise FileNotFoundError(msg)

    mask = read_mask(mask_file, mask_var)
    logger.info(f"Found {len(input_files)} input files matching {input_dir / fname}.")

    output_files = []
    if n_cpus == 1:
        for input_file in input_files:
            output_files.append(
                fill_one_file(input_file, input_dir, fill_value, mask, output_dir)
            )
    else:
        output_files = Parallel(n_jobs=n_cpus, backend="loky")(
            delayed(fill_one_file)(input_file, input_dir, fill_value, mask, output_dir)
            for input_file in input_files
        )
    return output_files


fill = fill_nearest
