"""Upscale a DEM and a flow direction grid to a coarser grid for mHM.

Every coarse cell covers a block of ``factor`` x ``factor`` input cells, and the
coarse grid keeps the lower-left corner of the input grid.

- The DEM is averaged over the valid cells of each block.
- The slope is calculated on the input DEM and averaged the same way.
- The aspect is calculated on the upscaled DEM.
- The flow direction follows the outlet rule of mHM: the outlet of a block is
  the cell with the largest flow accumulation among the cells that leave the
  block and the pits, and the coarse cell drains into the coarse cell that
  receives the water of its outlet.

The flow accumulation comes from a file, or is computed from the flow
direction, which in turn is derived from the DEM when no flow direction is
given. Routing flow needs the whole input grid in memory. With a flow direction
and a flow accumulation file every step only needs a block and its neighbours,
so the inputs can be processed in dask chunks within a fixed memory budget,
also for global grids.

Authors
-------
- Simon Lüdke
"""

import logging
from pathlib import Path

import numpy as np
import xarray as xr
from affine import Affine

from mhm_tools.common.constants import (
    FDIR_CODE_BY_OFFSET,
    FDIR_FILLVALUE,
    FDIR_SINKVALUE,
    NO_DATA,
)
from mhm_tools.common.crs_handler import resolve_crs, set_spatial_dims
from mhm_tools.common.esri_grid import check_resolutions, rescale_grid
from mhm_tools.common.file_handler import create_header, get_nodata, get_raster_data
from mhm_tools.common.logger import ErrorLogger, log_arguments
from mhm_tools.common.parallel import create_dask_cluster, resolve_ncpus
from mhm_tools.common.topography import calculate_aspect, calculate_slope
from mhm_tools.common.xarray_utils import get_masked_values
from mhm_tools.pre.dem_derivatives import (
    DERIVATIVE_ATTRS,
    _masked,
    _validate_extension,
    _write_netcdf,
    _write_per_layer,
    create_fdir_codes,
    create_variable_attrs,
    get_elevation_values,
)
from mhm_tools.pre.latlon import _create_grid

logger = logging.getLogger(__name__)

OUTSIDE_GRID = -1
"""Flow direction code of the ring beyond the grid edge around a block of codes."""
DOWNSTREAM_MISSING = 0
"""Downstream state of a cell that drains into a missing cell."""
DOWNSTREAM_VALID = 1
"""Downstream state of a cell that drains into a valid cell or is a pit."""
DOWNSTREAM_OUTSIDE = 2
"""Downstream state of a cell that drains off the grid edge."""
BYTES_PER_FINE_CELL_IN_MEMORY = 100
"""Rough peak memory per input cell when the whole input grid is held in memory."""
BYTES_PER_FINE_CELL_IN_CHUNK = 100
"""Rough working memory per input cell of one dask chunk."""
BYTES_PER_COARSE_CELL = 50
"""Rough memory per coarse cell of the results and their coordinates."""
GIB = 1024**3
"""Bytes per GiB."""


@log_arguments()
def upscale_topography(  # noqa: PLR0913
    output_path,
    dem_file=None,
    fdir_file=None,
    facc_file=None,
    upscale_factor=None,
    target_resolution=None,
    depression_fill_dem=False,
    write_facc=False,
    facc_unit="km2",
    keep_edge_outflow=False,
    missing_aspect=90.0,
    ftype="d8",
    output_extension="nc",
    one_file_per_variable=False,
    crs=None,
    dem_var=None,
    fdir_var=None,
    facc_var=None,
    compression=None,
    available_mem_gib=None,
    ncpus=1,
):
    """Upscale a DEM and/or a flow direction grid and write the coarse topography.

    Args:
        output_path (str or Path): Output file for ``nc``, or output directory
            for one file per variable.
        dem_file (str or Path): DEM (.nc, .asc, .tif), or None.
        fdir_file (str or Path): Flow direction grid, or None to derive it from
            the DEM.
        facc_file (str or Path): Flow accumulation that belongs to
            ``fdir_file``, or None to compute it from the flow direction.
        upscale_factor (int): Input cells per coarse cell along each axis.
        target_resolution (float): Coarse cell size, given instead of
            ``upscale_factor``.
        depression_fill_dem (bool): Fill the depressions of the DEM and derive
            dem, slope and aspect from the filled DEM.
        write_facc (bool): Also write the flow accumulation of the outlet cells.
        facc_unit (str): Unit of a computed flow accumulation, ``km2`` or
            ``cell`` for the number of upstream input cells.
        keep_edge_outflow (bool): Keep the direction of outlets that drain off
            the grid edge instead of making them sinks, for tiles that are
            mosaicked later.
        missing_aspect (float): Aspect of flat coarse cells; 90 is neutral in
            the PET aspect correction of mHM, NaN or ``NO_DATA`` leaves them
            missing.
        ftype (str): Flow direction type of ``fdir_file`` and of the output.
        output_extension (str): ``nc``, ``asc`` or ``tif``.
        one_file_per_variable (bool): With ``nc``, write one file per variable.
        crs (str): CRS to use when the inputs have none.
        dem_var (str): Variable of the DEM file, or None.
        fdir_var (str): Variable of the flow direction file, or None.
        facc_var (str): Variable of the flow accumulation file, or None.
        compression (NetcdfCompression): NetCDF compression, or None.
        available_mem_gib (float): Memory budget in GiB, which turns on chunked
            processing; None holds the inputs in memory.
        ncpus (int): Cores for chunked processing, 0 or less for all.

    Returns
    -------
        tuple: The written paths.
    """
    if dem_file is None and fdir_file is None:
        msg = "Give a DEM, a flow direction grid or both."
        with ErrorLogger(logger):
            raise ValueError(msg)
    if facc_file is not None and fdir_file is None:
        msg = "A flow accumulation file needs the flow direction grid it belongs to."
        with ErrorLogger(logger):
            raise ValueError(msg)
    if ftype not in FDIR_CODE_BY_OFFSET:
        msg = (
            f"Unsupported ftype {ftype!r}. "
            f"Choose one of: {', '.join(FDIR_CODE_BY_OFFSET)}."
        )
        with ErrorLogger(logger):
            raise ValueError(msg)
    extension = _validate_extension(output_extension)
    variable_attrs = create_variable_attrs(facc_unit)
    if depression_fill_dem:
        variable_attrs["dem"] = DERIVATIVE_ATTRS["dem_filled"]

    input_files = {
        "dem": (dem_file, dem_var),
        "fdir": (fdir_file, fdir_var),
        "facc": (facc_file, facc_var),
    }
    opened = []
    inputs = {}
    try:
        for name, (path, var_name) in input_files.items():
            if path is None:
                continue
            # flow direction codes assume that the first row is the northernmost
            data = get_raster_data(
                path, var_name=var_name, crs=crs, force_decending_y=True
            )
            opened.append(data)
            if data.dims != (data.rio.y_dim, data.rio.x_dim):
                data = set_spatial_dims(data.transpose(data.rio.y_dim, data.rio.x_dim))
            inputs[name] = data
        reference_name, reference = next(iter(inputs.items()))
        resolved_crs = resolve_crs(reference, crs, required=True)
        reference_transform = reference.rio.transform(recalc=True)
        for name, data in inputs.items():
            if data.shape != reference.shape or not data.rio.transform(
                recalc=True
            ).almost_equals(reference_transform):
                msg = (
                    f"The {name} grid differs from the {reference_name} grid; "
                    "all inputs must share one grid."
                )
                with ErrorLogger(logger):
                    raise ValueError(msg)
        coarse_header, factor = create_coarse_header(
            reference, upscale_factor, target_resolution
        )

        holds_input_in_memory = (
            available_mem_gib is None or "facc" not in inputs or depression_fill_dem
        )
        estimate_gib = calculate_memory_estimate_gib(
            reference.size, factor, holds_input_in_memory
        )
        logger.info(
            f"Upscaling {reference.shape[0]} x {reference.shape[1]} cells by "
            f"{factor} to {coarse_header['nrows']} x {coarse_header['ncols']} cells "
            f"of {coarse_header['cellsize']:g} needs about {estimate_gib:.1f} GiB"
            f"{'' if holds_input_in_memory else ' besides the chunks'}."
        )
        chunk_side = None
        if available_mem_gib is not None:
            if estimate_gib > available_mem_gib:
                hints = []
                if "fdir" not in inputs:
                    hints.append(
                        "give --fdir-file and --facc-file, so no flow is routed"
                    )
                elif "facc" not in inputs:
                    hints.append(
                        "give --facc-file, so no flow accumulation is computed"
                    )
                if depression_fill_dem:
                    hints.append("leave out --depression-fill-dem")
                hints.append("upscale tiles, with --keep-edge-outflow to mosaic them")
                msg = (
                    f"The upscaling needs about {estimate_gib:.1f} GiB, more than the "
                    f"{available_mem_gib:g} GiB available. To fit, {'; or '.join(hints)}."
                )
                with ErrorLogger(logger):
                    raise ValueError(msg)
            cells_per_chunk = (
                available_mem_gib
                * GIB
                / (resolve_ncpus(ncpus) * BYTES_PER_FINE_CELL_IN_CHUNK)
            )
            # whole blocks per chunk, so no block is split between two chunks
            chunk_side = max(factor, int(np.sqrt(cells_per_chunk)) // factor * factor)

        upscaled = calculate_upscaled_topography(
            inputs,
            factor,
            resolved_crs,
            depression_fill_dem=depression_fill_dem,
            write_facc=write_facc,
            facc_unit=facc_unit,
            keep_edge_outflow=keep_edge_outflow,
            missing_aspect=missing_aspect,
            ftype=ftype,
            chunk_side=chunk_side,
            ncpus=ncpus,
        )
        if "facc" in inputs:
            units = inputs["facc"].attrs.get("units")
            variable_attrs["facc"] = (
                "f8",
                {
                    "long_name": "flow accumulation",
                    **({"units": units} if units else {}),
                },
            )
    finally:
        for data in opened:
            data.close()

    derivatives = {}
    for name in ("dem", "slope", "aspect", "facc"):
        if name in upscaled:
            missing = ~np.isfinite(upscaled[name]) | (upscaled[name] == NO_DATA)
            values = np.where(missing, NO_DATA, upscaled[name])
            derivatives[name] = _masked(values, missing, variable_attrs[name][0])
    derivatives["fdir"] = _masked(
        upscaled["fdir"], upscaled["fdir"] == FDIR_FILLVALUE[ftype], "i4"
    )
    projection = None if resolved_crs.is_geographic else resolved_crs.to_wkt()
    x, y, lon, lat = _create_grid(coarse_header, projection, dtype="f8")
    grid = {
        "x": x,
        "y": y,
        "lon": lon,
        "lat": lat,
        "projected": not resolved_crs.is_geographic,
    }
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
            default_name="topography_upscaled.nc",
        )
    return _write_per_layer(
        derivatives, grid, resolved_crs, output_path, extension, variable_attrs
    )


def create_coarse_header(fine_data_array, upscale_factor=None, target_resolution=None):
    """Create the header of a coarse grid that keeps the lower-left corner.

    Args:
        fine_data_array (xarray.DataArray): Input grid with square cells.
        upscale_factor (int): Input cells per coarse cell along each axis.
        target_resolution (float): Coarse cell size, an integer multiple of the
            input cell size, given instead of ``upscale_factor``.

    Returns
    -------
        tuple: The coarse ESRI grid header (dict) and the upscaling factor (int).
    """
    if (upscale_factor is None) == (target_resolution is None):
        msg = "Give exactly one of the upscale factor and the target resolution."
        with ErrorLogger(logger):
            raise ValueError(msg)
    x_size, y_size = (abs(size) for size in fine_data_array.rio.resolution())
    if not np.isclose(x_size, y_size, rtol=1e-6):
        msg = f"The input cells are not square ({x_size} x {y_size})."
        with ErrorLogger(logger):
            raise ValueError(msg)
    fine_header = create_header(fine_data_array)
    if target_resolution is None:
        if int(upscale_factor) < 1:
            msg = f"The upscale factor must be at least 1, got {upscale_factor}."
            with ErrorLogger(logger):
                raise ValueError(msg)
        factor = int(upscale_factor)
        target_resolution = factor * fine_header["cellsize"]
    else:
        factor = int(
            check_resolutions(
                fine_header["cellsize"],
                target_resolution,
                first_finer=True,
                name_1="input grid",
                name_2="target grid",
            )
        )
    nrows, ncols = fine_header["nrows"], fine_header["ncols"]
    if nrows % factor or ncols % factor:
        msg = (
            f"The input grid of {nrows} rows and {ncols} columns is no integer "
            f"multiple of the factor {factor}: {nrows % factor} rows and "
            f"{ncols % factor} columns are left over. Crop the input to a multiple "
            f"of {factor} cells."
        )
        with ErrorLogger(logger):
            raise ValueError(msg)
    coarse_header = rescale_grid(
        fine_header, target_resolution, in_name="input grid", out_name="target grid"
    )
    return coarse_header, factor


def calculate_memory_estimate_gib(fine_cells, factor, holds_input_in_memory):
    """Estimate the peak memory of an upscaling run.

    Args:
        fine_cells (int): Number of input cells.
        factor (int): Input cells per coarse cell along each axis.
        holds_input_in_memory (bool): Whether the whole input grid is held in
            memory, as flow routing and runs without chunks do.

    Returns
    -------
        float: Rough peak memory in GiB, without the chunks of a chunked run.
    """
    fine_bytes = (
        fine_cells * BYTES_PER_FINE_CELL_IN_MEMORY if holds_input_in_memory else 0
    )
    return (fine_bytes + fine_cells / factor**2 * BYTES_PER_COARSE_CELL) / GIB


def calculate_upscaled_topography(
    inputs,
    factor,
    crs,
    depression_fill_dem=False,
    write_facc=False,
    facc_unit="km2",
    keep_edge_outflow=False,
    missing_aspect=90.0,
    ftype="d8",
    chunk_side=None,
    ncpus=1,
):
    """Calculate the coarse topography from the input grids.

    Args:
        inputs (dict): North-up input DataArrays on one grid by name, ``dem``,
            ``fdir`` and ``facc``; ``dem`` or ``fdir`` must be given.
        factor (int): Input cells per coarse cell along each axis.
        crs (rasterio.crs.CRS): CRS of the grid.
        depression_fill_dem (bool): Derive dem, slope and aspect from the
            depression-filled DEM.
        write_facc (bool): Also return the flow accumulation of the outlets.
        facc_unit (str): Unit of a computed flow accumulation.
        keep_edge_outflow (bool): Keep the direction of outlets that drain off
            the grid edge instead of making them sinks.
        missing_aspect (float): Aspect of flat coarse cells, NaN or ``NO_DATA``
            to leave them missing.
        ftype (str): Flow direction type of the input and output codes.
        chunk_side (int): Side length of the dask chunks in input cells, a
            multiple of ``factor``, or None to hold the inputs in memory.
        ncpus (int): Cores for the chunked computation.

    Returns
    -------
        dict: Coarse numpy arrays by name: ``fdir`` with the fill code in
        missing cells, and ``dem``, ``slope``, ``aspect`` and ``facc`` where
        available, NaN or ``NO_DATA`` in missing cells.
    """
    import pyflwdir

    dem, fdir, facc = (inputs.get(name) for name in ("dem", "fdir", "facc"))
    transform = next(iter(inputs.values())).rio.transform(recalc=True)
    latlon = bool(crs.is_geographic)
    coarse = {}
    network = None
    # a DEM used as given needs only its neighbours, so it can be chunked
    dem_chunked = (
        chunk_side is not None and fdir is not None and not depression_fill_dem
    )
    if dem is not None:
        if dem_chunked:
            surface = get_masked_values(dem, chunk_side)
            row_chunks, col_chunks = surface.data.chunks
            slope = surface.fillna(NO_DATA).data.map_overlap(
                calculate_chunk_slope,
                depth=1,
                boundary=NO_DATA,
                meta=np.array((), dtype="float64"),
                transform=transform,
                latlon=latlon,
                row_chunks=row_chunks,
                col_chunks=col_chunks,
            )
            slope = xr.DataArray(slope, dims=("y", "x"))
        else:
            elevation, invalid = get_elevation_values(dem)
            if fdir is None or depression_fill_dem:
                filled, d8 = pyflwdir.dem.fill_depressions(elevation, nodata=NO_DATA)
                if fdir is None:
                    # the flow directions of the filled DEM, without filling twice
                    network = pyflwdir.from_array(
                        d8,
                        ftype="d8",
                        transform=transform,
                        latlon=latlon,
                        check_ftype=False,
                    )
                if depression_fill_dem:
                    elevation = filled
                del filled, d8
            slope = xr.DataArray(
                calculate_slope(elevation, transform, latlon), dims=("y", "x")
            )
            surface = xr.DataArray(
                np.where(invalid, np.nan, elevation), dims=("y", "x")
            )
            del elevation
        coarse["dem"] = surface.coarsen(y=factor, x=factor, boundary="exact").mean()
        coarse["slope"] = (
            slope.where(slope != NO_DATA)
            .coarsen(y=factor, x=factor, boundary="exact")
            .mean()
        )

    # with a flow accumulation file every step needs only a block and its ring
    fdir_chunked = chunk_side is not None and facc is not None
    if network is not None:
        codes = xr.DataArray(network.to_array(ftype), dims=("y", "x"))
    elif fdir_chunked:
        nodata = get_nodata(fdir)
        codes = fdir.chunk(dict.fromkeys(fdir.dims, chunk_side)).data.map_blocks(
            lambda block: create_fdir_codes(block, ftype, nodata)[0],
            meta=np.array((), dtype=np.uint8),
        )
        codes = xr.DataArray(codes, dims=("y", "x"))
    else:
        codes, _ = create_fdir_codes(fdir.values, ftype, get_nodata(fdir))
        if facc is None:
            network = pyflwdir.from_array(
                codes,
                ftype=ftype,
                transform=transform,
                latlon=latlon,
                check_ftype=False,
            )
        codes = xr.DataArray(codes, dims=("y", "x"))
    output_facc = None
    if facc is not None:
        selection_facc = get_masked_values(facc, chunk_side)
    else:
        # mHM picks the outlets by the number of upstream cells
        selection_facc = get_upstream_area(network, "cell")
        if write_facc and facc_unit != "cell":
            output_facc = get_upstream_area(network, facc_unit)
    del network
    coarse_flwdir, inconsistent_blocks = calculate_upscaled_flwdir(
        codes,
        selection_facc,
        factor,
        ftype=ftype,
        keep_edge_outflow=keep_edge_outflow,
        output_facc=output_facc,
    )
    coarse["fdir"] = coarse_flwdir["fdir"]
    coarse["facc"] = coarse_flwdir["facc"]
    coarse["inconsistent_blocks"] = inconsistent_blocks
    if "dem" in coarse:
        coarse["mask_mismatch"] = (
            coarse["dem"].notnull() != (coarse["fdir"] != FDIR_FILLVALUE[ftype])
        ).sum()

    results = xr.Dataset(coarse)
    if any(variable.chunks for variable in results.data_vars.values()):
        # pyflwdir's slope holds the GIL, so the chunks of a DEM need processes
        with create_dask_cluster(ncpus, use_processes=dem is not None and dem_chunked):
            results = results.compute()

    inconsistent_blocks = int(results["inconsistent_blocks"])
    if inconsistent_blocks:
        logger.warning(
            f"{inconsistent_blocks} coarse cells have their largest flow "
            "accumulation on a cell that drains within the coarse cell, so the flow "
            "accumulation does not follow the flow directions there. Their outlet "
            "is the cell with the largest flow accumulation that leaves the cell."
        )
    if "mask_mismatch" in results and int(results["mask_mismatch"]):
        logger.warning(
            f"{int(results['mask_mismatch'])} coarse cells have a DEM value but no "
            "flow direction, or the other way round."
        )
    upscaled = {"fdir": results["fdir"].values}
    if write_facc:
        upscaled["facc"] = results["facc"].values
    if dem is not None:
        upscaled["dem"] = results["dem"].values
        upscaled["slope"] = results["slope"].values
        # the coarse grid starts where the input grid starts, with larger cells
        upscaled["aspect"] = calculate_aspect(
            upscaled["dem"],
            transform * Affine.scale(factor),
            latlon,
            missing_aspect=missing_aspect,
        )
    return upscaled


def calculate_upscaled_flwdir(
    fdir_codes, facc, factor, ftype="d8", keep_edge_outflow=False, output_facc=None
):
    """Upscale flow direction codes with the outlet rule of mHM.

    The outlet of each block is the cell with the largest flow accumulation
    among the cells that leave the block and the pits. On a tie a pit wins, then
    the northernmost and westernmost cell, as ``maxloc`` in mHM does. The coarse
    cell drains into the coarse cell that receives the water of its outlet, or
    is a sink when the outlet is a pit.

    Args:
        fdir_codes (xarray.DataArray): North-up flow direction codes with dims
            ``("y", "x")`` and ``FDIR_FILLVALUE[ftype]`` in missing cells, numpy
            or dask backed.
        facc (xarray.DataArray): Flow accumulation on the same grid that picks
            the outlets, NaN in missing cells.
        factor (int): Input cells per coarse cell along each axis.
        ftype (str): Flow direction type of the codes and of the result.
        keep_edge_outflow (bool): Keep the direction of outlets that drain off
            the grid edge instead of making them sinks.
        output_facc (xarray.DataArray): Flow accumulation to return for the
            outlets instead of ``facc``, or None.

    Returns
    -------
        tuple: An xarray.Dataset with the coarse codes ``fdir`` and the flow
        accumulation ``facc`` of the outlets on dims ``("y", "x")``, and the
        number of coarse cells whose largest flow accumulation lies on a cell
        that drains within the coarse cell, which flags a flow accumulation that
        does not follow the flow directions.
    """
    code_by_offset = np.asarray(FDIR_CODE_BY_OFFSET[ftype])
    row_offset_by_code = np.zeros(256, dtype=np.int8)
    col_offset_by_code = np.zeros(256, dtype=np.int8)
    for (row_index, col_index), code in np.ndenumerate(code_by_offset):
        row_offset_by_code[code] = row_index - 1
        col_offset_by_code[code] = col_index - 1

    codes = fdir_codes.astype(np.int16)
    if isinstance(codes.data, np.ndarray):
        padded = np.pad(codes.data, 1, constant_values=OUTSIDE_GRID)
        downstream_state = calculate_downstream_state(padded, ftype)
    else:
        # dask hands each chunk over with a ring of its neighbours
        downstream_state = codes.data.map_overlap(
            calculate_downstream_state,
            depth=1,
            boundary=OUTSIDE_GRID,
            trim=False,
            meta=np.array((), dtype=np.int8),
            ftype=ftype,
        )
    downstream_state = xr.DataArray(downstream_state, dims=("y", "x"))
    valid = fdir_codes != FDIR_FILLVALUE[ftype]
    row_offset = xr.apply_ufunc(
        row_offset_by_code.take,
        fdir_codes,
        dask="parallelized",
        output_dtypes=[np.int8],
    )
    col_offset = xr.apply_ufunc(
        col_offset_by_code.take,
        fdir_codes,
        dask="parallelized",
        output_dtypes=[np.int8],
    )
    pit = valid & (
        ((row_offset == 0) & (col_offset == 0))
        | (downstream_state == DOWNSTREAM_MISSING)
    )
    if not keep_edge_outflow:
        pit = pit | (valid & (downstream_state == DOWNSTREAM_OUTSIDE))

    fine = xr.Dataset(
        {
            "valid": valid,
            "pit": pit,
            "facc": facc,
            "output_facc": facc if output_facc is None else output_facc,
            "row_offset": row_offset,
            "col_offset": col_offset,
        }
    )
    blocks = (
        fine.coarsen(y=factor, x=factor, boundary="exact")
        .construct(y=("y_block", "row_in_block"), x=("x_block", "col_in_block"))
        .stack(cell=("row_in_block", "col_in_block"), create_index=False)
    )
    cell = xr.DataArray(np.arange(factor * factor), dims="cell")
    downstream_row = blocks["row_offset"] + cell // factor
    downstream_col = blocks["col_offset"] + cell % factor
    leaves_block = (
        (downstream_row < 0)
        | (downstream_row >= factor)
        | (downstream_col < 0)
        | (downstream_col >= factor)
    )
    candidate = blocks["pit"] | (blocks["valid"] & leaves_block)
    candidate_facc = blocks["facc"].where(candidate)
    outlet_facc = candidate_facc.max("cell")
    is_maximum = candidate & (candidate_facc == outlet_facc)
    # a pit wins a tie, then the first cell in north-up row order
    outlet_cell = (2 * is_maximum + (is_maximum & blocks["pit"])).argmax("cell")
    is_outlet = cell == outlet_cell
    outlet_is_pit = (is_outlet & blocks["pit"]).any("cell")
    coarse_row_offset = (downstream_row * is_outlet).sum("cell") // factor
    coarse_col_offset = (downstream_col * is_outlet).sum("cell") // factor
    coarse_codes = xr.apply_ufunc(
        code_by_offset.ravel().take,
        (coarse_row_offset + 1) * 3 + coarse_col_offset + 1,
        dask="parallelized",
        output_dtypes=[code_by_offset.dtype],
    )
    coarse_codes = xr.where(outlet_is_pit, FDIR_SINKVALUE[ftype], coarse_codes)
    coarse_codes = coarse_codes.where(outlet_facc.notnull(), FDIR_FILLVALUE[ftype])
    block_maximum = blocks["facc"].where(blocks["valid"]).max("cell")
    inconsistent_blocks = (block_maximum > outlet_facc).sum()
    coarse = xr.Dataset(
        {
            "fdir": coarse_codes,
            "facc": blocks["output_facc"].where(is_outlet).max("cell"),
        }
    ).rename_dims(y_block="y", x_block="x")
    return coarse, inconsistent_blocks


def calculate_downstream_state(padded_codes, ftype="d8"):
    """Calculate what each cell drains into from codes with a ring of neighbours.

    Args:
        padded_codes (numpy.ndarray): Flow direction codes as int16 with a ring
            of one neighbouring cell on every side, ``OUTSIDE_GRID`` beyond the
            grid edge.
        ftype (str): Flow direction type of the codes.

    Returns
    -------
        numpy.ndarray: For the cells within the ring ``DOWNSTREAM_VALID`` where
        the cell drains into a valid cell or is a pit, ``DOWNSTREAM_MISSING``
        where it drains into a missing cell and ``DOWNSTREAM_OUTSIDE`` where it
        drains off the grid.
    """
    state = np.where(
        padded_codes == OUTSIDE_GRID,
        DOWNSTREAM_OUTSIDE,
        np.where(
            padded_codes == FDIR_FILLVALUE[ftype],
            DOWNSTREAM_MISSING,
            DOWNSTREAM_VALID,
        ),
    ).astype(np.int8)
    codes = padded_codes[1:-1, 1:-1]
    rows, cols = codes.shape
    downstream_state = np.full(codes.shape, DOWNSTREAM_VALID, dtype=np.int8)
    for (row_index, col_index), code in np.ndenumerate(
        np.asarray(FDIR_CODE_BY_OFFSET[ftype])
    ):
        neighbour_state = state[
            row_index : row_index + rows, col_index : col_index + cols
        ]
        downstream_state = np.where(codes == code, neighbour_state, downstream_state)
    return downstream_state


def calculate_chunk_slope(
    padded_elevation, transform, latlon, row_chunks, col_chunks, block_info=None
):
    """Calculate the slope of one dask chunk that carries a ring of neighbours.

    Args:
        padded_elevation (numpy.ndarray): Elevation of the chunk with a ring of
            one cell on every side, ``NO_DATA`` beyond the grid edge.
        transform (affine.Affine): Transform of the whole grid.
        latlon (bool): Whether the grid is geographic.
        row_chunks (tuple): Row counts of the chunks of the whole grid.
        col_chunks (tuple): Column counts of the chunks of the whole grid.
        block_info (dict): Location of the chunk, passed by dask.

    Returns
    -------
        numpy.ndarray: Slope in percent of the chunk and its ring.
    """
    chunk_row, chunk_col = block_info[0]["chunk-location"]
    # the ring adds one row and one column before the first cell of the chunk
    row_start = sum(row_chunks[:chunk_row]) - 1
    col_start = sum(col_chunks[:chunk_col]) - 1
    chunk_transform = transform * Affine.translation(col_start, row_start)
    return calculate_slope(padded_elevation, chunk_transform, latlon)


def get_upstream_area(network, unit="cell"):
    """Get the upstream area of every cell of a pyflwdir flow direction network.

    Args:
        network (pyflwdir.FlwdirRaster): Flow direction network of the grid.
        unit (str): ``cell`` for the number of upstream cells, or an area unit
            such as ``km2``.

    Returns
    -------
        xarray.DataArray: Upstream area with dims ``("y", "x")``, NaN in
        missing cells.
    """
    area = network.upstream_area(unit=unit).astype("float64")
    # pyflwdir marks missing cells with -9999
    area[area == NO_DATA] = np.nan
    return xr.DataArray(area, dims=("y", "x"))
