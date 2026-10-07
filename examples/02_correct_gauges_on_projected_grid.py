"""Correct GRDC gauge locations on a projected flow direction grid.

The gauges of a lat/lon scc gauges file are projected onto the CRS of a flow
direction grid and moved by ``create_catchment`` to the outlet cell whose
catchment matches the reference shape of the gauge best. Every gauge gets its
own ``create_catchment`` run, which only reads the grid around its shape. The
corrected projected coordinates are written to a new scc gauges file and a
histogram shows how far the gauges were moved. Every dropped gauge is logged as
a warning.

Example
-------
python examples/02_correct_gauges_on_projected_grid.py \
    --fdir-file fdir.nc \
    --scc-gauges-file scc_gauges.nc \
    --shape-dir shapefiles_merit_burek2023 \
    --output-dir corrected_gauges \
    --ncpus 8
"""

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from pyproj import CRS, Transformer

from mhm_tools.common.file_handler import get_raster_data
from mhm_tools.common.logger import ErrorLogger, log_errors
from mhm_tools.common.plotter import (
    create_axis_label,
    create_comparison_title,
    create_summary_text,
    style_axes,
)
from mhm_tools.common.provenance import apply_output_provenance
from mhm_tools.pre import create_catchment

logger = logging.getLogger(__name__)
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def read_fdir_grid_info(fdir_file, fdir_var, fdir_crs):
    """Read the CRS, extent and cell size of the flow direction grid.

    Args:
        fdir_file (Path): Flow direction file (ascii, NetCDF or GeoTIFF).
        fdir_var (str): Name of the flow direction variable.
        fdir_crs (str): CRS of the grid, only used when the file carries none.

    Returns
    -------
        Tuple of the grid CRS (pyproj.CRS), the grid bounds as
        (left, bottom, right, top) in m and the cell size in m.
    """
    fdir = get_raster_data(fdir_file, var_name=fdir_var, crs=fdir_crs)
    grid_crs = CRS.from_user_input(fdir.rio.crs)
    grid_bounds = fdir.rio.bounds()
    cell_size_m = abs(fdir.rio.resolution()[0])
    fdir.close()
    if grid_crs.axis_info[0].unit_name != "metre":
        msg = f"The fdir CRS {grid_crs.name} is not a projection in metres."
        with ErrorLogger(logger):
            raise ValueError(msg)
    return grid_crs, grid_bounds, cell_size_m


def read_projected_gauges(scc_gauges_file, grid_crs, grid_bounds):
    """Read the gauges of a lat/lon scc gauges file projected onto the grid CRS.

    Gauges outside the grid extent are dropped with a warning.

    Args:
        scc_gauges_file (Path): scc gauges file with station, lon and lat.
        grid_crs (pyproj.CRS): CRS of the flow direction grid.
        grid_bounds (tuple): Grid extent as (left, bottom, right, top).

    Returns
    -------
        DataFrame with the columns id, x and y of the gauges inside the grid.
    """
    with xr.open_dataset(scc_gauges_file) as scc_ds:
        # station ids can be stored as floats, the shapefile names hold integers
        gauge_ids = scc_ds["station"].values.astype(np.int64)
        lons = scc_ds["lon"].values
        lats = scc_ds["lat"].values
    transformer = Transformer.from_crs("EPSG:4326", grid_crs, always_xy=True)
    x_coords, y_coords = transformer.transform(lons, lats)
    left, bottom, right, top = grid_bounds
    is_inside = (
        (left <= x_coords)
        & (x_coords <= right)
        & (bottom <= y_coords)
        & (y_coords <= top)
    )
    for gauge_id in gauge_ids[~is_inside]:
        logger.warning(f"Dropping gauge {gauge_id}: it lies outside the fdir extent.")
    return pd.DataFrame(
        {"id": gauge_ids[is_inside], "x": x_coords[is_inside], "y": y_coords[is_inside]}
    )


def write_projected_reference_shapes(gauge_ids, shape_dir, grid_crs, output_dir):
    """Write the reference shapes of the gauges reprojected into the grid CRS.

    Gauges without a matching shapefile are dropped with a warning.

    Args:
        gauge_ids (Iterable): Gauge ids to write the reference shapes for.
        shape_dir (Path): Folder with one shapefile per gauge, the id in its name.
        grid_crs (pyproj.CRS): CRS of the flow direction grid.
        output_dir (Path): Folder the reprojected shapefiles are written to.

    Returns
    -------
        List of the gauge ids that have a reference shape.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    gauge_ids_with_shape = []
    for gauge_id in gauge_ids:
        # create_catchment matches the shapes by the same file name pattern
        shape_files = sorted(shape_dir.glob(f"*{gauge_id}*.shp"))
        if not shape_files:
            logger.warning(
                f"Dropping gauge {gauge_id}: no shapefile *{gauge_id}*.shp in {shape_dir}."
            )
            continue
        for shape_file in shape_files:
            shape = gpd.read_file(shape_file).to_crs(grid_crs)
            shape.to_file(output_dir / shape_file.name)
        gauge_ids_with_shape.append(gauge_id)
    logger.info(
        f"Wrote the reference shapes of {len(gauge_ids_with_shape)} gauges to {output_dir}"
    )
    return gauge_ids_with_shape


def create_single_gauge_catchment(gauge_id, gauge_coords, gauge_dir, catchment_args):
    """Run create_catchment for one gauge, logging into the folder of the gauge.

    Args:
        gauge_id (int): Id of the gauge.
        gauge_coords (tuple): Projected gauge coordinates as (y, x).
        gauge_dir (Path): Output folder of the gauge.
        catchment_args (dict): Arguments of create_catchment shared by all gauges.

    Returns
    -------
        None if the gauge was delineated, otherwise the reason it was dropped.
    """
    gauge_dir.mkdir(parents=True, exist_ok=True)
    # create_catchment logs into the gauge folder only, which also works in the
    # worker processes that do not inherit the handlers of the main log
    log_handler = logging.FileHandler(gauge_dir / "create_catchment.log", mode="w")
    log_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    package_logger = logging.getLogger("mhm_tools")
    package_logger.addHandler(log_handler)
    package_logger.setLevel(logging.INFO)
    package_logger.propagate = False
    try:
        create_catchment(
            output_path=gauge_dir,
            gauge_coords=gauge_coords,
            gauge_ids=gauge_id,
            mask_file=gauge_dir / "mask.nc",
            id_gauges_out_path=gauge_dir,
            **catchment_args,
        )
    except Exception as exc:
        # any failure only drops this gauge, its traceback stays in the gauge log
        package_logger.exception(f"Delineation of gauge {gauge_id} failed.")
        return f"{type(exc).__name__}: {exc}"
    finally:
        package_logger.removeHandler(log_handler)
        log_handler.close()
        package_logger.propagate = True
    return None


def create_gauge_catchments(gauges, gauges_dir, catchment_args, ncpus):
    """Run create_catchment for every gauge in its own output folder.

    Gauges create_catchment fails for are dropped with a warning.

    Args:
        gauges (pandas.DataFrame): Gauges with the columns id, x and y.
        gauges_dir (Path): Folder holding one output folder per gauge.
        catchment_args (dict): Arguments of create_catchment shared by all gauges.
        ncpus (int): Number of gauges delineated in parallel.

    Returns
    -------
        List of the output folders of the delineated gauges.
    """
    gauge_dirs = [gauges_dir / str(gauge_id) for gauge_id in gauges["id"]]
    drop_reasons = Parallel(n_jobs=ncpus, verbose=10)(
        delayed(create_single_gauge_catchment)(
            int(gauge_id), (float(gauge_y), float(gauge_x)), gauge_dir, catchment_args
        )
        for gauge_id, gauge_x, gauge_y, gauge_dir in zip(
            gauges["id"], gauges["x"], gauges["y"], gauge_dirs
        )
    )
    delineated_gauge_dirs = []
    for gauge_id, gauge_dir, drop_reason in zip(gauges["id"], gauge_dirs, drop_reasons):
        if drop_reason is None:
            delineated_gauge_dirs.append(gauge_dir)
        else:
            logger.warning(
                f"Dropping gauge {gauge_id}: {drop_reason} (see {gauge_dir})"
            )
    if not delineated_gauge_dirs:
        msg = "create_catchment delineated none of the gauges."
        with ErrorLogger(logger):
            raise ValueError(msg)
    logger.info(f"Delineated {len(delineated_gauge_dirs)} of {len(gauge_dirs)} gauges.")
    return delineated_gauge_dirs


def write_merged_gauges_info(gauge_dirs, gauges_info_file):
    """Merge the gauges_info.csv files of the delineated gauges into one file.

    Gauges create_catchment left uncorrected are dropped with a warning.

    Args:
        gauge_dirs (Iterable): Output folders of the delineated gauges.
        gauges_info_file (Path): Path of the merged gauges_info.csv.

    Returns
    -------
        DataFrame of the merged gauges_info.csv.
    """
    gauges_info = pd.concat(
        [pd.read_csv(gauge_dir / "gauges_info.csv") for gauge_dir in gauge_dirs],
        ignore_index=True,
    )
    # a gauge without a matching method kept its original coordinates
    is_corrected = gauges_info["method"].notna()
    for gauge_id in gauges_info.loc[~is_corrected, "id"]:
        logger.warning(
            f"Dropping gauge {gauge_id}: create_catchment left it uncorrected."
        )
    gauges_info = gauges_info[is_corrected]
    gauges_info.to_csv(gauges_info_file, index=False)
    logger.info(
        f"Wrote the information of {len(gauges_info)} gauges to {gauges_info_file}"
    )
    return gauges_info


def write_corrected_scc_gauges_file(
    scc_gauges_file, gauges_info, grid_crs, corrected_scc_file
):
    """Write the scc gauges file again with the corrected projected coordinates.

    Only the gauges in gauges_info are kept. lon and lat are replaced by x and y
    in the grid CRS, every other variable and attribute is copied.

    Args:
        scc_gauges_file (Path): Original lat/lon scc gauges file.
        gauges_info (pandas.DataFrame): gauges_info.csv of create_catchment.
        grid_crs (pyproj.CRS): CRS of the flow direction grid.
        corrected_scc_file (Path): Path of the written scc gauges file.
    """
    with xr.open_dataset(scc_gauges_file) as scc_ds:
        # match the gauges by integer id, as read_projected_gauges does
        station_ids = scc_ds["station"].values.astype(np.int64)
        is_matched = np.isin(station_ids, gauges_info["id"].to_numpy())
        corrected_ds = scc_ds.isel(station=is_matched).load().drop_encoding()
    corrected_ds = corrected_ds.drop_vars(["lon", "lat"])
    # create_catchment keeps the projected x and y in its lon and lat columns
    corrected_coords = gauges_info.set_index("id").loc[station_ids[is_matched]]
    for name, column in (("x", "lon"), ("y", "lat")):
        corrected_ds[name] = (
            "station",
            corrected_coords[column].to_numpy(),
            {
                "standard_name": f"projection_{name}_coordinate",
                "long_name": f"{name} coordinate of the corrected gauge location",
                "units": "m",
                "grid_mapping": "crs",
            },
        )
    corrected_ds["crs"] = xr.DataArray(0, attrs=grid_crs.to_cf())
    apply_output_provenance(corrected_ds).to_netcdf(corrected_scc_file)
    logger.info(
        f"Wrote {corrected_ds.sizes['station']} corrected gauges to {corrected_scc_file}"
    )


@log_errors(raise_exceptions=True)
def plot_gauge_shift_histogram(
    gauges_info, cell_size_km, max_distance_km, input_name, ref_name, output_dir
):
    """Plot a histogram of the distances the gauges were moved by.

    Args:
        gauges_info (pandas.DataFrame): gauges_info.csv of create_catchment.
        cell_size_km (float): Grid cell size in km, used as bin width.
        max_distance_km (float): Largest allowed shift in km.
        input_name (str): Label of the corrected gauges.
        ref_name (str): Label of the original gauges.
        output_dir (Path): Folder the figure is written to.
    """
    distances_km = gauges_info["distance"].replace([np.inf, -np.inf], np.nan)
    shift_km = distances_km.dropna().to_numpy()
    bin_count = int(np.ceil(max_distance_km / cell_size_km))
    bin_edges = np.linspace(0, bin_count * cell_size_km, bin_count + 1)
    fig, ax = plt.subplots(figsize=(10.5, 5))
    ax.hist(
        shift_km,
        bins=bin_edges,
        color="#79A3E6",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.8,
    )
    style_axes(ax, show_grid=True)
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.set_xlabel(create_axis_label("shift distance", "km"))
    ax.set_ylabel(create_axis_label("number of gauges"))
    ax.set_title(
        f"Shift of {shift_km.size} gauge locations "
        f"({create_summary_text(shift_km, units='km')})"
    )
    fig.suptitle(
        create_comparison_title(input_name, ref_name),
        fontweight="normal",
        fontsize="x-large",
    )
    plt.tight_layout()
    file_name = f"gauge_shift_histogram_{input_name}_{ref_name}.png".replace(" ", "_")
    output_file = output_dir / file_name
    fig.savefig(output_file, dpi=400)
    plt.close(fig)
    logger.info(f"Wrote the gauge shift histogram to {output_file}")


def main():
    """Project, correct and rewrite the gauges of an scc gauges file."""
    parser = argparse.ArgumentParser(
        description="Correct the gauges of a lat/lon scc gauges file on a projected "
        "flow direction grid."
    )
    parser.add_argument(
        "--fdir-file",
        required=True,
        type=Path,
        help="Flow direction NetCDF on the projected grid. Each gauge only reads "
        "the window around its reference shape.",
    )
    parser.add_argument(
        "--fdir-var",
        default="fdir",
        help="Name of the flow direction variable (default: fdir).",
    )
    parser.add_argument(
        "--fdir-type",
        default="d8",
        help="Flow direction type: d8, ldd or nextxy (default: d8).",
    )
    parser.add_argument(
        "--fdir-crs",
        default="EPSG:3035",
        help="CRS of the fdir grid, only used when the file carries none "
        "(default: EPSG:3035).",
    )
    parser.add_argument(
        "--scc-gauges-file",
        required=True,
        type=Path,
        help="scc gauges file with the variables station, lon and lat in EPSG:4326.",
    )
    parser.add_argument(
        "--shape-dir",
        required=True,
        type=Path,
        help="Folder with one reference shapefile per gauge, the id in the file name.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Folder all outputs are written to.",
    )
    parser.add_argument(
        "--max-distance-m",
        default=5000.0,
        type=float,
        help="Maximum distance in m a gauge may be moved (default: 5000).",
    )
    parser.add_argument(
        "--max-error",
        default=0.5,
        type=float,
        help="Maximum shape error (1 - IoU) of the selected outlet (default: 0.5).",
    )
    parser.add_argument(
        "--ncpus",
        default=1,
        type=int,
        help="Number of gauges delineated in parallel processes. Each needs the "
        "memory of its crop, which nears the full grid for the largest basins "
        "(default: 1).",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # force replaces the root handler that importing mhm_tools may already add
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.output_dir / "correct_gauges.log", mode="w"),
        ],
        force=True,
    )

    # 1. project the gauges and their reference shapes onto the grid
    grid_crs, grid_bounds, cell_size_m = read_fdir_grid_info(
        args.fdir_file, args.fdir_var, args.fdir_crs
    )
    gauges = read_projected_gauges(args.scc_gauges_file, grid_crs, grid_bounds)
    reference_shape_dir = args.output_dir / "reference_shapes"
    gauge_ids_with_shape = write_projected_reference_shapes(
        gauges["id"], args.shape_dir, grid_crs, reference_shape_dir
    )
    gauges = gauges[gauges["id"].isin(gauge_ids_with_shape)]
    if gauges.empty:
        msg = "No gauge with a reference shape lies inside the fdir extent."
        with ErrorLogger(logger):
            raise ValueError(msg)

    # 2. move every gauge to the outlet matching its reference shape best
    catchment_args = {
        "input_file": args.fdir_file,
        "var_name": args.fdir_var,
        "var": "fdir",
        "ftype": args.fdir_type,
        "frame": 0,
        "latlon": False,
        "max_distance_m": args.max_distance_m,
        "max_error": args.max_error,
        "shape_folder": reference_shape_dir,
        "gauge_info_file": "gauges_info",
    }
    gauge_dirs = create_gauge_catchments(
        gauges, args.output_dir / "gauges", catchment_args, args.ncpus
    )
    gauges_info = write_merged_gauges_info(
        gauge_dirs, args.output_dir / "gauges_info.csv"
    )

    # 3. write the corrected coordinates to a new scc gauges file
    write_corrected_scc_gauges_file(
        args.scc_gauges_file,
        gauges_info,
        grid_crs,
        args.output_dir / f"{args.scc_gauges_file.stem}_epsg{grid_crs.to_epsg()}.nc",
    )

    # 4. show how far the gauges were moved
    plot_gauge_shift_histogram(
        gauges_info,
        cell_size_km=cell_size_m / 1000,
        max_distance_km=args.max_distance_m / 1000,
        input_name=args.fdir_file.stem,
        ref_name=args.scc_gauges_file.stem,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
