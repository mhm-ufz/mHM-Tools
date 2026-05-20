"""Create basin id file and deliniate catchments."""

import csv
import logging

logger = logging.getLogger(__name__)


def add_args(parser):
    """Add cli arguments for the create_catchment subcommand.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        the main argument parser
    """
    required_args = parser.add_argument_group("required arguments")
    flags = parser.add_argument_group("flags")
    required_args.add_argument(
        "-i",
        "--input-file",
        required=True,
        help=("Path to the input file"),
    )
    required_args.add_argument(
        "-o",
        "--output-path",
        required=True,
        help=(
            "Path to the output file. If a single file is written "
            "this is the path to this file, if multiple files are "
            "written, this is the prefix."
        ),
    )
    # optional
    optional_args = parser.add_argument_group("optional arguments")
    optional_args.add_argument(
        "--vn",
        "--varname",
        default="flwdir",
        help=("Name of variable in output file"),
    )
    optional_args.add_argument(
        "-v",
        "--var",
        default="fdir",
        help=("Input variable, use 'fdir' or 'dem'"),
    )
    optional_args.add_argument(
        "--ftp",
        "--ftype",
        default="ldd",
        help=("ftype of input variable, use 'nextxy', 'ldd' or 'd8'"),
    )
    optional_args.add_argument(
        "--gauge-coords",
        default=None,
        help=(
            "Gauge coordinates in the form 'lat,lon' or multiple pairs like "
            "'lat1,lon1;lat2,lon2' (quote the value)."
        ),
    )
    optional_args.add_argument(
        "--gauges-csv",
        default=None,
        help=(
            "Path to CSV with gauge definitions. Required columns: id, lat, lon. "
            "Optional column: area (km2)."
        ),
    )
    optional_args.add_argument(
        "--lonlatbox",
        required=False,
        default=None,
        help=(
            """coordinates in the form of 'lon_min,lon_max,lat_min,lat_max,resolution_l0'"""
        ),
    )
    optional_args.add_argument(
        "--l0-resolution",
        required=False,
        type=float,
        default=None,
        help=("""Resolution of the morphological input grid."""),
    )
    optional_args.add_argument(
        "--l1-resolution",
        required=False,
        type=float,
        default=None,
        help=("""Resolution of the mHM target grid."""),
    )
    optional_args.add_argument(
        "--l11-resolution",
        required=False,
        type=float,
        default=None,
        help=(
            """Resolution of the mRM routing resolution. Only used to extend the grid to cleanly fit this data."""
        ),
    )
    optional_args.add_argument(
        "--l2-resolution",
        required=False,
        type=float,
        default=None,
        help=(
            """Resolution of the mHM meteo input resolution. Only used to extend the grid to cleanly fit this data."""
        ),
    )
    optional_args.add_argument(
        "--meteo-file",
        required=False,
        type=str,
        default=None,
        help=(
            """Path to a meteo file to extract the l2 resolution from. Overwrites the l2_resolution argument."""
        ),
    )
    flags.add_argument(
        "--upscale",
        action="store_true",
        default=False,
        help=("""Upscale to l1_resolution."""),
    )
    optional_args.add_argument(
        "--coords-are-not-latlon",
        action="store_false",
        default=True,
        help=("""Set this flag if the coordinates are in m not degree."""),
    )
    optional_args.add_argument(
        "--mask-file",
        default="mask.nc",
        help=(
            "Path where to save the mask file. Default saving to output_path/mask.nc"
        ),
    )
    optional_args.add_argument(
        "--frame",
        default=0,
        type=int,
        help=(
            "Creates a frame of nonflow cells around the domain to enable non global domains in ulysses mrm which connects the eastern and western boundaries."
        ),
    )
    optional_args.add_argument(
        "--gauge-optimization-method",
        default="basinex",
        help=(
            "Selection of the gauge optimization method. There are two methods implemented: "
            "1. basinex: with increaing error (steps of 0.1) it selects all cells in the allowed radius and chooses the one closest to original gauge location. If none is found the allowed error is inceased up to max_error"
            "2. burek: Based on Burek et. al. 2023 this calculates the metric as (distance_error + 2*facc_error) and chooses the minimum error"
            "3. all: Uses both loggs comparison and then uses the one with over error value usually basinex"
        ),
    )
    optional_args.add_argument(
        "--ref-catchment-area",
        default=None,
        help=(
            "Reference catchment area in km^2 used to identify the outlet cell near the gauge coordinates. "
            "For multiple gauges, pass a comma-separated list."
        ),
    )
    optional_args.add_argument(
        "--max-distance-cells",
        default=5,
        type=int,
        help=("""Maximum distance in cells to search for the outlet cell."""),
    )
    optional_args.add_argument(
        "--max-error",
        default=0.05,
        type=float,
        help=("""Maximum error allowed when searching for the outlet cell."""),
    )
    optional_args.add_argument(
        "--shape-folder",
        default=None,
        help=(
            "Folder with gauge shapefiles used for shape-based outlet matching. "
            "Files are matched by gauge id contained in the filename."
        ),
    )
    optional_args.add_argument(
        "--gauge-id",
        required=False,
        default=None,
        help=(
            "If Gauge id is provided in addition to the other output a id_gauges file is created in the output_folder. "
            "For multiple gauges, pass a comma-separated list."
        ),
    )
    optional_args.add_argument(
        "--gauge-info-csv",
        required=False,
        default="gauges_info.csv",
        help=(
            "Output gauge info csv (default: gauges_info.csv). "
            "Columns: id, lon, lat, lon_old, lat_old, distance, area, old_area, area_error."
        ),
    )
    optional_args.add_argument(
        "--output-vars",
        default="all",
        help=(
            "Comma-separated list of variables to output in the catchment file. "
            "Default is 'all', which outputs all variables. "
            "Variables include: 'flwdir', 'basin', 'uparea_grid', 'upgrid', 'grdare', 'elevtn'"
        ),
    )
    optional_args.add_argument(
        "--available-mem",
        required=False,
        type=str,
        default=None,
        help=("""Available memory per cpu in Gb or Mb (default Gb)"""),
    )
    optional_args.add_argument(
        "--ncpus",
        required=False,
        default=1,
        type=int,
        help=(
            "Number of cores used for parallelisation. Only needed if multiple gauges are created."
        ),
    )


def run(args):  # noqa: PLR0912,PLR0915
    """Create the catchment file.

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments
    """
    from pathlib import Path

    from mhm_tools.common.cli_utils import get_available_mem_in_unit
    from mhm_tools.common.logger import ErrorLogger
    from mhm_tools.common.utils import Resolution

    from ..pre import create_catchment

    gauge_coords = None
    coordinate_slices = None
    gauge_ids = None
    ref_catchment_area = None

    if args.gauges_csv is not None:
        csv_path = Path(args.gauges_csv)
        if not csv_path.is_file():
            msg = f"Gauge CSV file not found: {csv_path}"
            with ErrorLogger(logger):
                raise FileNotFoundError(msg)
        with csv_path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            if reader.fieldnames is None:
                msg = f"Gauge CSV '{csv_path}' has no header."
                with ErrorLogger(logger):
                    raise ValueError(msg)
            field_map = {name.strip().lower(): name for name in reader.fieldnames}
            required = ("id", "lat", "lon")
            missing = [col for col in required if col not in field_map]
            if missing:
                msg = (
                    f"Gauge CSV '{csv_path}' is missing required column(s): {missing}. "
                    "Required columns are: id, lat, lon."
                )
                with ErrorLogger(logger):
                    raise ValueError(msg)
            area_col = field_map.get("area")
            gauge_coords = []
            gauge_ids = []
            areas = []
            for row_no, row in enumerate(reader, start=2):
                try:
                    gid = str(row[field_map["id"]]).strip()
                    if not gid:
                        empty_id_msg = "empty gauge id"
                        raise ValueError(empty_id_msg)
                    lat = float(str(row[field_map["lat"]]).strip())
                    lon = float(str(row[field_map["lon"]]).strip())
                except Exception as exc:
                    msg = f"Invalid id/lat/lon at row {row_no} in '{csv_path}': {exc}"
                    with ErrorLogger(logger):
                        raise ValueError(msg) from exc
                gauge_ids.append(gid)
                gauge_coords.append((lat, lon))
                if area_col is not None:
                    raw_area = str(row[area_col]).strip()
                    areas.append(float(raw_area) if raw_area else None)
            if not gauge_coords:
                msg = f"Gauge CSV '{csv_path}' does not contain any gauge rows."
                with ErrorLogger(logger):
                    raise ValueError(msg)
            if area_col is not None and any(area is not None for area in areas):
                ref_catchment_area = areas
            else:
                ref_catchment_area = None
        if (
            args.gauge_coords is not None
            or args.gauge_id is not None
            or args.ref_catchment_area is not None
        ):
            logger.warning(
                "Using gauges from --gauges_csv and ignoring --gauge_coords, --gauge_id and --ref_catchment_area."
            )
        logger.info(
            f"Loaded {len(gauge_coords)} gauges from CSV '{csv_path}'. "
            f"Area column {'found' if ref_catchment_area is not None else 'not provided'}."
        )

    if args.gauge_coords is not None and args.gauges_csv is None:

        def _parse_gauge_coords(raw_coords):
            if ";" in raw_coords:
                pairs = [p.strip() for p in raw_coords.split(";") if p.strip()]
                coords = []
                for pair in pairs:
                    lat_str, lon_str = map(str.strip, pair.split(","))
                    coords.append((float(lat_str), float(lon_str)))
                return coords
            parts = [p.strip() for p in raw_coords.split(",") if p.strip()]
            if len(parts) == 2:
                lat, lon = map(float, parts)
                return (lat, lon)
            if len(parts) % 2 != 0:
                error_msg = "gauge_coords must be 'lat,lon' or a list of lat,lon pairs."
                with ErrorLogger(logger):
                    raise ValueError(error_msg)
            coords = []
            for lat_str, lon_str in zip(parts[0::2], parts[1::2]):
                coords.append((float(lat_str), float(lon_str)))
            return coords

        if args.lonlatbox is not None:
            logger.warning(
                "You are using --gauge_coords and --lonlatbox at the same time. Make sure this is intendet and the lonlatbox contains the whole catchment area."
            )
        gauge_coords = _parse_gauge_coords(args.gauge_coords)
        if isinstance(gauge_coords, list):
            logger.info(
                f"using gauge coordinates (n={len(gauge_coords)}): {gauge_coords}"
            )
        else:
            logger.info(f"using gauge coordinates {gauge_coords}")
    if args.lonlatbox is not None:
        lonmin, lonmax, latmin, latmax, resl0 = map(float, args.lonlatbox.split(","))
        coordinate_slices = {"lat": slice(latmax, latmin), "lon": slice(lonmin, lonmax)}
        logger.info(
            f"using lonlatbox with extends: lat=({latmax}, {latmin}); lon=({lonmin}, {lonmax})"
        )
    if args.gauge_id is not None and args.gauges_csv is None:
        raw_ids = str(args.gauge_id)
        if "," in raw_ids:
            gauge_ids = [val.strip() for val in raw_ids.split(",") if val.strip()]
        else:
            gauge_ids = raw_ids.strip()
    if args.ref_catchment_area is not None and args.gauges_csv is None:
        raw_areas = str(args.ref_catchment_area)
        if "," in raw_areas:
            ref_catchment_area = [
                float(val.strip()) for val in raw_areas.split(",") if val.strip()
            ]
        else:
            ref_catchment_area = float(raw_areas)
    list_lengths = {
        name: len(val)
        for name, val in (
            ("gauge_coords", gauge_coords),
            ("gauge_ids", gauge_ids),
            ("ref_catchment_area", ref_catchment_area),
        )
        if isinstance(val, list)
    }
    if list_lengths and not isinstance(gauge_coords, list):
        error_msg = "gauge_coords must be a list when gauge_ids or ref_catchment_area is a list."
        with ErrorLogger(logger):
            raise ValueError(error_msg)
    if list_lengths and len(set(list_lengths.values())) > 1:
        error_msg = "gauge_coords, gauge_ids and ref_catchment_area lists must have the same length."
        with ErrorLogger(logger):
            raise ValueError(error_msg)
    if args.upscale and not args.l1_resolution:
        msg = "If upscaling is enabled l1_resolution must be provided."
        with ErrorLogger(logger):
            raise ValueError(msg)
    available_mem = get_available_mem_in_unit(args.available_mem)
    if Path(args.mask_file).name == str(Path(args.mask_file)):
        mask_file = str(Path(args.output_path) / Path(args.mask_file))
    else:
        mask_file = args.mask_file
    coarse_resolutions = Resolution(
        l0=args.l0_resolution if args.l0_resolution is not None else None,
        l1=args.l1_resolution,
        l11=args.l11_resolution,
        l2=args.l2_resolution,
        l2_file=args.meteo_file,
    )
    if args.ncpus > 1:
        logger.info(f"Using {args.ncpus} cpus for catchment creation.")
    create_catchment(
        input_file=args.input_file,
        output_path=args.output_path,
        var_name=args.vn,
        var=args.var,
        ftype=args.ftp,
        gauge_coords=gauge_coords,
        coordinate_slices=coordinate_slices,
        mask_file=mask_file,
        resolutions=coarse_resolutions,
        frame=args.frame,
        upscale=args.upscale,
        latlon=args.coords_are_not_latlon,
        available_mem=available_mem,
        ref_catchment_area=ref_catchment_area,
        max_distance_cells=args.max_distance_cells,
        max_error=args.max_error,
        gauge_ids=gauge_ids,
        ncpus=args.ncpus,
        output_vars=(
            None if str(args.output_vars).strip().lower() == "all" else args.output_vars
        ),
        gauge_opti_method=args.gauge_optimization_method,
        shape_folder=args.shape_folder,
    )
