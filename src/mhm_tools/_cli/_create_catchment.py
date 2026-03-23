"""Create basin id file and deliniate catchments."""

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
    required_args.add_argument(
        "-i",
        "--input_file",
        required=True,
        help=("Path to the input file"),
    )
    required_args.add_argument(
        "-o",
        "--output_path",
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
        "-S",
        "--split_file",
        action="store_false",
        default=True,
        help=(
            "Write out multiple files. By default a single file is written "
            "alternative is that the file is split up by variables."
        ),
    )
    optional_args.add_argument(
        "-C",
        "--creator",
        default=None,
        help=(
            "Level-2 (meteorology) information. Either an ascii (header) file, "
            "a dictionary containing the header information "
            "or a cell-size to determine information from level-0. "
            "Level-2 information wont be written to the latlon file."
        ),
    )
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
        "--gauge_coords",
        default=None,
        help=(
            "Gauge coordinates in the form 'lat,lon' or multiple pairs like "
            "'lat1,lon1;lat2,lon2' (quote the value)."
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
        "--l1_resolution",
        required=False,
        type=float,
        default=None,
        help=("""Resolution of the mHM target grid."""),
    )
    optional_args.add_argument(
        "--l11_resolution",
        required=False,
        type=float,
        default=None,
        help=(
            """Resolution of the mRM routing resolution. Only used to extend the grid to cleanly fit this data."""
        ),
    )
    optional_args.add_argument(
        "--l2_resolution",
        required=False,
        type=float,
        default=None,
        help=(
            """Resolution of the mHM meteo input resolution. Only used to extend the grid to cleanly fit this data."""
        ),
    )
    optional_args.add_argument(
        "--meteo_file",
        required=False,
        type=str,
        default=None,
        help=(
            """Path to a meteo file to extract the l2 resolution from. Overwrites the l2_resolution argument."""
        ),
    )
    optional_args.add_argument(
        "--upscale",
        action="store_true",
        default=False,
        help=("""Upscale to l1_resolution."""),
    )
    optional_args.add_argument(
        "--coords_are_not_latlon",
        action="store_false",
        default=True,
        help=("""Set this flag if the coordinates are in m not degree."""),
    )
    optional_args.add_argument(
        "--mask_file",
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
        "--ref_catchment_area",
        default=None,
        help=(
            "Reference catchment area in km^2 used to identify the outlet cell near the gauge coordinates. "
            "For multiple gauges, pass a comma-separated list."
        ),
    )
    optional_args.add_argument(
        "--max_distance_cells",
        default=5,
        type=int,
        help=("""Maximum distance in cells to search for the outlet cell."""),
    )
    optional_args.add_argument(
        "--max_error",
        default=0.05,
        type=float,
        help=("""Maximum error allowed when searching for the outlet cell."""),
    )
    optional_args.add_argument(
        "--gauge_id",
        required=False,
        default=None,
        help=(
            "If Gauge id is provided in addition to the other output a id_gauges file is created in the output_folder. "
            "For multiple gauges, pass a comma-separated list."
        ),
    )
    optional_args.add_argument(
        "--available_mem",
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
    optional_args.add_argument(
        "--vars",
        default="all",
        help=(
            "Comma-separated list of output variables to write. "
            "Default is 'all'. Example: --vars basin,flwdir"
        ),
    )
    optional_args.add_argument(
        "--gauge-optimization-method",
        default="basinex",
        help=(
            "Selection of the gauge optimization method. There are two methods implemented: " \
            "1. basinex: with increaing error (steps of 0.1) it selects all cells in the allowed radius and chooses the one closest to original gauge location. If none is found the allowed error is inceased up to max_error",
            "2. burek: Based on Burek et. al. 2023 this calculates the metric as (distance_error + 2*facc_error) and chooses the minimum error"
        )
    )


def run(args):
    """Create the catchment file.

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments
    """
    from pathlib import Path

    from mhm_tools.common.cli_utils import get_available_mem_in_unit
    from mhm_tools.common.logger import ErrorLogger
    from mhm_tools.pre.catchment import Resolution

    from ..pre import create_catchment

    gauge_coords = None
    coordinate_slices = None
    gauge_ids = None
    ref_catchment_area = None

    if args.gauge_coords is not None:

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
    if args.gauge_id is not None:
        raw_ids = str(args.gauge_id)
        if "," in raw_ids:
            gauge_ids = [int(val.strip()) for val in raw_ids.split(",") if val.strip()]
        else:
            gauge_ids = int(raw_ids)
    if args.ref_catchment_area is not None:
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
        output_vars=None if str(args.vars).strip().lower() == "all" else args.vars,
        gauge_opti_method=args.gauge_optimization_method
    )
