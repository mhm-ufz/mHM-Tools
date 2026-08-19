"""Create per-region WMO masks from global basins, shapefiles, and a GeoJSON.

Delineates (or reuses) global basins, rasterizes WMO regions from a shapefile
set and a GeoJSON, maps every basin to exactly one region while prioritising
basins covered by the original shapefiles and enforcing region contiguity,
excludes Greenland (or another polygon), and writes one mask per region plus
a 4-panel overview plot.

Authors
-------
- Simon Lüdke
"""


def add_args(parser):
    """Add CLI arguments for the create_wmo_region_masks subcommand.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        the main argument parser
    """
    required_args = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")
    flags = parser.add_argument_group("flags")

    required_args.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Output directory for masks, region rasters, and the overview plot.",
    )
    required_args.add_argument(
        "-m",
        "--mask-file",
        "--land-mask",
        dest="mask_file",
        required=True,
        help="Reference NetCDF defining the target grid and valid land cells.",
    )

    optional.add_argument(
        "--mask-var",
        "--land-mask-variable",
        dest="mask_var",
        default=None,
        help="Mask variable (default: auto-detect 'mask', 'land_mask', 'mask_l2').",
    )
    optional.add_argument(
        "-b",
        "--basin-ids",
        dest="basin_ids",
        default=None,
        help="NetCDF with global unique basin ids. Required unless --fdir-file is given.",
    )
    optional.add_argument(
        "--basin-var",
        default="basin",
        help="Basin id variable (default 'basin').",
    )
    optional.add_argument(
        "-i",
        "--fdir-file",
        default=None,
        help=(
            "Flow direction file; delineates global basins via create-catchment "
            "first when --basin-ids is not given."
        ),
    )
    optional.add_argument(
        "--vn",
        "--varname",
        dest="vn",
        default="fdir",
        help="Name of the --fdir-file variable, passed to create-catchment (default 'fdir').",
    )
    optional.add_argument(
        "--var",
        default="fdir",
        help="Input variable type passed to create-catchment: 'fdir' or 'dem' (default 'fdir').",
    )
    optional.add_argument(
        "--ftype",
        default="d8",
        help="ftype of --fdir-file passed to create-catchment: 'd8', 'ldd', or 'nextxy'.",
    )
    optional.add_argument(
        "--lonlatbox",
        default=None,
        help=(
            "'lon_min,lon_max,lat_min,lat_max,resolution_l0', passed to create-catchment "
            "when delineating global basins."
        ),
    )
    optional.add_argument(
        "--l0-resolution",
        type=float,
        default=None,
        help="Resolution of the flow direction grid, passed to create-catchment.",
    )
    optional.add_argument(
        "--l1-resolution",
        type=float,
        default=None,
        help="Target resolution; should match --mask-file, passed to create-catchment.",
    )
    optional.add_argument(
        "--available-mem",
        default=None,
        help="Available memory per cpu in Gb or Mb (default Gb), passed to create-catchment.",
    )
    optional.add_argument(
        "--shape-dir",
        default=None,
        help="Directory with per-basin shapefiles for the shape-derived regions/anchors.",
    )
    optional.add_argument(
        "--shape-region-regex",
        default=r"(\d{7})$",
        help="Regex whose first captured group's leading digit is the region id.",
    )
    optional.add_argument(
        "--region-geojson",
        default=None,
        help="WMO region GeoJSON (https://github.com/OGCMetOceanDWG/wmo-ra).",
    )
    optional.add_argument(
        "--region-property",
        default="WMO_RA",
        help="Integer region property in the GeoJSON (default 'WMO_RA').",
    )
    optional.add_argument(
        "--region-raster",
        default=None,
        help="Pre-computed combined region raster; skips shapefile/GeoJSON rasterization.",
    )
    optional.add_argument(
        "--region-var",
        default="wmo_region",
        help="Region variable name (default 'wmo_region').",
    )
    optional.add_argument(
        "--exclude-polygon",
        default="greenland",
        help="'greenland' (default), a .geojson/.shp path, or 'none'.",
    )
    optional.add_argument(
        "--min-shape-overlap-fraction",
        type=float,
        default=0.5,
        help=(
            "Minimum fraction of a shapefile's overlapping cells that must fall in its "
            "plurality basin for that basin to become a hard-pinned anchor (default 0.5)."
        ),
    )
    optional.add_argument(
        "--max-island-cells-fraction",
        type=float,
        default=0.001,
        help=(
            "Cleanup-pass size threshold, as a fraction of total land cells "
            "(default 0.001)."
        ),
    )
    optional.add_argument(
        "--min-surround-fraction",
        type=float,
        default=0.8,
        help="Cleanup-pass 'mostly surrounded by one region' threshold (default 0.8).",
    )
    optional.add_argument(
        "--dissolve-batch-size",
        type=int,
        default=200,
        help="Shapefiles read per dissolve batch (default 200).",
    )
    optional.add_argument(
        "-f",
        "--output-file-name",
        default="wmo_region_mask",
        help="Stem for the per-region mask files (default 'wmo_region_mask').",
    )
    optional.add_argument(
        "--plot-file",
        default=None,
        help="Overview plot path (default <output-dir>/wmo_regions_overview.png).",
    )

    flags.add_argument(
        "--no-plot",
        dest="no_plot",
        action="store_true",
        default=False,
        help="Do not create the overview plot.",
    )
    flags.add_argument(
        "--no-basin-mapping",
        dest="no_basin_mapping",
        action="store_true",
        default=False,
        help="Write masks straight from the combined region raster (skip basin-to-region mapping).",
    )
    flags.add_argument(
        "--strict-anchors",
        dest="strict_anchors",
        action="store_true",
        default=False,
        help="Raise on shapefile region conflicts instead of warning and auto-resolving.",
    )
    flags.add_argument(
        "--recreate-basin-ids",
        dest="recreate_basin_ids",
        action="store_true",
        default=False,
        help="Re-delineate global basins instead of reusing the cached file.",
    )
    flags.add_argument(
        "--recreate-region-rasters",
        dest="recreate_region_rasters",
        action="store_true",
        default=False,
        help="Rebuild shape/geojson/combined rasters (cascades: also redoes the basin mapping).",
    )
    flags.add_argument(
        "--recreate-basin-mapping",
        dest="recreate_basin_mapping",
        action="store_true",
        default=False,
        help="Redo the basin-to-region mapping instead of reusing basin_region_map.csv.",
    )


def run(args):
    """Create the WMO region mask files.

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments
    """
    from mhm_tools.common.cli_utils import get_available_mem_in_unit
    from mhm_tools.common.resolution_handler import Resolution

    from ..pre import create_wmo_region_masks

    catchment_kwargs = {}
    if args.fdir_file is not None:
        coordinate_slices = None
        l0_resolution = args.l0_resolution
        if args.lonlatbox is not None:
            lon_min, lon_max, lat_min, lat_max, resolution_l0 = map(
                float, args.lonlatbox.split(",")
            )
            coordinate_slices = {
                "lat": slice(lat_max, lat_min),
                "lon": slice(lon_min, lon_max),
            }
            if l0_resolution is None:
                l0_resolution = resolution_l0
        catchment_kwargs = {
            "var_name": args.vn,
            "var": args.var,
            "ftype": args.ftype,
            "coordinate_slices": coordinate_slices,
            "resolutions": Resolution(l0=l0_resolution, l1=args.l1_resolution),
            "available_mem": get_available_mem_in_unit(args.available_mem),
        }

    create_wmo_region_masks(
        output_dir=args.output_dir,
        mask_file=args.mask_file,
        mask_var=args.mask_var,
        basin_id_file=args.basin_ids,
        basin_var=args.basin_var,
        fdir_file=args.fdir_file,
        shape_dir=args.shape_dir,
        shape_region_regex=args.shape_region_regex,
        region_geojson=args.region_geojson,
        region_property=args.region_property,
        region_raster_file=args.region_raster,
        region_var=args.region_var,
        output_file_name=args.output_file_name,
        exclude_polygon=args.exclude_polygon,
        min_shape_overlap_fraction=args.min_shape_overlap_fraction,
        max_island_cells_fraction=args.max_island_cells_fraction,
        min_surround_fraction=args.min_surround_fraction,
        strict_anchors=args.strict_anchors,
        dissolve_batch_size=args.dissolve_batch_size,
        map_basins=not args.no_basin_mapping,
        create_plot=not args.no_plot,
        plot_file=args.plot_file,
        recreate_basin_ids=args.recreate_basin_ids,
        recreate_region_rasters=args.recreate_region_rasters,
        recreate_basin_mapping=args.recreate_basin_mapping,
        **catchment_kwargs,
    )
