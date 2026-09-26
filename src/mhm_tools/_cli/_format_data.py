"""
Format categorical soil, geology, LAI, or land-cover data for use by mHM.

Authors
-------
- Sanjeev Bashyal
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import click

_MANIFEST_SUFFIXES = {".csv", ".txt"}

EPILOG = """Manifest input (-i pointing at a .csv or .txt file) formats several rasters in one run. It is a plain comma-separated file with a header row; column names are matched case- and punctuation-insensitively. Relative paths resolve against the folder holding the manifest. Manifests are accepted only for -t lc and -t soil.

-t lc -- historical land-cover periods (needs -l, -m and -c):

\b
  StartDateTime,EndDateTime,FilePath
  2000-01-01T00:00:00,2005-07-01T12:00:00,landcover_2000.tif
  2005-07-01T12:00:00,2010-01-01T00:00:00,landcover_2005.tif

Periods use inclusive start and exclusive end boundaries. They must be ordered without gaps or overlaps: each StartDateTime equals the previous EndDateTime. Year-only values are accepted. Legacy StartYear/EndYear manifests remain supported with inclusive end years. -e asc writes one file per period; -e nc writes a single lc_periods.nc.

-t soil -- physical horizon layers (no lookup table; -l, -m and -c are ignored):

\b
  Horizon,Upper Depth,Lower Depth,Clay Layer,Sand Layer,Silt Layer,Bulk Density Layer,Bulk Density Unit
  1,0,100,clay1.tif,sand1.tif,silt1.tif,bd1.tif,kg/m3
  2,100,300,clay2.tif,sand2.tif,silt2.tif,bd2.tif,kg/m3

Horizons are numbered from 1 without gaps, the first starts at depth 0, and each Lower Depth is the next Upper Depth. Depths are in mm. Bulk Density Unit must be identical in every row, one of: g/cm3, kg/m3, cg/cm3, mg/cm3, g/dm3, kg/dm3. -e asc writes soil_class.asc plus soil_classdefinition.txt; -e nc writes soil_horizon_class.nc plus soil_classdefinition_iFlag_soilDB_1.txt; -e tif is rejected.
"""


def add_args(parser: ArgumentParser) -> None:
    """Add CLI arguments for the ``format-data`` command."""
    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")
    required.add_argument(
        "-t",
        "--type",
        dest="data_type",
        required=True,
        choices=("soil", "geology", "lai", "lc"),
        help="Data type to format.",
    )
    required.add_argument(
        "-i",
        "--input-file",
        dest="input_file",
        required=True,
        help=(
            "Path to one input raster or a CSV/TXT manifest file. The manifest "
            "layouts are described at the end of this help."
        ),
    )
    required.add_argument(
        "-d",
        "--dem-file",
        required=True,
        help="DEM providing the output grid and coordinate reference system.",
    )
    required.add_argument(
        "-o",
        "--output-path",
        required=True,
        help="Directory where the formatted output files are written.",
    )
    optional.add_argument(
        "-l",
        "--lookup-table",
        help="Path to the categorical lookup table.",
    )
    optional.add_argument(
        "-m",
        "--mapping-field",
        help="Lookup-table column containing the input raster values.",
    )
    optional.add_argument(
        "-c",
        "--class-field",
        help="Lookup-table column containing the output class values.",
    )
    optional.add_argument(
        "-e",
        "--extension",
        choices=("nc", "asc", "tif"),
        default="nc",
        help="Output raster extension. Default: nc.",
    )
    optional.add_argument(
        "-s",
        "--input-crs",
        help="CRS to assign when the input raster has no CRS metadata.",
    )
    optional.add_argument(
        "-r",
        "--dem-crs",
        help="CRS to assign when the DEM has no CRS metadata.",
    )
    optional.add_argument(
        "--resampling",
        choices=("auto", "nearest", "bilinear", "cubic", "average", "mode"),
        help=(
            "Spatial resampling method. 'auto' uses majority/nearest for "
            "classes and average/bilinear for continuous soil layers."
        ),
    )
    optional.add_argument(
        "--output-temporal-resolution",
        choices=("daily", "monthly", "annual", "long-term-mean-monthly"),
        help=(
            "Temporal resolution for gridded LAI NetCDF output. "
            "Default: long-term-mean-monthly."
        ),
    )
    optional.add_argument(
        "--no-fill-nodata",
        dest="fill_nodata",
        action="store_false",
        help=(
            "Keep input nodata gaps instead of taking them from their nearest "
            "valid neighbour. Soil gaps are filled per input layer, so a hole "
            "in one of clay, sand, silt, or bulk density no longer drops the "
            "cell from every horizon."
        ),
    )


def _require_lookup_options(args: Namespace) -> None:
    """Require lookup options only for categorical input workflows."""
    missing = [
        option
        for option, value in (
            ("--lookup-table", args.lookup_table),
            ("--mapping-field", args.mapping_field),
            ("--class-field", args.class_field),
        )
        if value is None
    ]
    if missing:
        names = ", ".join(missing)
        msg = f"Missing required option(s): {names}"
        raise click.UsageError(msg)


def run(args: Namespace) -> None:
    """Dispatch formatting to the selected data workflow."""
    input_file = Path(args.input_file)
    if input_file.is_dir():
        msg = "--input-file must be a raster, CSV, or TXT file, not a directory."
        raise click.UsageError(msg)
    is_manifest_input = input_file.suffix.lower() in _MANIFEST_SUFFIXES
    lookup_values = (args.lookup_table, args.mapping_field, args.class_field)
    gridded_lai = (
        args.data_type == "lai"
        and input_file.suffix.lower() == ".nc"
        and not any(value is not None for value in lookup_values)
    )
    if args.output_temporal_resolution is not None and not gridded_lai:
        msg = "--output-temporal-resolution is only valid for gridded LAI NetCDF."
        raise click.UsageError(msg)
    if gridded_lai:
        if args.extension != "nc":
            msg = "Gridded LAI supports only --extension nc."
            raise click.UsageError(msg)
        if args.resampling not in {None, "nearest", "bilinear"}:
            msg = "Gridded LAI supports only nearest or bilinear resampling."
            raise click.UsageError(msg)
        from mhm_tools.pre.format_lai import format_lai_netcdf_data

        kwargs = {
            "input_file": input_file,
            "dem_file": Path(args.dem_file),
            "output_path": Path(args.output_path),
            "output_temporal_resolution": (
                args.output_temporal_resolution or "long-term-mean-monthly"
            ),
            "dem_crs": args.dem_crs,
        }
        if args.resampling is not None:
            kwargs["resampling"] = args.resampling
        format_lai_netcdf_data(**kwargs)
        return

    if is_manifest_input and args.data_type == "soil":
        from mhm_tools.pre.format_soil import format_soil_horizons as formatter
    elif is_manifest_input and args.data_type == "lc":
        _require_lookup_options(args)
        from mhm_tools.pre.format_lc_data import format_lc_periods as formatter
    elif is_manifest_input:
        msg = "Manifest inputs are supported only for --type lc or soil."
        raise click.UsageError(msg)
    elif args.data_type == "soil":
        _require_lookup_options(args)
        from mhm_tools.pre.format_soil import format_soil_data as formatter
    elif args.data_type == "geology":
        _require_lookup_options(args)
        from mhm_tools.pre.format_geology import format_geology_data as formatter
    elif args.data_type == "lai":
        _require_lookup_options(args)
        from mhm_tools.pre.format_lai import format_lai_data as formatter
    else:
        _require_lookup_options(args)
        from mhm_tools.pre.format_lc_data import format_lc_data as formatter

    kwargs = {
        "input_file": input_file,
        "dem_file": Path(args.dem_file),
        "output_path": Path(args.output_path),
        "output_type": args.extension,
        "input_crs": args.input_crs,
        "dem_crs": args.dem_crs,
    }
    if args.data_type != "soil" or not is_manifest_input:
        kwargs.update(
            lookup_table=Path(args.lookup_table),
            mapping_field=args.mapping_field,
            class_field=args.class_field,
        )
    if args.resampling is not None:
        kwargs["resampling"] = args.resampling
    kwargs["fill_nodata"] = args.fill_nodata
    formatter(**kwargs)
