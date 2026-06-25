"""Create mHM restart files by cropping and running an existing setup.

It can create a continues global restart file with continent specific parameters.
You just have to provide as many masks as parameter sets.

As additional requirement this tool needs the install of a mhm version (tested with mhm v5.13.3) from pip.
"""

import logging
import shlex
from pathlib import Path

logger = logging.getLogger(__name__)


def _parse_fill_nearest_files(value):
    """Parse fill-nearest patterns while ignoring empty whitespace fields."""
    if value is None:
        return None
    if isinstance(value, str):
        return value.split()
    patterns = []
    for item in value:
        if item is None:
            continue
        patterns.extend(str(item).split())
    return patterns or None


def _split_values(value):
    """Flatten repeated and quoted whitespace-separated CLI values."""
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    parsed = []
    for item in values:
        if item is None:
            continue
        parsed.extend(str(item).split())
    return parsed


def add_args(parser):
    """Add CLI arguments for the create_mhm_restart_from_setup subcommand."""
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument(
        "-i",
        "--input-dir",
        "--input-path",
        dest="input_path",
        required=True,
        help="Path to the existing mHM setup.",
    )
    required_args.add_argument(
        "-o",
        "--output-dir",
        "--output-path",
        dest="output_path",
        required=True,
        help="Path where the cropped setup should be saved and run.",
    )
    required_args.add_argument(
        "-m",
        "--mask-file",
        "--mask_file",
        required=True,
        nargs="+",
        help="Path to the mask file used to derive the crop extent.",
    )
    required_args.add_argument(
        "--mhm",
        required=True,
        default="mhm",
        help="Path to the mHM executable.",
    )
    required_args.add_argument(
        "--l1-resolution",
        type=float,
        required=True,
        help="Hydrological resolution used to define tile extents.",
    )

    optional = parser.add_argument_group("optional arguments")
    flags = parser.add_argument_group("flags")
    optional.add_argument(
        "-f",
        "--input-name",
        "--file-name",
        dest="file_name",
        required=False,
        default="*.*",
        help="Input file name glob used while cropping the setup.",
    )
    optional.add_argument(
        "--l11-resolution",
        required=False,
        help="Routing resolution used for latlon creation.",
    )
    optional.add_argument(
        "--l1-increment",
        required=False,
        default=20,
        type=int,
        help="Tile width and height in number of L1 cells.",
    )
    optional.add_argument(
        "--crs",
        default=None,
        help=(
            "Coordinates reference system (e.g. 'epsg:3035'). Needed to create "
            "a new latlon file."
        ),
    )
    optional.add_argument(
        "--n-cpus",
        "--n_cpus",
        required=False,
        default=1,
        type=int,
        help="Number of setup tiles prepared in parallel.",
    )
    optional.add_argument(
        "--mhm-ncpus",
        "--mhm-n-cpus",
        "--mhm_ncpus",
        "--mhm_n_cpus",
        dest="mhm_ncpus",
        required=False,
        default=None,
        type=int,
        help=(
            "Number of prepared tile setups to run with mHM in parallel. "
            "Defaults to --n-cpus."
        ),
    )
    optional.add_argument(
        "--crop-ncpus",
        "--crop_ncpus",
        required=False,
        default=1,
        type=int,
        help="Number of cores used by crop_mhm_setup inside each tile.",
    )
    optional.add_argument(
        "--available-mem",
        required=False,
        default="5",
        help="Available memory per cpu in Gb or Mb (default Gb).",
    )
    optional.add_argument(
        "--mask-var",
        required=False,
        default="mask",
        help="Mask variable name.",
    )
    optional.add_argument(
        "--lat-order",
        required=False,
        default="decreasing",
        help="Direction of the latitude coordinate.",
    )
    optional.add_argument(
        "--output-var",
        required=False,
        default=None,
        help="Output variable name for single data var.",
    )
    optional.add_argument(
        "--output-suffix",
        required=False,
        default=None,
        help="Suffix added to output file names leading to file type conversion.",
    )
    optional.add_argument(
        "--mhm-packages",
        default=None,
        help="Packages to load using module load before running mHM.",
    )
    optional.add_argument(
        "--mhm-args",
        default=None,
        help="Additional arguments passed to the mHM executable.",
    )
    optional.add_argument(
        "-p",
        "--parameter-file",
        "--parameter-files",
        "--parameter_file",
        "--parameter_files",
        dest="parameter_files",
        nargs="+",
        default=None,
        help=(
            "mHM parameter namelist file(s). Provide the same number as "
            "--mask-file entries to run one restart workflow per mask/parameter "
            "pair."
        ),
    )
    optional.add_argument(
        "--fill-nearest-file",
        "--fill_nearest_file",
        dest="fill_nearest_files",
        default=None,
        help=(
            "File name or glob pattern below each cropped tile that should be "
            "filled with nearest neighbours before mHM is run. Uses the cropped "
            "tile DEM as mask. Cropped NetCDF files below meteo/ are always "
            "filled without a mask. Can be repeated."
        ),
    )
    optional.add_argument(
        "--l0-mask-file",
        "--l0_mask_file",
        dest="l0_mask_files",
        default=None,
        help=(
            "File name or glob pattern below each cropped tile used as an L0 "
            "mask for the cropped DEM before mHM is run. The mask variable is "
            "detected with get_single_data_var, and all non-missing values are "
            "treated as active. Can be repeated."
        ),
    )
    optional.add_argument(
        "--restart-pattern",
        default="**/mHM_restart*.nc",
        help="Glob pattern used below the cropped setup to find mHM restart files.",
    )
    optional.add_argument(
        "--restart-output-dir",
        default=None,
        help="Optional directory where tiled restart files are moved.",
    )
    optional.add_argument(
        "--merged-restart-file",
        default=None,
        help="Path for the merged mHM restart file.",
    )

    flags.add_argument(
        "--chunking",
        action="store_true",
        help="Set if each dataset should be read as a chunked dask array.",
    )
    flags.add_argument(
        "--create-header",
        required=False,
        default=True,
        action="store_true",
        help="Force creation of header file for all files.",
    )
    flags.add_argument(
        "--no-cropping",
        required=False,
        default=False,
        action="store_true",
        help="Do not crop the file but only write headers where applicable.",
    )
    flags.add_argument(
        "--no-tile-creation",
        dest="skip_tile_creation",
        required=False,
        default=False,
        action="store_true",
        help=(
            "Do not create or prepare setup tiles. Reuse existing tile "
            "directories below --output-dir and jump directly to running mHM."
        ),
    )
    flags.add_argument(
        "--no-mhm-run",
        "--skip-mhm-run",
        dest="skip_mhm_run",
        required=False,
        default=False,
        action="store_true",
        help=(
            "Do not run mHM. Collect existing restart files below each prepared "
            "tile directory and jump directly to merging."
        ),
    )
    flags.add_argument(
        "--no-restart-check",
        dest="no_restart_check",
        required=False,
        default=False,
        action="store_true",
        help="Do not fail if mHM does not create or update a restart file.",
    )
    flags.add_argument(
        "--recreate-restart",
        dest="recreate_restart",
        required=False,
        default=False,
        action="store_true",
        help=(
            "For tiles with missing restart files, regenerate the meteo header, "
            "fill fallback inputs, and rerun mHM."
        ),
    )
    flags.add_argument(
        "--no-merge",
        dest="no_merge",
        required=False,
        default=False,
        action="store_true",
        help="Do not merge the tiled mHM restart files after running mHM.",
    )


def _as_list(value):
    return _split_values(value)


def _mask_output_name(mask_file, index):
    stem = Path(mask_file).stem
    safe_stem = "".join(
        char if char.isalnum() or char in "._-" else "_" for char in stem
    )
    return f"{index:03d}_{safe_stem}"


def _mhm_args_with_parameter_file(mhm_args, parameter_file):
    if parameter_file is None:
        return mhm_args
    parameter_arg = f"-p {shlex.quote(str(parameter_file))}"
    return parameter_arg if mhm_args is None else f"{mhm_args} {parameter_arg}"


def run(args):
    """Crop an existing setup and run mHM to create mHM restart files."""
    from mhm_tools.common.cli_utils import get_available_mem_in_unit, get_coords
    from mhm_tools.common.file_handler import get_xarray_ds_from_file
    from mhm_tools.pre.create_mhm_restart_from_setup import (
        create_mhm_restart_from_setup,
        merge_restart_files,
    )

    mask_files = _as_list(args.mask_file)
    parameter_files = _as_list(args.parameter_files)
    if parameter_files and len(parameter_files) != len(mask_files):
        msg = (
            "The number of --parameter-file entries must match the number of "
            f"--mask-file entries ({len(parameter_files)} != {len(mask_files)})."
        )
        raise ValueError(msg)

    available_mem = get_available_mem_in_unit(args.available_mem)
    output_path = Path(args.output_path)
    restart_output_dir = (
        Path(args.restart_output_dir) if args.restart_output_dir is not None else None
    )
    multi_mask = len(mask_files) > 1
    results = []
    all_restart_files = []
    lon_mins = []
    lon_maxs = []
    lat_mins = []
    lat_maxs = []
    mask_datasets = []

    for index, mask_file in enumerate(mask_files):
        (
            lon_min_target_grid,
            lon_max_target_grid,
            lat_min_target_grid,
            lat_max_target_grid,
            _mask_da,
        ) = get_coords(mask_file=mask_file, mask_var=args.mask_var)
        lon_mins.append(lon_min_target_grid)
        lon_maxs.append(lon_max_target_grid)
        lat_mins.append(lat_min_target_grid)
        lat_maxs.append(lat_max_target_grid)
        mask_ds = get_xarray_ds_from_file(mask_file)
        mask_datasets.append(mask_ds)
        parameter_file = parameter_files[index] if parameter_files else None

        if multi_mask:
            mask_name = _mask_output_name(mask_file, index)
            run_output_path = output_path / mask_name
            run_restart_output_path = (
                restart_output_dir / mask_name
                if restart_output_dir is not None
                else output_path / "restart_files" / mask_name
            )
            run_merge = False
            run_merged_restart_file = None
        else:
            run_output_path = output_path
            run_restart_output_path = restart_output_dir
            run_merge = not args.no_merge
            run_merged_restart_file = args.merged_restart_file

        result = create_mhm_restart_from_setup(
            input_path=args.input_path,
            output_path=run_output_path,
            mask_da=mask_ds,
            lon_min=lon_min_target_grid,
            lon_max=lon_max_target_grid,
            lat_min=lat_min_target_grid,
            lat_max=lat_max_target_grid,
            mhm_executable=args.mhm,
            l1_resolution=args.l1_resolution,
            l1_increment=args.l1_increment,
            l11_resolution=args.l11_resolution,
            crs=args.crs,
            n_jobs=args.n_cpus,
            mhm_n_jobs=args.mhm_ncpus,
            crop_n_jobs=args.crop_ncpus,
            filename=args.file_name,
            available_mem_gib=available_mem,
            force_header_creation=args.create_header,
            chunking=args.chunking,
            output_var=args.output_var,
            no_cropping=args.no_cropping,
            lat_order=args.lat_order,
            output_suffix=args.output_suffix,
            mask_var=args.mask_var,
            mhm_packages=args.mhm_packages,
            mhm_args=_mhm_args_with_parameter_file(args.mhm_args, parameter_file),
            restart_pattern=args.restart_pattern,
            restart_output_path=run_restart_output_path,
            require_restart=not args.no_restart_check,
            merge=run_merge,
            merged_restart_file=run_merged_restart_file,
            fill_nearest_files=_parse_fill_nearest_files(args.fill_nearest_files),
            l0_mask_files=_parse_fill_nearest_files(args.l0_mask_files),
            skip_tile_creation=args.skip_tile_creation or args.skip_mhm_run,
            skip_mhm_run=args.skip_mhm_run,
            recreate_restart=args.recreate_restart,
        )
        results.append(result)
        all_restart_files.extend(result["restart_files"])

    if multi_mask and not args.no_merge:
        final_restart_file = (
            Path(args.merged_restart_file)
            if args.merged_restart_file is not None
            else output_path / "mHM_restart_001.nc"
        )
        merge_restart_files(
            restart_file_paths=all_restart_files,
            lon_min=min(lon_mins),
            lon_max=max(lon_maxs),
            lat_min=min(lat_mins),
            lat_max=max(lat_maxs),
            l1_resolution=args.l1_resolution,
            output_file=final_restart_file,
            mask_ds=mask_datasets,
            mask_var=args.mask_var,
        )

    return results[0] if len(results) == 1 else results
