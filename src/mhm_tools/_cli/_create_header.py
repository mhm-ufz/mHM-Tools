"""Create an ASCII header.txt from a NetCDF or ASCII input file."""

from pathlib import Path


def add_args(parser):
    """Add CLI arguments for the create_header subcommand."""
    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")
    required.add_argument(
        "-i",
        "--input-file",
        required=True,
        help="Path to input file (.nc or .asc).",
    )
    optional.add_argument(
        "-o",
        "--output-dir",
        required=False,
        default=".",
        help="Directory where header.txt is written. Default: current directory.",
    )


def run(args):
    """Create header.txt from an input dataset."""
    from mhm_tools.common.file_handler import create_header, get_xarray_ds_from_file

    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with get_xarray_ds_from_file(input_file) as ds:
        create_header(ds, output_path=output_dir)
