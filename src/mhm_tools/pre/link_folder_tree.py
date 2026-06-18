"""Create symlinked copies of folder trees.

The module recreates an input directory layout under a target directory and
creates symbolic links for files matching a configurable name pattern. Existing
links can be kept or overwritten.

Authors
-------
- Simon Lüdke
"""

import logging
from pathlib import Path

from mhm_tools.common.logger import ErrorLogger

logger = logging.getLogger(__name__)


def link_folder_tree(
    input_dir: Path, output_dir: Path, overwrite: bool = False, file_name: str = "*.*"
):
    """Link all files in a folder tree to another folder tree creating symlinks for each file.

    Parameters
    ----------
    input_dir : Path
        The path to input directory.
    output_dir : Path
        The name of the output directory.
    overwrite : bool, optional
        Overwrite existing symlinks, by default False

    """
    if not input_dir.exists():
        msg = f"Input directory does not exist: {input_dir}"
        with ErrorLogger(logger):
            raise FileNotFoundError(msg)
    logger.info(f"Linking files from {input_dir} to {output_dir}")
    for file in input_dir.rglob(file_name):
        relative_path = file.relative_to(input_dir)
        output_file = output_dir / relative_path
        if output_file.exists() and not overwrite:
            continue
        if output_file.exists() and overwrite:
            logger.debug(f"Overwriting existing file {output_file}")
            output_file.unlink()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Linking {file} to {output_file}")
        output_file.symlink_to(file.resolve())
    logger.info("Linking completed.")
