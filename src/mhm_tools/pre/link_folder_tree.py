"""Create symlinked copies of folder trees.

The module recreates an input directory layout under a target directory and
creates symbolic links for files matching a configurable name pattern. Existing
links can be kept or overwritten.

Authors
-------
- Simon Lüdke
"""

from __future__ import annotations

import logging
from fnmatch import fnmatch
from pathlib import Path

from mhm_tools.common.logger import ErrorLogger

logger = logging.getLogger(__name__)


def _matches_any_pattern(relative_path, patterns):
    """Check whether a file matches any of the patterns.

    A pattern containing ``/`` is matched against the path relative to the
    input directory, any other pattern against the file name only.

    Parameters
    ----------
    relative_path : str
        File path relative to the input directory, with ``/`` separators.
    patterns : list[str]
        fnmatch patterns.

    Returns
    -------
    bool
        True if at least one pattern matches.
    """
    file_name = relative_path.rsplit("/", 1)[-1]
    return any(
        fnmatch(relative_path if "/" in pattern else file_name, pattern)
        for pattern in patterns
    )


def link_folder_tree(
    input_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
    file_name: str = "*.*",
    exclude_patterns: list[str] | None = None,
    include_patterns: list[str] | None = None,
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
    exclude_patterns : list[str], optional
        Patterns of files that are not linked, by default None. A pattern with
        ``/`` is matched against the path relative to ``input_dir``, any other
        pattern against the file name.
    include_patterns : list[str], optional
        Patterns of the only files that are linked, matched like
        ``exclude_patterns``, by default None (all)

    """
    exclude_patterns = exclude_patterns or []
    if not input_dir.exists():
        msg = f"Input directory does not exist: {input_dir}"
        with ErrorLogger(logger):
            raise FileNotFoundError(msg)
    logger.info(f"Linking files from {input_dir} to {output_dir}")
    for file in input_dir.rglob(file_name):
        relative_path = file.relative_to(input_dir).as_posix()
        if include_patterns and not _matches_any_pattern(
            relative_path, include_patterns
        ):
            logger.debug(f"Not included {file}")
            continue
        if _matches_any_pattern(relative_path, exclude_patterns):
            logger.debug(f"Excluding {file}")
            continue
        output_file = output_dir / relative_path
        already_linked = output_file.exists() or output_file.is_symlink()
        if already_linked and not overwrite:
            continue
        if already_linked and overwrite:
            logger.debug(f"Overwriting existing file {output_file}")
            output_file.unlink()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Linking {file} to {output_file}")
        output_file.symlink_to(file.resolve())
    logger.info("Linking completed.")
