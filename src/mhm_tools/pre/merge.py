"""Utilities to merge NetCDF files using CDO."""

import logging
import math
import tempfile
from pathlib import Path

try:
    from cdo import Cdo, CDOException

    cdo = Cdo(returnNoneOnError=False)
except Exception:
    cdo = None
from joblib import Parallel, delayed

from mhm_tools.common.logger import ErrorLogger, log_arguments

logger = logging.getLogger(__name__)


def _merge_chunk(files, out_path, options):
    try:
        res = cdo.mergetime(
            input=" ".join(map(str, files)), output=str(out_path), options=options
        )
    except CDOException as e:
        # keep the debugging gold, but raise a picklable error#
        logger.error(f"merge to {out_path} failed")
        msg = (
            f"cdo mergetime failed (returncode={getattr(e, 'returncode', 'NA')})\n"
            f"STDOUT:\n{getattr(e, 'stdout', '')}\n"
            f"STDERR:\n{getattr(e, 'stderr', '')}"
        )
        with ErrorLogger(logger):
            raise RuntimeError(msg) from e
    if res is None:
        logger.error(f"merge to {out_path} failed")
        return ""
    logger.debug(res)
    return str(out_path)


def merge_files_from_folder(
    tmpdir, files, out_file, n_cpus, max_files=30, recursive_depth=0
):
    """Merge files from a folder in chunks, writing intermediates to tmpdir."""
    # Chunk the inputs ~ evenly across workers
    chunk_size = math.ceil(min(len(files) / n_cpus, max_files))
    chunks = [files[i : i + chunk_size] for i in range(0, len(files), chunk_size)]
    logger.info(
        f"Found {len(files)} files in the folder. Merging them in chunks of {chunk_size}"
    )

    Path(tmpdir).mkdir(parents=True, exist_ok=True)
    part_paths = [
        Path(tmpdir) / f"part_{i:04d}_rec{recursive_depth}.nc"
        for i in range(len(chunks))
    ]

    # Merge each chunk in parallel; keep CDO single-threaded per task (-P 1)
    n_jobs = min(n_cpus, len(chunks))
    logger.info(f"Parallelizing it on {n_jobs} jobs.")
    part_parts_merged = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_merge_chunk)(chunk, part_paths[i], options="-P 1 -O")
        for i, chunk in enumerate(chunks)
    )
    part_parts_merged = [
        Path(p) for p in part_parts_merged if p is not None and p != ""
    ]
    # if len(part_parts_merged) != n_jobs:
    #     raise RuntimeError('mergtime failed probably OOM Error')
    # Final merge of parts (now allow CDO to use n_cpus internally)
    if len(part_parts_merged) > max_files:
        logger.info(
            f"Merged files {len(part_parts_merged)}/{max_files} increasing recursive depth to {recursive_depth+1}"
        )
        out_files = merge_files_from_folder(
            tmpdir,
            part_parts_merged,
            out_file=out_file,
            n_cpus=n_cpus,
            max_files=max_files,
            recursive_depth=recursive_depth + 1,
        )

        part_parts_merged = [
            Path(p)
            for p in out_files
            if p is not None and p != "" and Path(p).is_file()
        ]
        logger.debug(f"{recursive_depth} {len(out_files)} {len(part_parts_merged)}")
        if len(out_files) != len(part_parts_merged):
            logger.error(f"out_files {out_files}")
    if recursive_depth > 0:
        logger.info(
            f"Reducing recusive depth to {recursive_depth-1} for {len(part_parts_merged)} files"
        )
        return part_parts_merged
    logger.info(part_parts_merged)
    if len(part_parts_merged) == 1:
        logger.info("Only one file that can be moved.")
        # same filesystem (tmp inside out_dir), so rename is atomic/fast
        part_parts_merged[0].rename(out_file)
    else:
        logger.info(f"Merging {len(part_parts_merged)} files")
        logger.info(" ".join(map(str, part_parts_merged)))
        logger.info(str(out_file))
        cdo.mergetime(
            input=" ".join(map(str, part_parts_merged)),
            output=str(out_file),
            env={"SKIP_SAME_TIME": "1"},
            options="-O",
        )
    return [str(out_file)]


@log_arguments()
def merge_files(input_path, input_file_part, output, n_cpus, preserve_folders=False):
    """
    Merge NetCDF files along time using CDO via its Python API.

    - Assumes files share the same grid and DO NOT overlap in time.
    - Parallelizes by merging chunks in parallel with joblib, then merges parts.

    Args:
        input_path (str): directory containing input files
        input_file_part (str): glob pattern (e.g., "tavg_*.nc")
        output_path (str): directory to write output
        output_file_name (str): name of output NetCDF (e.g., "merged.nc")
        n_cpus (int): number of parallel workers
        preserve_folders (bool): Decides if the top level folder structure should be preserved.

    Returns
    -------
        str: path to the merged file
    """
    n_cpus = max(1, int(n_cpus)) if n_cpus is not None else 1
    in_dir = Path(input_path)
    output = Path(output)
    if not output.suffix:
        msg = "Output must specifiy a filename"
        with ErrorLogger(logger):
            raise ValueError(msg)
    # subdir_files = {'.': }
    files = sorted(in_dir.glob(input_file_part))
    subdir_files = {".": sorted(in_dir.glob(input_file_part))} if files else {}
    for f in in_dir.glob("*"):
        if not f.is_dir():
            continue
        files = sorted(f.rglob(input_file_part))
        if files:
            subdir_files[f.name] = files
    # all_files = [file for rel_path, file_list in subdir_files.items() for file in file_list]
    sum_files = 0
    out_files = []
    for folder, file_list in subdir_files.items():
        # Fast path: single file -> direct copy via CDO (keeps everything simple)
        if not file_list or len(file_list) == 0:
            continue
        out_dir = output.parent / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / output.name
        logger.info(f"Found {len(file_list)} files in {in_dir / folder}")
        if len(file_list) == 1:
            # cdo.copy(input=str(files[0]), output=str(out_file), options="-O")
            if out_file.exists():
                out_file.unlink()
            out_file.symlink_to(file_list[0])
            out_files.append(out_file)
        elif len(file_list) > 1:
            with tempfile.TemporaryDirectory(dir=out_file.parent) as tmpdir:
                out_files.extend(
                    merge_files_from_folder(tmpdir, file_list, out_file, n_cpus)
                )
        sum_files += len(file_list)
    if sum_files == 0:
        msg = f"No files match {in_dir}/{input_file_part}"
        with ErrorLogger(logger):
            raise FileNotFoundError(msg)
    if not preserve_folders:
        with tempfile.TemporaryDirectory(dir=out_file.parent) as tmpdir:
            final_merge = merge_files_from_folder(tmpdir, out_files, output, n_cpus)
        logger.info(f"Merged a total of {sum_files} to: {final_merge}")
    else:
        logger.info(
            f"Merged a total of {sum_files} into these {len(out_files)} files: {out_files}"
        )
