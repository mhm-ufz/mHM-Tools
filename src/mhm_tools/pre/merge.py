import logging
from pathlib import Path
import math
import tempfile
from joblib import Parallel, delayed
from cdo import Cdo

logger = logging.getLogger(__name__)

# _cdo = Cdo()  # requires the `cdo` binary installed
_cdo = Cdo(returnNoneOnError=False)

def _merge_chunk(files, out_path, options):
    res = _cdo.mergetime(input=" ".join(map(str, files)), output=str(out_path), options=options)
    if res is None: 
        logger.error(f'merge to {out_path} failed')

def merge_files_from_folder(files, out_file, n_cpus):
    # Chunk the inputs ~ evenly across workers
    chunk_size = math.ceil(len(files) / n_cpus)
    chunks = [files[i : i + chunk_size] for i in range(0, len(files), chunk_size)]
    with tempfile.TemporaryDirectory(dir=out_file.parent) as tmpdir:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)
        part_paths = [Path(tmpdir) / f"part_{i:04d}.nc" for i in range(len(chunks))]

        # Merge each chunk in parallel; keep CDO single-threaded per task (-P 1)
        n_jobs = min(n_cpus, len(chunks))
        logger.info(f'Parallelizing it on {n_jobs} jobs.')
        Parallel(n_jobs=n_jobs)(
            delayed(_merge_chunk)(chunk, part_paths[i], options="-P 1 -O")
            for i, chunk in enumerate(chunks)
        )
        part_parts_merged = [p for p in part_paths if p.is_file()]
        if len(part_parts_merged) != n_jobs: 
            raise RuntimeError('mergtime failed probably OOM Error')
        logger.info(f'Merging {len(part_parts_merged)} files')
        # Final merge of parts (now allow CDO to use n_cpus internally)
        if len(part_parts_merged) == 1:
            # same filesystem (tmp inside out_dir), so rename is atomic/fast
            part_parts_merged[0].rename(out_file)
        else:
            _cdo.mergetime(
                input=" ".join(map(str, part_parts_merged)),
                output=str(out_file),
                env={"SKIP_SAME_TIME": 1},
                options=f"-P {n_cpus} -O --skip",
            )
    return str(out_file)

def merge_files(input_path, input_file_part, output, n_cpus):
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
    Returns:
        str: path to the merged file
    """
    n_cpus = max(1, int(n_cpus)) if n_cpus is not None else 1
    in_dir = Path(input_path)
    output = Path(output)
    if not output.suffix:
        raise ValueError("Output must specifiy a filename")
    # subdir_files = {'.': }
    files = sorted(in_dir.glob(input_file_part))
    subdir_files= {'.': sorted(in_dir.glob(input_file_part))} if files else {}
    for f in in_dir.glob('*'):
        if not f.is_dir():
            continue
        files = sorted(f.rglob(input_file_part))
        if files: 
            subdir_files[f.name] = files
    # all_files = [file for rel_path, file_list in subdir_files.items() for file in file_list]
    sum_files = 0
    for folder, file_list in subdir_files.items():
        # Fast path: single file -> direct copy via CDO (keeps everything simple)
        if not file_list or len(file_list) == 0:
            continue
        out_dir = output.parent / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / output.name
        logger.info(f'Found {len(file_list)} files in {in_dir / folder}')
        if len(file_list) == 1:
            # _cdo.copy(input=str(files[0]), output=str(out_file), options="-O")
            if out_file.exists():
                out_file.unlink()
            out_file.symlink_to(file_list[0])
        elif len(file_list) > 1: 
            merge_files_from_folder(file_list, out_file, n_cpus)
        sum_files += len(file_list)
    if sum_files == 0:
        raise FileNotFoundError(f"No files match {in_dir}/{input_file_part}")
    
