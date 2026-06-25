"""Create mHM restart files from an existing setup.

The module splits a setup into L1 tiles, optionally filters tiles by a mask,
runs mHM for each tile, collects the produced restart files, and can merge
those tile restarts into a final CF-style restart file.
"""

import argparse
import json
import logging
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from joblib import Parallel, delayed

from mhm_tools.common.file_handler import get_xarray_ds_from_file, write_xarray_to_file
from mhm_tools.common.logger import ErrorLogger, log_arguments
from mhm_tools.common.resolution_handler import Resolution
from mhm_tools.common.xarray_utils import get_coord_key, get_single_data_var
from mhm_tools.pre.crop_mhm_setup import crop_mhm_setup, regrid_mask
from mhm_tools.pre.fill_nearest import fill_nearest

logger = logging.getLogger(__name__)


def _snap_grid_bound(value, resolution=None):
    """Snap grid bounds that only differ from exact grid lines by roundoff."""
    value = float(value)
    if not np.isfinite(value):
        return value
    scale = max(abs(value), 1.0)
    tolerance = np.finfo(float).eps * scale * 4096
    if resolution is not None:
        resolution = abs(float(resolution))
        if np.isfinite(resolution) and resolution > 0:
            snapped = round(value / resolution) * resolution
            if abs(value - snapped) <= tolerance:
                value = float(snapped)
    nearest_integer = round(value)
    if abs(value - nearest_integer) <= tolerance:
        return float(nearest_integer)
    return value


@dataclass
class MHMSetupTile:
    """Definition of a single cropped setup tile.

    ``lonslice`` and ``latslice`` contain cell bounds. The other coordinate
    fields contain center-cell coordinates.
    """

    name: str
    output_path: Path
    lonslice: slice
    latslice: slice
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float


def get_crop_slices(lon_min, lon_max, lat_min, lat_max, lat_order="decreasing"):
    """Create longitude and latitude slices for ``crop_mhm_setup``."""
    if lat_order == "decreasing":
        latslice = slice(lat_max, lat_min)
    elif lat_order == "increasing":
        latslice = slice(lat_min, lat_max)
    else:
        msg = f"Unknown lat_order {lat_order!r}. Use 'increasing' or 'decreasing'."
        with ErrorLogger(logger):
            raise ValueError(msg)
    return slice(lon_min, lon_max), latslice


def create_setup_tiles(
    lon_min_bound,
    lon_max_bound,
    lat_min_bound,
    lat_max_bound,
    l1_resolution,
    l1_increment,
    output_path,
    lat_order="decreasing",
):
    """Create L1 tiles using the same tile-size logic as ``MHMRestartFile``."""
    l1_resolution = float(l1_resolution)
    l1_increment = int(l1_increment)
    tile_size = l1_resolution * l1_increment
    if tile_size <= 0:
        msg = "Tile size must be positive. Check l1_resolution and l1_increment."
        with ErrorLogger(logger):
            raise ValueError(msg)
    logger.info(
        f"Creating tiles from lon {lon_min_bound} to {lon_max_bound} and lat {lat_min_bound} to {lat_max_bound} with increment {l1_increment}"
    )
    output_path = Path(output_path)
    tiles = []
    lon_min_bound = _snap_grid_bound(lon_min_bound, l1_resolution)
    lon_max_bound = _snap_grid_bound(lon_max_bound, l1_resolution)
    lat_min_bound = _snap_grid_bound(lat_min_bound, l1_resolution)
    lat_max_bound = _snap_grid_bound(lat_max_bound, l1_resolution)
    n_lon_tiles = int(np.ceil((lon_max_bound - lon_min_bound) / tile_size))
    n_lat_tiles = int(np.ceil((lat_max_bound - lat_min_bound) / tile_size))
    for lon_index in range(n_lon_tiles):
        tile_lon_min_bound = _snap_grid_bound(
            lon_min_bound + lon_index * tile_size,
            l1_resolution,
        )
        tile_lon_max_bound = min(
            _snap_grid_bound(
                lon_min_bound + (lon_index + 1) * tile_size,
                l1_resolution,
            ),
            lon_max_bound,
        )
        for lat_index in range(n_lat_tiles):
            tile_lat_min_bound = _snap_grid_bound(
                lat_min_bound + lat_index * tile_size,
                l1_resolution,
            )
            tile_lat_max_bound = min(
                _snap_grid_bound(
                    lat_min_bound + (lat_index + 1) * tile_size,
                    l1_resolution,
                ),
                lat_max_bound,
            )
            lonslice, latslice = get_crop_slices(
                tile_lon_min_bound,
                tile_lon_max_bound,
                tile_lat_min_bound,
                tile_lat_max_bound,
                lat_order=lat_order,
            )
            tile_name = f"slice_{lon_index}_{lat_index}"
            tiles.append(
                MHMSetupTile(
                    name=tile_name,
                    output_path=output_path / tile_name,
                    lonslice=lonslice,
                    latslice=latslice,
                    lon_min=float(tile_lon_min_bound) + l1_resolution / 2,
                    lon_max=float(tile_lon_max_bound) - l1_resolution / 2,
                    lat_min=float(tile_lat_min_bound) + l1_resolution / 2,
                    lat_max=float(tile_lat_max_bound) - l1_resolution / 2,
                )
            )
    if not tiles:
        msg = "No setup tiles were created for the requested extent."
        with ErrorLogger(logger):
            raise ValueError(msg)
    logger.info(f"Creating {len(tiles)} tiles")
    return tiles


def _get_mask_data_array(mask_ds, mask_var):
    """Return the mask DataArray selected by name or inferred from one variable."""
    if mask_ds is None:
        return None
    if isinstance(mask_ds, xr.DataArray):
        return mask_ds
    if mask_var in mask_ds.data_vars:
        return mask_ds[mask_var]
    data_var = get_single_data_var(mask_ds)
    if data_var is None:
        msg = f"Mask dataset has multiple data variables; provide one of {list(mask_ds.data_vars)}."
        with ErrorLogger(logger):
            raise ValueError(msg)
    return mask_ds[data_var]


def _coord_bounds(values):
    """Return lower and upper bounds for 1D cell-center coordinates."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        msg = "Mask coordinates must be non-empty 1D arrays."
        with ErrorLogger(logger):
            raise ValueError(msg)
    if values.size == 1:
        half_width = 0.5
        lower = values - half_width
        upper = values + half_width
        return lower, upper

    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = (values[:-1] + values[1:]) / 2
    edges[0] = values[0] - (edges[1] - values[0])
    edges[-1] = values[-1] + (values[-1] - edges[-2])
    lower = np.minimum(edges[:-1], edges[1:])
    upper = np.maximum(edges[:-1], edges[1:])
    return lower, upper


def _tile_has_active_mask_cell(tile, mask_da):
    """Return whether an L1 tile overlaps at least one active mask cell."""
    lon_key = get_coord_key(mask_da, lon=True)
    lat_key = get_coord_key(mask_da, lat=True)
    # mask = mask_da.sel({lon_key: tile.lonslice, lat_key: tile.latslice})
    # active = np.isfinite(mask) & (mask > 0)
    # return bool(active.any())
    lon_min = min(float(tile.lonslice.start), float(tile.lonslice.stop))
    lon_max = max(float(tile.lonslice.start), float(tile.lonslice.stop))
    lat_min = min(float(tile.latslice.start), float(tile.latslice.stop))
    lat_max = max(float(tile.latslice.start), float(tile.latslice.stop))

    active = mask_da.transpose(..., lat_key, lon_key)
    extra_dims = [dim for dim in active.dims if dim not in (lat_key, lon_key)]
    active_values = np.isfinite(active.values) & (active.values > 0)
    if extra_dims:
        active_values = np.any(active_values, axis=tuple(range(len(extra_dims))))

    lon_lower, lon_upper = _coord_bounds(mask_da[lon_key].values)
    lat_lower, lat_upper = _coord_bounds(mask_da[lat_key].values)
    lon_overlaps = (lon_lower < lon_max) & (lon_upper > lon_min)
    lat_overlaps = (lat_lower < lat_max) & (lat_upper > lat_min)
    overlap = np.outer(lat_overlaps, lon_overlaps)
    return bool(np.any(active_values & overlap))


def _tile_mask_section(tile, mask_da):
    """Return the mask cells whose bounds overlap a setup tile."""
    lon_key = get_coord_key(mask_da, lon=True)
    lat_key = get_coord_key(mask_da, lat=True)
    lon_min = min(float(tile.lonslice.start), float(tile.lonslice.stop))
    lon_max = max(float(tile.lonslice.start), float(tile.lonslice.stop))
    lat_min = min(float(tile.latslice.start), float(tile.latslice.stop))
    lat_max = max(float(tile.latslice.start), float(tile.latslice.stop))

    lon_lower, lon_upper = _coord_bounds(mask_da[lon_key].values)
    lat_lower, lat_upper = _coord_bounds(mask_da[lat_key].values)
    lon_overlaps = (lon_lower < lon_max) & (lon_upper > lon_min)
    lat_overlaps = (lat_lower < lat_max) & (lat_upper > lat_min)
    return mask_da.isel({lon_key: lon_overlaps, lat_key: lat_overlaps})


def _write_tile_mask_section(tile, mask_ds, mask_var, fname="mask_tile.nc"):
    """Write the mask section used for tile selection into the tile directory."""
    mask_da = _get_mask_data_array(mask_ds, mask_var)
    if mask_da is None:
        return None

    mask_section = _tile_mask_section(tile, mask_da)
    if mask_section.name is None:
        mask_section = mask_section.rename(mask_var)
    output_file = tile.output_path / fname
    logger.info(f"Writing mask section for tile {tile.name} to {output_file}.")
    write_xarray_to_file(mask_section, output_file, var_name=mask_section.name)
    return output_file


def _meteo_fill_nearest_files(setup_path):
    """Return NetCDF meteo files that should be filled before mHM runs."""
    meteo_paths = sorted(d for d in Path(setup_path).rglob("meteo") if d.is_dir())
    return sorted(
        path
        for meteo_path in meteo_paths
        for path in meteo_path.rglob("*.nc")
        if path.is_file()
    )


def _filter_tiles_by_mask(tiles, mask_ds, mask_var):
    """Keep only tiles that contain active L1 mask cells."""
    mask_da = _get_mask_data_array(mask_ds, mask_var)
    if mask_da is None:
        return tiles
    active_tiles = [tile for tile in tiles if _tile_has_active_mask_cell(tile, mask_da)]
    skipped = len(tiles) - len(active_tiles)
    if skipped:
        logger.info(
            f"Skipping {skipped} setup tiles without active mask cells; "
            f"running {len(active_tiles)} of {len(tiles)} tiles."
        )
    if not active_tiles:
        msg = "Mask does not overlap any setup tile."
        with ErrorLogger(logger):
            raise ValueError(msg)
    return active_tiles


class MHMRunner:
    """Run the mHM executable in the base directory of a setup."""

    def __init__(self, mhm_executable="mhm", mhm_packages=None, mhm_args=None):
        """Initialize the mHM runner."""
        self.mhm_executable = str(mhm_executable)
        self.mhm_packages = mhm_packages
        self.mhm_args = mhm_args

    def _get_module_command(self):
        """Return shell commands needed to prepare the module environment."""
        command = ""
        if self.mhm_packages is not None:
            package_tokens = shlex.split(str(self.mhm_packages))
            module_paths = [token for token in package_tokens if token.startswith("/")]
            modules = [token for token in package_tokens if not token.startswith("/")]
            if module_paths:
                command += "module use " + " ".join(module_paths) + "\n"
            if modules:
                command += "module load " + " ".join(modules) + "\n"
        return command

    def _parse_mhm_args(self):
        """Parse optional command-line arguments for the Python mHM runner."""
        args = {
            "namelist_mhm": "mhm.nml",
            "namelist_mhm_param": "mhm_parameter.nml",
            "namelist_mhm_output": "mhm_outputs.nml",
            "namelist_mrm_output": "mrm_outputs.nml",
            "verbosity": 3,
        }
        if self.mhm_args is None:
            return args
        tokens = shlex.split(str(self.mhm_args))
        idx = 0
        while idx < len(tokens):
            token = tokens[idx]
            if token in ("-n", "--nml"):
                idx += 1
                args["namelist_mhm"] = tokens[idx]
            elif token in ("-p", "--parameter"):
                idx += 1
                args["namelist_mhm_param"] = tokens[idx]
            elif token in ("-o", "--mhm_output"):
                idx += 1
                args["namelist_mhm_output"] = tokens[idx]
            elif token in ("-r", "--mrm_output"):
                idx += 1
                args["namelist_mrm_output"] = tokens[idx]
            elif token in ("-q", "--quiet"):
                args["verbosity"] = max(0, int(args["verbosity"]) - 1)
            else:
                msg = (
                    f"Unsupported mHM Python runner argument {token!r}. "
                    "Supported arguments are -n/--nml, -p/--parameter, "
                    "-o/--mhm_output, -r/--mrm_output, and -q/--quiet."
                )
                with ErrorLogger(logger):
                    raise ValueError(msg)
            idx += 1
        return args

    def get_command(self):
        """Return the shell command used to run mHM via its Python bindings."""
        args = self._parse_mhm_args()
        script = "\n".join(
            [
                "import mhm",
                f"mhm.model.set_verbosity(level={int(args['verbosity'])})",
                "mhm.model.init("
                f"namelist_mhm={json.dumps(args['namelist_mhm'])}, "
                f"namelist_mhm_param={json.dumps(args['namelist_mhm_param'])}, "
                f"namelist_mhm_output={json.dumps(args['namelist_mhm_output'])}, "
                f"namelist_mrm_output={json.dumps(args['namelist_mrm_output'])}, "
                "cwd='.'"
                ")",
                "try:",
                "    mhm.model.run()",
                "finally:",
                "    mhm.model.finalize()",
            ]
        )
        command = self._get_module_command()
        command += "python -c " + shlex.quote(script)
        return command

    def run_mhm(self, setup_path):
        """Run mHM in ``setup_path`` and raise when it exits unsuccessfully."""
        setup_path = Path(setup_path)
        command = self.get_command()
        logger.info(f"Running mHM with: {command}")
        process = subprocess.run(
            command,
            shell=True,
            cwd=setup_path,
            capture_output=True,
            check=False,
        )
        stdout = process.stdout.decode(errors="replace")
        stderr = process.stderr.decode(errors="replace")
        if stdout:
            logger.debug(stdout)
        if stderr:
            logger.debug(stderr)
        if process.returncode != 0:
            msg = (
                f"mHM failed with return code {process.returncode} "
                f"in {setup_path} and command {command}. STDERR: {stderr}"
            )
            with ErrorLogger(logger):
                raise RuntimeError(msg)
        return process


def _find_restart_files(setup_path, restart_pattern, min_mtime=None):
    """Find restart files below a setup path, optionally filtering by mtime."""
    restart_files = sorted(Path(setup_path).glob(restart_pattern))
    if min_mtime is None:
        return restart_files
    return [path for path in restart_files if path.stat().st_mtime >= min_mtime]


def _path_relative_to(path, parent):
    """Return ``path`` relative to ``parent`` or ``None`` when unrelated."""
    try:
        return Path(path).relative_to(parent)
    except ValueError:
        return None


def _move_restart_files(restart_files, setup_path, restart_output_path):
    """Move restart files from tile folders into a separate output tree."""
    setup_path = Path(setup_path)
    restart_output_path = Path(restart_output_path)
    restart_output_path.mkdir(parents=True, exist_ok=True)
    moved_files = []
    for restart_file in restart_files:
        restart_file_path = Path(restart_file)
        if _path_relative_to(restart_file_path, restart_output_path) is not None:
            target = restart_file_path
        else:
            relative_path = restart_file_path.relative_to(setup_path)
            target = restart_output_path / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                target.unlink()
            shutil.move(restart_file_path, target)
            tile_mask_file = _tile_mask_file_for_restart(restart_file_path)
            if tile_mask_file is not None:
                target_tile_mask = target.parent.parent / tile_mask_file.name
                target_tile_mask.parent.mkdir(parents=True, exist_ok=True)
                if target_tile_mask.exists():
                    target_tile_mask.unlink()
                shutil.move(tile_mask_file, target_tile_mask)
        moved_files.append(target)
    return moved_files


def _find_dem_file(setup_path):
    """Find the cropped DEM file in one tile setup."""
    setup_path = Path(setup_path)
    dem_files = sorted(
        path
        for path in setup_path.rglob("*.nc")
        if path.is_file() and "dem" in path.name.lower()
    )
    if not dem_files:
        msg = f"No cropped DEM NetCDF file found below {setup_path}."
        with ErrorLogger(logger):
            raise FileNotFoundError(msg)
    if len(dem_files) > 1:
        logger.warning(
            f"Multiple DEM files found below {setup_path}; using {dem_files[0]}."
        )
    return dem_files[0]


def _fill_nearest_for_tile(setup_path, fill_nearest_files):
    """Fill configured files and all cropped meteo files with nearest neighbours."""
    setup_path = Path(setup_path)
    filled_files = []
    filled_inputs = set()
    for input_file in _meteo_fill_nearest_files(setup_path):
        logger.info(f"Filling meteo file {input_file} without mask.")
        try:
            staged_files = fill_nearest(
                input_dir=input_file.parent,
                fname=input_file.name,
                output_dir=input_file.parent.parent / "meteo_filled",
                mask_file=None,
                mask_var=None,
            )
        except Exception as exc:
            msg = (
                f"Failed to fill meteo file {input_file} "
                f"for setup tile {setup_path}."
            )
            with ErrorLogger(logger):
                raise RuntimeError(msg) from exc
        for staged_file in staged_files:
            target = input_file.parent / staged_file.name
            logger.info(
                f"Replacing meteo input {target} with filled file {staged_file}."
            )
            shutil.move(staged_file, target)
            filled_files.append(target)
        filled_inputs.add(input_file)

    if not fill_nearest_files:
        return filled_files

    dem_file = _find_dem_file(setup_path)
    for file_pattern in fill_nearest_files:
        matches = sorted(setup_path.rglob(file_pattern))
        if not matches:
            msg = (
                f"No files match fill-nearest pattern {file_pattern!r} "
                f"below {setup_path}."
            )
            with ErrorLogger(logger):
                raise FileNotFoundError(msg)
        for input_file in matches:
            if input_file in filled_inputs:
                continue
            logger.info(
                f"Filling {input_file} using DEM mask {dem_file} with "
                "fill_nearest parameters: output_dir=input file parent, "
                "mask_var=None, fill_value=-9999.0, default_value=None."
            )
            filled_files.extend(
                fill_nearest(
                    input_dir=input_file.parent,
                    fname=input_file.name,
                    output_dir=input_file.parent,
                    mask_file=dem_file,
                    mask_var=None,
                )
            )
            filled_inputs.add(input_file)
    return filled_files


def _missing_value(data_array):
    """Return the missing-value marker stored in attrs or encoding."""
    return data_array.encoding.get(
        "missing_value",
        data_array.attrs.get(
            "missing_value",
            data_array.encoding.get(
                "_FillValue", data_array.attrs.get("_FillValue", np.nan)
            ),
        ),
    )


def _active_l0_mask_from_file(mask_file):
    """Read a mask file and return a 1/0 mask from non-missing values."""
    mask_file = Path(mask_file)
    with xr.open_dataset(mask_file, engine="netcdf4", mask_and_scale=False) as dataset:
        mask_ds = dataset.load()

    mask_var = get_single_data_var(mask_ds)
    if mask_var is None:
        msg = f"Could not determine single L0 mask variable in {mask_file}."
        with ErrorLogger(logger):
            raise ValueError(msg)

    mask_da = mask_ds[mask_var]
    missing_value = _missing_value(mask_da)
    valid = ~np.isnan(mask_da)
    if missing_value is not None and not np.isnan(missing_value):
        valid &= ~np.isclose(mask_da, float(missing_value))
    return valid.astype(float).to_dataset(name="mask")


def _mask_dem_with_l0_file(dem_file, mask_file):
    """Apply one cropped L0 mask file to the cropped DEM file."""
    dem_file = Path(dem_file)
    mask_file = Path(mask_file)
    logger.info(f"Masking DEM {dem_file} with L0 mask {mask_file}.")

    mask_ds = _active_l0_mask_from_file(mask_file)
    with xr.open_dataset(dem_file, engine="netcdf4", mask_and_scale=False) as dataset:
        dem_ds = dataset.load()

    dem_lon_key = get_coord_key(dem_ds, lon=True)
    dem_lat_key = get_coord_key(dem_ds, lat=True)
    mask_lon_key = get_coord_key(mask_ds, lon=True)
    mask_lat_key = get_coord_key(mask_ds, lat=True)
    mask_regridded = regrid_mask(
        mask_ds=mask_ds,
        lon_key_mask=mask_lon_key,
        lat_key_mask=mask_lat_key,
        target_lon=dem_ds[dem_lon_key],
        target_lat=dem_ds[dem_lat_key],
        mask_key="mask",
        lon_key_target=dem_lon_key,
        lat_key_target=dem_lat_key,
    )

    masked_vars = []
    for var in dem_ds.data_vars:
        if {dem_lat_key, dem_lon_key}.issubset(dem_ds[var].dims):
            dem_ds[var] = dem_ds[var].where(mask_regridded == 1, np.nan)
            masked_vars.append(var)

    if not masked_vars:
        msg = f"No spatial data variables in DEM file {dem_file} could be masked."
        with ErrorLogger(logger):
            raise ValueError(msg)

    write_xarray_to_file(dem_ds, dem_file)
    logger.info(
        f"Applied L0 mask {mask_file} to DEM variables {masked_vars} in {dem_file}."
    )
    return dem_file


def _mask_dem_for_tile(setup_path, l0_mask_files):
    """Apply configured cropped L0 masks to the tile DEM."""
    if not l0_mask_files:
        return []

    setup_path = Path(setup_path)
    dem_file = _find_dem_file(setup_path)
    masked_files = []
    for file_pattern in l0_mask_files:
        matches = sorted(setup_path.rglob(file_pattern))
        if not matches:
            msg = (
                f"No files match L0 DEM mask pattern {file_pattern!r} "
                f"below {setup_path}."
            )
            with ErrorLogger(logger):
                raise FileNotFoundError(msg)
        for mask_file in matches:
            masked_files.append(_mask_dem_with_l0_file(dem_file, mask_file))
    return masked_files


def _ensure_mhm_tile_dirs(setup_path):
    """Create directories needed by mHM inside a tile setup."""
    setup_path = Path(setup_path)
    for dirname in ("output", "restart"):
        directory = setup_path / dirname
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured mHM tile directory exists: {directory}.")


def _write_tile_meteo_header(setup_path, l1_resolution):
    """Regenerate the meteo header for a prepared tile."""
    setup_path = Path(setup_path)
    pre_file = setup_path / "input" / "meteo" / "pre.nc"
    dem_file = setup_path / "input" / "morph" / "dem.nc"
    output_dir = setup_path / "input" / "meteo"
    if not pre_file.is_file():
        msg = f"Cannot recreate restart; missing meteo file {pre_file}."
        with ErrorLogger(logger):
            raise FileNotFoundError(msg)
    if not dem_file.is_file():
        msg = f"Cannot recreate restart; missing DEM file {dem_file}."
        with ErrorLogger(logger):
            raise FileNotFoundError(msg)

    from mhm_tools._cli._create_header import run as run_create_header

    logger.info(
        f"Regenerating meteo header for {setup_path} using {pre_file} and {dem_file}."
    )
    args = argparse.Namespace(
        input_file=pre_file,
        mask_file=dem_file,
        output_dir=output_dir,
        mask_var=None,
        resolution=l1_resolution,
    )
    run_create_header(args)


def _meteo_source_root(input_path):
    """Return the original meteo directory used to rebuild a tile."""
    input_path = Path(input_path)
    for meteo_root in (input_path / "input" / "meteo", input_path / "meteo"):
        if meteo_root.is_dir():
            return meteo_root
    msg = f"Could not find original meteo directory below {input_path}."
    with ErrorLogger(logger):
        raise FileNotFoundError(msg)


def _remove_tile_meteo_dirs(tile):
    """Remove existing meteo directories below a tile setup."""
    tile_path = Path(tile.output_path)
    meteo_dirs = sorted(
        (path for path in tile_path.rglob("meteo") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for meteo_dir in meteo_dirs:
        logger.info(f"Removing damaged meteo directory {meteo_dir}.")
        shutil.rmtree(meteo_dir)


def _crop_tile_meteo_from_original(
    tile,
    input_path,
    l1_resolution,
    l11_resolution,
    crs,
    crop_n_jobs,
    available_mem_gib,
    chunking,
    lat_order,
):
    """Restore a tile meteo directory by cropping it from the original setup."""
    meteo_source = _meteo_source_root(input_path)
    input_path = Path(input_path)
    meteo_output = Path(tile.output_path) / meteo_source.relative_to(input_path)
    meteo_output.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Cropping original meteo directory {meteo_source} to {meteo_output} "
        f"for tile {tile.name}."
    )
    crop_mhm_setup(
        input_path=meteo_source,
        output_path=meteo_output,
        mask_ds=None,
        resolutions=Resolution(l1=l1_resolution, l11=l11_resolution),
        lonslice=tile.lonslice,
        latslice=tile.latslice,
        crs=crs,
        n_jobs=crop_n_jobs,
        filename="*.nc",
        available_mem_gib=available_mem_gib,
        force_header_creation=False,
        chunking=chunking,
        output_var=None,
        no_cropping=False,
        lat_order=lat_order,
        output_suffix=None,
        mask_all=False,
        mask_var="mask",
    )


def _restore_tile_meteo_from_original(
    tile,
    input_path,
    l1_resolution,
    l11_resolution,
    crs,
    crop_n_jobs,
    available_mem_gib,
    chunking,
    lat_order,
):
    """Remove and recrop the meteo directory for a tile."""
    _remove_tile_meteo_dirs(tile)
    _crop_tile_meteo_from_original(
        tile=tile,
        input_path=input_path,
        l1_resolution=l1_resolution,
        l11_resolution=l11_resolution,
        crs=crs,
        crop_n_jobs=crop_n_jobs,
        available_mem_gib=available_mem_gib,
        chunking=chunking,
        lat_order=lat_order,
    )


def _restore_recreated_fill_file_from_original(
    input_file,
    tile,
    input_path,
    l1_resolution,
    l11_resolution,
    crs,
    crop_n_jobs,
    available_mem_gib,
    chunking,
    lat_order,
):
    """Restore one recreated fill file by cropping its original setup source."""
    input_file = Path(input_file)
    tile_path = Path(tile.output_path)
    input_path = Path(input_path)
    try:
        relative_file = input_file.relative_to(tile_path)
    except ValueError:
        logger.debug(
            f"Skipping restore for {input_file}; it is not below tile path "
            f"{tile_path}."
        )
        return None
    if "meteo" in relative_file.parts:
        logger.debug(
            f"Skipping restore for {input_file}; meteo files are restored "
            "through the dedicated meteo recreation path."
        )
        return None
    source_file = input_path / relative_file
    if not source_file.is_file():
        logger.info(
            f"Skipping restore for {input_file}; original source file "
            f"{source_file} does not exist."
        )
        return None
    logger.info(
        f"Restoring recreated fill file {input_file} from original source "
        f"{source_file}."
    )
    crop_mhm_setup(
        input_path=source_file,
        output_path=input_file.parent,
        mask_ds=None,
        resolutions=Resolution(l1=l1_resolution, l11=l11_resolution),
        lonslice=tile.lonslice,
        latslice=tile.latslice,
        crs=crs,
        n_jobs=crop_n_jobs,
        filename=source_file.name,
        available_mem_gib=available_mem_gib,
        force_header_creation=False,
        chunking=chunking,
        output_var=None,
        no_cropping=False,
        lat_order=lat_order,
        output_suffix=None,
        mask_all=False,
        mask_var="mask",
    )
    return input_file


def _restore_recreated_fill_files_from_original(
    tile,
    input_path,
    fill_nearest_files,
    l1_resolution,
    l11_resolution,
    crs,
    crop_n_jobs,
    available_mem_gib,
    chunking,
    lat_order,
):
    """Restore configured recreated fill files from the original setup."""
    if not fill_nearest_files:
        return []
    restored_files = []
    for file_pattern in fill_nearest_files:
        matches = sorted(Path(tile.output_path).rglob(file_pattern))
        if not matches:
            logger.info(
                f"No existing recreated fill files match pattern "
                f"{file_pattern!r} below {tile.output_path}; nothing to "
                "restore from original setup before filling."
            )
            continue
        for input_file in matches:
            restored_file = _restore_recreated_fill_file_from_original(
                input_file=input_file,
                tile=tile,
                input_path=input_path,
                l1_resolution=l1_resolution,
                l11_resolution=l11_resolution,
                crs=crs,
                crop_n_jobs=crop_n_jobs,
                available_mem_gib=available_mem_gib,
                chunking=chunking,
                lat_order=lat_order,
            )
            if restored_file is not None:
                restored_files.append(restored_file)
    logger.info(
        f"Restored {len(restored_files)} recreated fill files for tile "
        f"{tile.name} from original setup."
    )
    logger.debug(f"Restored recreated fill files: {restored_files}.")
    return restored_files


def _tile_meteo_dir(tile):
    """Return the recreated tile meteo directory."""
    for meteo_dir in (
        Path(tile.output_path) / "input" / "meteo",
        Path(tile.output_path) / "meteo",
    ):
        if meteo_dir.is_dir():
            return meteo_dir
    msg = f"Could not find recreated meteo directory below {tile.output_path}."
    with ErrorLogger(logger):
        raise FileNotFoundError(msg)


def _fill_recreated_restart_inputs(tile, fill_nearest_files):
    """Fill tile inputs before restart recreation."""
    meteo_dir = _tile_meteo_dir(tile)
    for meteo_name in ("pre.nc", "pet.nc", "tavg.nc"):
        meteo_file = meteo_dir / meteo_name
        if not meteo_file.is_file():
            msg = f"Cannot recreate restart for {tile.name}; missing {meteo_file}."
            with ErrorLogger(logger):
                raise FileNotFoundError(msg)
        logger.info(
            f"Nearest-neighbour filling recreated meteo file {meteo_file} "
            "without mask and fill_value=2.2."
        )
        staged_files = fill_nearest(
            input_dir=meteo_file.parent,
            fname=meteo_file.name,
            output_dir=meteo_file.parent.parent / "meteo_filled",
            mask_file=None,
            mask_var=None,
            fill_value=2.2,
        )
        for staged_file in staged_files:
            target = meteo_file.parent / staged_file.name
            logger.info(
                f"Replacing meteo input {target} with filled file {staged_file}."
            )
            shutil.move(staged_file, target)

    if not fill_nearest_files:
        return
    for file_pattern in fill_nearest_files:
        matches = sorted(Path(tile.output_path).rglob(file_pattern))
        if not matches:
            msg = (
                f"No files match recreate-restart fill pattern {file_pattern!r} "
                f"below {tile.output_path}."
            )
            with ErrorLogger(logger):
                raise FileNotFoundError(msg)
        for input_file in matches:
            logger.info(
                f"Nearest-neighbour filling recreated input file {input_file} "
                "with fill_nearest parameters: output_dir=input file parent, "
                "mask_file=None, mask_var=None, fill_value=-9999.0, "
                "default_value=1."
            )
            staged_files = fill_nearest(
                input_dir=input_file.parent,
                fname=input_file.name,
                output_dir=input_file.parent,
                mask_file=None,
                mask_var=None,
                fill_value=1,
            )
            logger.debug(f"Filled recreated input files: {staged_files}.")


def _global_center_coords(
    min_bound,
    max_bound,
    resolution,
    increasing=True,
):
    """Return global cell-center coordinates for one spatial axis."""
    resolution = float(resolution)
    n_cells = int((float(max_bound) - float(min_bound)) / resolution + 0.5)
    if n_cells <= 0:
        msg = "Global restart coordinate has no cells. Check bounds and resolution."
        with ErrorLogger(logger):
            raise ValueError(msg)
    coords = float(min_bound) + resolution / 2 + np.arange(n_cells) * resolution
    if not increasing:
        coords = coords[::-1]
    return coords


def _restart_level_from_dim(dim_name, prefix):
    """Return the mHM level suffix for restart grid dimensions."""
    dim_name = str(dim_name)
    if not dim_name.startswith(prefix):
        return None
    level = dim_name[len(prefix) :]
    if not level.isdigit():
        return None
    return f"L{level}"


def _restart_center_coords(lower_left, cellsize, n_cells):
    """Create cell-center coordinates from mHM lower-left grid metadata."""
    return float(lower_left) + float(cellsize) * (np.arange(int(n_cells)) + 0.5)


def _restart_descending_center_coords(lower_left, cellsize, n_cells):
    """Create north-to-south cell centers from lower-left grid metadata."""
    centers = _restart_center_coords(lower_left, cellsize, n_cells)
    return centers[::-1]


def _add_restart_spatial_coords(dataset):
    """Attach spatial coordinates derived from mHM restart grid attributes."""
    rename_dims = {}
    coord_updates = {}
    grid_attrs = {
        key: dataset.attrs[key]
        for key in dataset.attrs
        if str(key).startswith(("xllcorner_", "yllcorner_", "cellsize_"))
    }
    logger.debug(
        f"Preparing restart spatial coordinates from dims={dict(dataset.sizes)} "
        f"attrs={grid_attrs}."
    )
    for col_dim in dataset.dims:
        level = _restart_level_from_dim(col_dim, "ncols")
        if level is None:
            continue
        row_dim = f"nrows{level[1:]}"
        if row_dim not in dataset.dims:
            continue

        xllcorner_key = f"xllcorner_{level}"
        yllcorner_key = f"yllcorner_{level}"
        cellsize_key = f"cellsize_{level}"
        missing_keys = [
            key
            for key in (xllcorner_key, yllcorner_key, cellsize_key)
            if key not in dataset.attrs
        ]
        if missing_keys:
            msg = (
                f"Restart file has dimensions {col_dim!r}/{row_dim!r} but is "
                f"missing grid attributes {missing_keys}."
            )
            with ErrorLogger(logger):
                raise ValueError(msg)

        lon_name = "lon" if level == "L1" else f"lon_{level}"
        lat_name = "lat" if level == "L1" else f"lat_{level}"
        logger.info(
            f"Transforming restart {level} grid: {col_dim}/{row_dim} -> "
            f"{lon_name}/{lat_name} (xllcorner={dataset.attrs[xllcorner_key]}, "
            f"yllcorner={dataset.attrs[yllcorner_key]}, "
            f"cellsize={dataset.attrs[cellsize_key]}, "
            f"shape={dataset.sizes[col_dim]}x{dataset.sizes[row_dim]})."
        )
        coord_updates[lon_name] = (
            row_dim,
            _restart_center_coords(
                dataset.attrs[xllcorner_key],
                dataset.attrs[cellsize_key],
                dataset.sizes[row_dim],
            ),
            {"axis": "X"},
        )
        coord_updates[lat_name] = (
            col_dim,
            _restart_descending_center_coords(
                dataset.attrs[yllcorner_key],
                dataset.attrs[cellsize_key],
                dataset.sizes[col_dim],
            ),
            {"axis": "Y"},
        )
        rename_dims[row_dim] = lon_name
        rename_dims[col_dim] = lat_name

    if not rename_dims:
        logger.debug(
            "No mHM restart ncols*/nrows* dimensions found; keeping existing coords."
        )
        return dataset
    logger.debug(f"Swapping restart dimensions with spatial coords: {rename_dims}.")
    return dataset.assign_coords(coord_updates).swap_dims(rename_dims)


def _load_restart_dataset_for_merge(restart_file):
    """Load one restart file and attach merge coordinates."""
    with get_xarray_ds_from_file(restart_file) as dataset:
        loaded = dataset.load()
    logger.debug(
        f"Loaded restart file {restart_file} sizes={dict(loaded.sizes)} "
        f"data_vars={list(loaded.data_vars)} coords={list(loaded.coords)}."
    )
    return _add_restart_spatial_coords(loaded)


def _restart_spatial_coord_names(level):
    """Return lon/lat coordinate names used for a transformed restart level."""
    if level == "L1":
        return "lon", "lat"
    return f"lon_{level}", f"lat_{level}"


def _restart_levels_from_dataset(dataset):
    """Return restart levels that have transformed spatial coordinates."""
    levels = []
    for key in dataset.attrs:
        key_str = str(key)
        if not key_str.startswith("xllcorner_L"):
            continue
        level = key_str.replace("xllcorner_", "", 1)
        lon_key, lat_key = _restart_spatial_coord_names(level)
        if lon_key in dataset.coords and lat_key in dataset.coords:
            levels.append(level)
    return sorted(levels)


def _global_restart_coords_for_level(
    level,
    template,
    lon_min_bound,
    lon_max_bound,
    lat_min_bound,
    lat_max_bound,
    l1_resolution,
    lat_order,
):
    """Create final-domain restart coordinates for one restart level."""
    resolution = (
        float(l1_resolution)
        if level == "L1"
        else float(template.attrs[f"cellsize_{level}"])
    )
    lon_coords = _global_center_coords(
        min_bound=lon_min_bound,
        max_bound=lon_max_bound,
        resolution=resolution,
        increasing=True,
    )
    lat_coords = _global_center_coords(
        min_bound=lat_min_bound,
        max_bound=lat_max_bound,
        resolution=resolution,
        increasing=lat_order == "increasing",
    )
    return lon_coords, lat_coords


def _global_restart_coords(
    template,
    lon_min_bound,
    lon_max_bound,
    lat_min_bound,
    lat_max_bound,
    l1_resolution,
    lat_order,
):
    """Build final-domain coordinates for all transformed restart levels."""
    global_coords = {}
    for level in _restart_levels_from_dataset(template):
        lon_key, lat_key = _restart_spatial_coord_names(level)
        lon_coords, lat_coords = _global_restart_coords_for_level(
            level=level,
            template=template,
            lon_min_bound=lon_min_bound,
            lon_max_bound=lon_max_bound,
            lat_min_bound=lat_min_bound,
            lat_max_bound=lat_max_bound,
            l1_resolution=l1_resolution,
            lat_order=lat_order,
        )
        global_coords[lon_key] = xr.DataArray(
            lon_coords,
            dims=[lon_key],
            attrs=template.coords.get(lon_key, xr.DataArray()).attrs,
        )
        global_coords[lat_key] = xr.DataArray(
            lat_coords,
            dims=[lat_key],
            attrs=template.coords.get(lat_key, xr.DataArray()).attrs,
        )
    if "lon" not in global_coords and "lon" in template.dims:
        lon_coords = _global_center_coords(
            min_bound=lon_min_bound,
            max_bound=lon_max_bound,
            resolution=l1_resolution,
            increasing=True,
        )
        global_coords["lon"] = xr.DataArray(lon_coords, dims=["lon"])
    if "lat" not in global_coords and "lat" in template.dims:
        lat_coords = _global_center_coords(
            min_bound=lat_min_bound,
            max_bound=lat_max_bound,
            resolution=l1_resolution,
            increasing=lat_order == "increasing",
        )
        global_coords["lat"] = xr.DataArray(lat_coords, dims=["lat"])
    return global_coords


def _init_merged_restart_dataset(template, global_coords):
    """Create an empty final restart dataset from one tile template."""
    merged = xr.Dataset(attrs=dict(template.attrs))
    for coord_name, coord in template.coords.items():
        if coord_name in global_coords:
            continue
        if any(dim in global_coords for dim in coord.dims):
            continue
        merged = merged.assign_coords({coord_name: coord.copy(deep=True)})
    for coord_name, coord in global_coords.items():
        merged = merged.assign_coords({coord_name: coord})
    return merged


def _is_spatial_restart_var(data_array, global_coords):
    """Return whether a restart variable uses any merged spatial dimension."""
    return any(dim in global_coords for dim in data_array.dims)


def _spatial_fill_dtype(data_array):
    """Return a safe dtype for initializing merged spatial restart arrays."""
    dtype = data_array.dtype
    if np.issubdtype(dtype, np.floating):
        return dtype
    return float


def _ensure_merged_restart_var(merged, name, tile_data_array, global_coords):
    """Initialize one output variable if it is not present yet."""
    if name in merged:
        return merged
    if not _is_spatial_restart_var(tile_data_array, global_coords):
        merged[name] = tile_data_array.copy(deep=True)
        return merged

    coords = {}
    shape = []
    for dim in tile_data_array.dims:
        if dim in global_coords:
            coords[dim] = global_coords[dim]
            shape.append(global_coords[dim].size)
        else:
            if dim in tile_data_array.coords:
                coords[dim] = tile_data_array[dim].copy(deep=True)
            shape.append(tile_data_array.sizes[dim])
    merged[name] = xr.DataArray(
        np.full(shape, np.nan, dtype=_spatial_fill_dtype(tile_data_array)),
        dims=tile_data_array.dims,
        coords=coords,
        attrs=tile_data_array.attrs.copy(),
        name=name,
    )
    logger.debug(
        f"Initialized merged restart variable {name} with dims "
        f"{merged[name].dims} and shape {merged[name].shape}."
    )
    return merged


def _restart_coord_match_tolerance(global_values):
    """Return tolerance for matching restart tile coordinates to global coords."""
    values = np.asarray(global_values, dtype=float)
    magnitude = max(float(np.nanmax(np.abs(values))), 1.0)
    if values.size < 2:
        return np.finfo(float).eps * magnitude * 64
    steps = np.abs(np.diff(values))
    steps = steps[steps > 0]
    if steps.size == 0:
        return np.finfo(float).eps * magnitude * 64
    grid_tolerance = float(np.nanmin(steps)) * 1.0e-6
    precision_tolerance = np.finfo(float).eps * magnitude * 64
    return max(grid_tolerance, precision_tolerance)


def _nearest_global_coord_labels(dim, tile_values, global_values):
    """Map tile coordinate values to nearest global coordinate labels."""
    tile_values = np.asarray(tile_values, dtype=float)
    global_values = np.asarray(global_values, dtype=float)
    if global_values.size == 0:
        msg = f"Global restart coordinate {dim!r} has no values."
        with ErrorLogger(logger):
            raise ValueError(msg)

    increasing = global_values[0] <= global_values[-1]
    sorted_global = global_values if increasing else global_values[::-1]
    insert_positions = np.searchsorted(sorted_global, tile_values, side="left")
    right_positions = np.clip(insert_positions, 0, sorted_global.size - 1)
    left_positions = np.clip(insert_positions - 1, 0, sorted_global.size - 1)
    choose_left = np.abs(tile_values - sorted_global[left_positions]) <= np.abs(
        tile_values - sorted_global[right_positions]
    )
    sorted_positions = np.where(choose_left, left_positions, right_positions)
    global_positions = (
        sorted_positions if increasing else global_values.size - 1 - sorted_positions
    )

    matched_values = global_values[global_positions]
    tolerance = _restart_coord_match_tolerance(global_values)
    missing = np.abs(tile_values - matched_values) > tolerance
    if np.any(missing):
        sample = tile_values[missing][:5].tolist()
        msg = (
            f"Restart tile coordinate {dim!r} contains values outside the merged "
            f"domain or off the global grid; sample unmatched values: {sample}."
        )
        with ErrorLogger(logger):
            raise KeyError(msg)
    return matched_values


def _tile_mask_file_for_restart(restart_file):
    """Find the mask section written for the tile that produced a restart file."""
    restart_file = Path(restart_file)
    for directory in (restart_file.parent, *restart_file.parents):
        mask_file = directory / "mask_tile.nc"
        if mask_file.is_file():
            return mask_file
    return None


def _load_tile_mask_for_restart(restart_file):
    """Load the mask section belonging to a restart tile, if available."""
    mask_file = _tile_mask_file_for_restart(restart_file)
    if mask_file is None:
        logger.warning(f"No tile mask found for restart file {restart_file}.")
        return None
    logger.info(f"Applying tile mask {mask_file} to restart tile {restart_file}.")
    with get_xarray_ds_from_file(mask_file) as mask_ds:
        return mask_ds.load()


def _mask_restart_tile_with_tile_mask(dataset, restart_file, mask_var):
    """Mask L1 spatial variables in one restart tile using its tile mask."""
    tile_mask = _load_tile_mask_for_restart(restart_file)
    if tile_mask is None:
        return dataset
    mask_da = _get_mask_data_array(tile_mask, mask_var)
    if mask_da is None:
        return dataset
    if mask_da.name is None:
        mask_da = mask_da.rename(mask_var)
    tile_mask_ds = mask_da.to_dataset(name=mask_da.name)
    mask_lon_key = get_coord_key(mask_da, lon=True)
    mask_lat_key = get_coord_key(mask_da, lat=True)
    masked_vars = []
    for var in dataset.data_vars:
        data_array = dataset[var]
        if "lon" not in data_array.dims or "lat" not in data_array.dims:
            continue
        mask_regridded = regrid_mask(
            mask_ds=tile_mask_ds,
            lon_key_mask=mask_lon_key,
            lat_key_mask=mask_lat_key,
            target_lon=dataset["lon"],
            target_lat=dataset["lat"],
            mask_key=mask_da.name,
            lon_key_target="lon",
            lat_key_target="lat",
        )
        dataset[var] = data_array.where(mask_regridded == 1, np.nan)
        masked_vars.append(var)
    logger.info(
        f"Applied tile mask for {restart_file} to {len(masked_vars)} variables."
    )
    logger.debug(f"Tile-masked restart variables: {masked_vars}.")
    return dataset


def _write_restart_tile_to_merged(merged, tile_dataset, global_coords):
    """Write one transformed restart tile into the final-domain dataset."""
    for var in tile_dataset.data_vars:
        tile_data_array = tile_dataset[var]
        merged = _ensure_merged_restart_var(
            merged=merged,
            name=var,
            tile_data_array=tile_data_array,
            global_coords=global_coords,
        )
        if not _is_spatial_restart_var(tile_data_array, global_coords):
            continue
        indexers = {
            dim: _nearest_global_coord_labels(
                dim=dim,
                tile_values=tile_data_array[dim].values,
                global_values=global_coords[dim].values,
            )
            for dim in tile_data_array.dims
            if dim in global_coords
        }
        logger.debug(f"Writing restart variable {var} to output indexers {indexers}.")
        target = merged[var].loc[indexers]
        aligned_tile = tile_data_array.transpose(*target.dims)
        merged[var].loc[indexers] = aligned_tile.data
    return merged


def _set_restart_grid_attrs(dataset, lon_min_bound=None, lat_min_bound=None):
    """Set merged restart grid attributes from transformed spatial coordinates."""
    levels = sorted(
        str(key).replace("xllcorner_", "", 1)
        for key in dataset.attrs
        if str(key).startswith("xllcorner_L")
    )
    for level in levels:
        lon_key = "lon" if level == "L1" else f"lon_{level}"
        lat_key = "lat" if level == "L1" else f"lat_{level}"
        cellsize_key = f"cellsize_{level}"
        if lon_key not in dataset.coords or lat_key not in dataset.coords:
            logger.debug(
                f"Skipping {level} restart attrs update; missing coords "
                f"{lon_key}/{lat_key}."
            )
            continue
        cellsize = float(dataset.attrs.get(cellsize_key, np.nan))
        if np.isnan(cellsize):
            logger.debug(
                f"Skipping {level} restart attrs update; missing {cellsize_key}."
            )
            continue
        lon_values = np.asarray(dataset[lon_key].values, dtype=float)
        lat_values = np.asarray(dataset[lat_key].values, dtype=float)
        xllcorner = (
            float(lon_min_bound)
            if lon_min_bound is not None
            else float(np.nanmin(lon_values) - cellsize / 2)
        )
        yllcorner = (
            float(lat_min_bound)
            if lat_min_bound is not None
            else float(np.nanmin(lat_values) - cellsize / 2)
        )
        dataset.attrs.update(
            {
                f"xllcorner_{level}": xllcorner,
                f"yllcorner_{level}": yllcorner,
                cellsize_key: cellsize,
                f"ncols_{level}": len(lon_values),
                f"nrows_{level}": len(lat_values),
            }
        )
        logger.info(
            f"Updated merged {level} grid attrs: "
            f"xllcorner={dataset.attrs[f'xllcorner_{level}']}, "
            f"yllcorner={dataset.attrs[f'yllcorner_{level}']}, "
            f"cellsize={dataset.attrs[cellsize_key]}, "
            f"ncols={dataset.attrs[f'ncols_{level}']}, "
            f"nrows={dataset.attrs[f'nrows_{level}']}."
        )
    return dataset


def _mask_sources(mask_ds):
    """Normalize one or more final restart mask datasets to a list."""
    if mask_ds is None:
        return []
    if isinstance(mask_ds, (list, tuple)):
        return [mask_source for mask_source in mask_ds if mask_source is not None]
    return [mask_ds]


def _combined_restart_mask(mask_ds, mask_var, target_lon, target_lat):
    """Regrid one or more masks to the restart grid and combine them with OR."""
    combined_mask = None
    lon_key_target = target_lon.dims[0]
    lat_key_target = target_lat.dims[0]
    for mask_source in _mask_sources(mask_ds):
        mask_da = _get_mask_data_array(mask_source, mask_var)
        if mask_da is None:
            continue
        if mask_da.name is None:
            mask_da = mask_da.rename(mask_var)
        mask_input = mask_da.to_dataset(name=mask_da.name)
        mask_lon_key = get_coord_key(mask_da, lon=True)
        mask_lat_key = get_coord_key(mask_da, lat=True)
        logger.info(
            f"Applying final restart mask {mask_da.name} on coords "
            f"{mask_lon_key}/{mask_lat_key}."
        )
        mask_regridded = regrid_mask(
            mask_ds=mask_input,
            lon_key_mask=mask_lon_key,
            lat_key_mask=mask_lat_key,
            target_lon=target_lon,
            target_lat=target_lat,
            mask_key=mask_da.name,
            lon_key_target=lon_key_target,
            lat_key_target=lat_key_target,
        )
        active_mask = mask_regridded > 0
        combined_mask = (
            active_mask if combined_mask is None else combined_mask | active_mask
        )
    return combined_mask


def _batched(items, batch_size):
    """Yield lists of items with at most batch_size elements."""
    batch_size = max(1, int(batch_size))
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _prepare_restart_tile_for_merge(restart_file, mask_var):
    """Load and mask one restart tile before serial insertion."""
    dataset = _load_restart_dataset_for_merge(restart_file)
    return _mask_restart_tile_with_tile_mask(dataset, restart_file, mask_var)


def _prepare_restart_tile_batch_for_merge(restart_files, mask_var, n_jobs):
    """Prepare one batch of restart tiles, preserving input order."""
    if int(n_jobs) == 1:
        return [
            _prepare_restart_tile_for_merge(restart_file, mask_var)
            for restart_file in restart_files
        ]
    return Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_prepare_restart_tile_for_merge)(restart_file, mask_var)
        for restart_file in restart_files
    )


def merge_mhm_restart_files(
    restart_files,
    output_file,
    lon_min_bound,
    lon_max_bound,
    lat_min_bound,
    lat_max_bound,
    l1_resolution,
    lat_order="decreasing",
    mask_ds=None,
    mask_var="mask",
    n_jobs=1,
):
    """Merge tiled mHM restart files without renaming variables."""
    restart_files = [Path(restart_file) for restart_file in restart_files]
    if not restart_files:
        msg = "No mHM restart files were provided for merging."
        with ErrorLogger(logger):
            raise ValueError(msg)
    logger.info(
        f"Starting mHM restart merge for {len(restart_files)} files into {output_file}."
    )
    logger.debug(f"Restart merge input files: {restart_files}.")
    merged = None
    global_coords = None
    n_jobs = max(1, int(n_jobs))
    batch_size = max(1, 2 * n_jobs)
    logger.info(
        f"Preparing restart merge tiles in batches of {batch_size} "
        f"with n_jobs={n_jobs}."
    )
    written_tiles = 0
    for batch_number, restart_file_batch in enumerate(
        _batched(restart_files, batch_size), start=1
    ):
        logger.info(
            f"Preparing restart merge batch {batch_number} with "
            f"{len(restart_file_batch)} files."
        )
        datasets = _prepare_restart_tile_batch_for_merge(
            restart_files=restart_file_batch,
            mask_var=mask_var,
            n_jobs=n_jobs,
        )
        for dataset in datasets:
            written_tiles += 1
            if merged is None:
                global_coords = _global_restart_coords(
                    template=dataset,
                    lon_min_bound=lon_min_bound,
                    lon_max_bound=lon_max_bound,
                    lat_min_bound=lat_min_bound,
                    lat_max_bound=lat_max_bound,
                    l1_resolution=l1_resolution,
                    lat_order=lat_order,
                )
                logger.info(
                    f"Initialized direct restart tile writer with global coords: "
                    f"{ {key: value.size for key, value in global_coords.items()} }."
                )
                merged = _init_merged_restart_dataset(dataset, global_coords)
                logger.debug(
                    f"Initial restart merge dataset sizes: {dict(merged.sizes)}."
                )
            logger.info(
                f"Writing restart tile {written_tiles}/{len(restart_files)} "
                f"into output."
            )
            logger.debug(f"Next restart dataset sizes: {dict(dataset.sizes)}.")
            merged = _write_restart_tile_to_merged(merged, dataset, global_coords)
            del dataset
            logger.debug(f"Merged restart dataset sizes now: {dict(merged.sizes)}.")
    logger.info(f"Wrote {len(restart_files)} mHM restart tiles for {output_file}.")
    lon_key = get_coord_key(merged, lon=True)
    lat_key = get_coord_key(merged, lat=True)
    logger.info(
        f"Merged restart grid uses "
        f"lon={lon_key} ({merged.sizes.get(lon_key)} cells), "
        f"lat={lat_key} ({merged.sizes.get(lat_key)} cells)."
    )
    merged = _set_restart_grid_attrs(
        merged,
        lon_min_bound=lon_min_bound,
        lat_min_bound=lat_min_bound,
    )
    if mask_ds is not None:
        mask_regridded = _combined_restart_mask(
            mask_ds=mask_ds,
            mask_var=mask_var,
            target_lon=merged[lon_key],
            target_lat=merged[lat_key],
        )
        masked_vars = []
        if mask_regridded is not None:
            for var in merged.data_vars:
                if lon_key in merged[var].dims and lat_key in merged[var].dims:
                    merged[var] = merged[var].where(mask_regridded, np.nan)
                    masked_vars.append(var)
            merged.attrs["nCells_L1"] = int(np.sum(mask_regridded.values))
            logger.info(
                f"Applied final restart mask to {len(masked_vars)} variables; "
                f"nCells_L1={merged.attrs['nCells_L1']}."
            )
            logger.debug(f"Masked restart variables: {masked_vars}.")
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing merged mHM restart file to {output_file}.")
    write_xarray_to_file(merged, output_file)
    logger.info(f"Finished writing merged mHM restart file to {output_file}.")
    return output_file


def _prepare_tile_setup(  # noqa: PLR0913
    tile,
    input_path,
    mask_ds,
    l1_resolution,
    l11_resolution,
    crs,
    filename,
    available_mem_gib,
    force_header_creation,
    chunking,
    output_var,
    no_cropping,
    lat_order,
    output_suffix,
    mask_var,
    crop_n_jobs,
    fill_nearest_files,
    l0_mask_files,
    tile_number,
):
    """Crop and prepare one setup tile before any mHM runs are started."""
    logger.info(f"Preparing setup for tile {tile.name}, number: {tile_number}")
    logger.info(f"Crop_setup tile {tile.name} - {tile_number}")
    crop_mhm_setup(
        input_path=input_path,
        output_path=tile.output_path,
        mask_ds=None,
        resolutions=Resolution(l1=l1_resolution, l11=l11_resolution),
        lonslice=tile.lonslice,
        latslice=tile.latslice,
        crs=crs,
        n_jobs=crop_n_jobs,
        filename=filename,
        available_mem_gib=available_mem_gib,
        force_header_creation=force_header_creation,
        chunking=chunking,
        output_var=output_var,
        no_cropping=no_cropping,
        lat_order=lat_order,
        output_suffix=output_suffix,
        mask_all=False,
        mask_var=mask_var,
    )
    logger.info(f"Write mask for tile {tile.name} - {tile_number}")
    _write_tile_mask_section(tile, mask_ds, mask_var)

    logger.info(f"Prepare mHM directories for tile {tile.name} - {tile_number}")
    _ensure_mhm_tile_dirs(tile.output_path)

    logger.info(f"Mask DEM for tile {tile.name} - {tile_number}")
    _mask_dem_for_tile(tile.output_path, l0_mask_files)

    logger.info(f"Fill nearest for tile {tile.name} - {tile_number}")
    _fill_nearest_for_tile(tile.output_path, fill_nearest_files)

    return tile


def _run_mhm_for_tile(
    tile,
    mhm_executable,
    mhm_packages,
    mhm_args,
    restart_pattern,
    require_restart,
    tile_number,
):
    """Run mHM for one prepared setup tile and return its restart files."""
    logger.info(f"Running mHM for tile {tile.name}, number: {tile_number}")
    restart_files = []
    msg = ""
    status = 0
    run_started_at = time.time() - 1.0
    runner = MHMRunner(
        mhm_executable=mhm_executable,
        mhm_packages=mhm_packages,
        mhm_args=mhm_args,
    )
    logger.info(f"Run mhm for tile {tile.name} - {tile_number}")
    try:
        runner.run_mhm(tile.output_path)
        restart_files = _find_restart_files(
            tile.output_path,
            restart_pattern=restart_pattern,
            min_mtime=run_started_at,
        )
        if require_restart and not restart_files:
            all_matching_restart_files = _find_restart_files(
                tile.output_path,
                restart_pattern=restart_pattern,
            )
            msg = (
                f"mHM finished for {tile.name} but no restart file matching "
                f"{restart_pattern!r} was created or updated in {tile.output_path}."
            )
            if all_matching_restart_files:
                msg += f" Existing matching files: {all_matching_restart_files}"
            logger.error(msg)
            status = 1
    except RuntimeError as e:
        msg = f"mHM run for tile {tile.name} - {tile_number} failed with {e}"
        logger.error(msg)
        status = 1

    return {
        "tile": tile.name,
        "restart_files": restart_files,
        "status": status,
        "message": msg,
    }


def _summarize_failed_tiles(
    failed_tiles,
    output_path,
    filename="failed_mhm_tiles.txt",
    title="Failed mHM tiles",
):
    """Print and persist tile failure messages."""
    output_path = Path(output_path)
    failed_tiles_file = output_path / filename
    if failed_tiles:
        lines = [
            f"{failed_tile['tile']}: {failed_tile['message']}"
            for failed_tile in failed_tiles
        ]
        summary = f"{title}:\n" + "\n".join(lines)
    else:
        summary = f"{title}: none"
    print(summary)  # noqa: T201 - CLI summary requested for tile failures.
    failed_tiles_file.parent.mkdir(parents=True, exist_ok=True)
    failed_tiles_file.write_text(summary + "\n")
    return failed_tiles_file


def _tile_dir_missing(tile):
    """Return the tile output path if it is missing."""
    return None if tile.output_path.is_dir() else tile.output_path


def _tile_if_dir_missing(tile):
    """Return the tile object if its output path is missing."""
    return tile if _tile_dir_missing(tile) else None


def _validate_prepared_tile_dirs(tiles, n_jobs=1, raise_on_missing=True):
    """Require previously prepared tile directories when setup creation is skipped."""
    if int(n_jobs) == 1:
        missing_tiles = [tile for tile in tiles if _tile_if_dir_missing(tile)]
    else:
        missing_tiles = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_tile_if_dir_missing)(tile) for tile in tiles
        )
        missing_tiles = [tile for tile in missing_tiles if tile is not None]
    if missing_tiles:
        missing_paths = [tile.output_path for tile in missing_tiles]
        msg = (
            "Tile creation is disabled, but the following tile directories do "
            f"not exist: {missing_paths}"
        )
        if raise_on_missing:
            with ErrorLogger(logger):
                raise FileNotFoundError(msg)
        else:
            logger.error(msg)
            return missing_tiles
    return []


def _prepare_tiles_for_mhm(  # noqa: PLR0913
    tiles,
    skip_tile_creation,
    n_jobs,
    input_path,
    mask_ds,
    l1_resolution,
    l11_resolution,
    crs,
    filename,
    available_mem_gib,
    force_header_creation,
    chunking,
    output_var,
    no_cropping,
    lat_order,
    output_suffix,
    mask_var,
    crop_n_jobs,
    fill_nearest_files,
    l0_mask_files,
):
    """Prepare setup tiles unless existing tile directories should be reused."""
    missing_tiles = []
    created_tiles = []
    if skip_tile_creation:
        logger.info(
            f"Tile creation disabled; reusing {len(tiles)} existing tile setups."
        )
        missing_tiles = _validate_prepared_tile_dirs(
            tiles,
            n_jobs=n_jobs,
            raise_on_missing=False,
        )
        if not missing_tiles:
            return tiles
        logger.info(f"Preparing {len(missing_tiles)} missing tile setups before reuse.")
    else:
        missing_tiles = tiles
    if int(n_jobs) == 1:
        created_tiles = [
            _prepare_tile_setup(
                tile=tile,
                input_path=input_path,
                mask_ds=mask_ds,
                l1_resolution=l1_resolution,
                l11_resolution=l11_resolution,
                crs=crs,
                filename=filename,
                available_mem_gib=available_mem_gib,
                force_header_creation=force_header_creation,
                chunking=chunking,
                output_var=output_var,
                no_cropping=no_cropping,
                lat_order=lat_order,
                output_suffix=output_suffix,
                mask_var=mask_var,
                crop_n_jobs=crop_n_jobs,
                fill_nearest_files=fill_nearest_files,
                l0_mask_files=l0_mask_files,
                tile_number=tile_number,
            )
            for tile_number, tile in enumerate(missing_tiles)
        ]
    else:
        created_tiles = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_prepare_tile_setup)(
                tile=tile,
                input_path=input_path,
                mask_ds=mask_ds,
                l1_resolution=l1_resolution,
                l11_resolution=l11_resolution,
                crs=crs,
                filename=filename,
                available_mem_gib=available_mem_gib,
                force_header_creation=force_header_creation,
                chunking=chunking,
                output_var=output_var,
                no_cropping=no_cropping,
                lat_order=lat_order,
                output_suffix=output_suffix,
                mask_var=mask_var,
                crop_n_jobs=crop_n_jobs,
                fill_nearest_files=fill_nearest_files,
                l0_mask_files=l0_mask_files,
                tile_number=tile_number,
            )
            for tile_number, tile in enumerate(missing_tiles)
        )
    return tiles if skip_tile_creation else created_tiles


def _run_mhm_for_tiles(
    prepared_tiles,
    mhm_n_jobs,
    mhm_executable,
    mhm_packages,
    mhm_args,
    restart_pattern,
    require_restart,
):
    """Run mHM for prepared setup tiles."""
    if int(mhm_n_jobs) == 1:
        return [
            _run_mhm_for_tile(
                tile=tile,
                mhm_executable=mhm_executable,
                mhm_packages=mhm_packages,
                mhm_args=mhm_args,
                restart_pattern=restart_pattern,
                require_restart=require_restart,
                tile_number=tile_number,
            )
            for tile_number, tile in enumerate(prepared_tiles)
        ]

    return Parallel(n_jobs=mhm_n_jobs, backend="loky")(
        delayed(_run_mhm_for_tile)(
            tile=tile,
            mhm_executable=mhm_executable,
            mhm_packages=mhm_packages,
            mhm_args=mhm_args,
            restart_pattern=restart_pattern,
            require_restart=require_restart,
            tile_number=tile_number,
        )
        for tile_number, tile in enumerate(prepared_tiles)
    )


def _deduplicate_restart_files_by_relative_path(search_results):
    """Remove duplicate restart files that share the same relative tile path."""
    restart_files = []
    seen = set()
    for search_root, files in search_results:
        for restart_file in files:
            relative_path = _path_relative_to(restart_file, search_root)
            key = relative_path if relative_path is not None else restart_file
            if key in seen:
                continue
            seen.add(key)
            restart_files.append(restart_file)
    return restart_files


def _restart_output_tile_path(tile, setup_path, restart_output_path):
    """Return the relocated restart-output path for a prepared tile."""
    if setup_path is None or restart_output_path is None:
        return None
    relative_tile_path = _path_relative_to(tile.output_path, setup_path)
    if relative_tile_path is None:
        return None
    return Path(restart_output_path) / relative_tile_path


def _collect_restart_files_for_tile(
    tile,
    restart_pattern,
    require_restart,
    tile_number,
    setup_path=None,
    restart_output_path=None,
):
    """Collect existing restart files for one prepared setup tile."""
    search_results = []
    restart_files_in_tile = _find_restart_files(
        tile.output_path,
        restart_pattern=restart_pattern,
    )
    search_results.append((tile.output_path, restart_files_in_tile))
    moved_tile_path = _restart_output_tile_path(
        tile,
        setup_path=setup_path,
        restart_output_path=restart_output_path,
    )
    if moved_tile_path is not None and moved_tile_path != tile.output_path:
        restart_files_in_output = _find_restart_files(
            moved_tile_path,
            restart_pattern=restart_pattern,
        )
        search_results.append((moved_tile_path, restart_files_in_output))
    restart_files = _deduplicate_restart_files_by_relative_path(search_results)
    msg = ""
    status = 0
    if require_restart and not restart_files:
        search_paths = [tile.output_path]
        if moved_tile_path is not None and moved_tile_path != tile.output_path:
            search_paths.append(moved_tile_path)
        search_location = search_paths[0] if len(search_paths) == 1 else search_paths
        msg = (
            f"No restart file matching {restart_pattern!r} was found for "
            f"{tile.name} in {search_location} while mHM runs were skipped."
        )
        logger.error(msg)
        status = 1
    logger.info(
        f"Collected {len(restart_files)} restart files for tile {tile.name}, "
        f"number: {tile_number}"
    )
    return {
        "tile": tile.name,
        "restart_files": restart_files,
        "status": status,
        "message": msg,
    }


def _collect_restart_files_for_tiles(
    prepared_tiles,
    restart_pattern,
    require_restart,
    n_jobs=1,
    setup_path=None,
    restart_output_path=None,
):
    """Collect existing restart files for prepared setup tiles."""
    if int(n_jobs) == 1:
        return [
            _collect_restart_files_for_tile(
                tile=tile,
                restart_pattern=restart_pattern,
                require_restart=require_restart,
                tile_number=tile_number,
                setup_path=setup_path,
                restart_output_path=restart_output_path,
            )
            for tile_number, tile in enumerate(prepared_tiles)
        ]

    return Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_collect_restart_files_for_tile)(
            tile=tile,
            restart_pattern=restart_pattern,
            require_restart=require_restart,
            tile_number=tile_number,
            setup_path=setup_path,
            restart_output_path=restart_output_path,
        )
        for tile_number, tile in enumerate(prepared_tiles)
    )


def _restart_result_needs_recreation(result):
    """Return whether a tile result failed because it has no restart files."""
    return int(result.get("status", result.get("Status", 0))) == 1 and not result.get(
        "restart_files", []
    )


def _recreate_restart_for_tile(  # noqa: PLR0913
    tile,
    tile_number,
    input_path,
    l1_resolution,
    l11_resolution,
    crs,
    crop_n_jobs,
    available_mem_gib,
    chunking,
    lat_order,
    fill_nearest_files,
    mhm_executable,
    mhm_packages,
    mhm_args,
    restart_pattern,
    require_restart,
):
    """Repair one tile with missing restart inputs and rerun mHM."""
    logger.info(f"Recreating missing restart for tile {tile.name}.")
    _restore_tile_meteo_from_original(
        tile=tile,
        input_path=input_path,
        l1_resolution=l1_resolution,
        l11_resolution=l11_resolution,
        crs=crs,
        crop_n_jobs=crop_n_jobs,
        available_mem_gib=available_mem_gib,
        chunking=chunking,
        lat_order=lat_order,
    )
    _write_tile_meteo_header(tile.output_path, l1_resolution)
    _restore_recreated_fill_files_from_original(
        tile=tile,
        input_path=input_path,
        fill_nearest_files=fill_nearest_files,
        l1_resolution=l1_resolution,
        l11_resolution=l11_resolution,
        crs=crs,
        crop_n_jobs=crop_n_jobs,
        available_mem_gib=available_mem_gib,
        chunking=chunking,
        lat_order=lat_order,
    )
    _fill_recreated_restart_inputs(tile, fill_nearest_files)
    return _run_mhm_for_tile(
        tile=tile,
        mhm_executable=mhm_executable,
        mhm_packages=mhm_packages,
        mhm_args=mhm_args,
        restart_pattern=restart_pattern,
        require_restart=require_restart,
        tile_number=tile_number,
    )


def _recreate_missing_restart_results(  # noqa: PLR0913
    restart_files_by_tile,
    prepared_tiles,
    input_path,
    l1_resolution,
    l11_resolution,
    crs,
    crop_n_jobs,
    available_mem_gib,
    chunking,
    lat_order,
    fill_nearest_files,
    mhm_executable,
    mhm_packages,
    mhm_args,
    restart_pattern,
    require_restart,
    n_jobs,
):
    """Recreate and rerun tiles whose restart files are missing."""
    recreate_indices = [
        index
        for index, result in enumerate(restart_files_by_tile)
        if _restart_result_needs_recreation(result)
    ]
    if not recreate_indices:
        return restart_files_by_tile

    logger.info(
        f"Recreating restart files for {len(recreate_indices)} tiles "
        f"with n_jobs={n_jobs}."
    )

    def _recreate(index):
        return _recreate_restart_for_tile(
            tile=prepared_tiles[index],
            tile_number=index,
            input_path=input_path,
            l1_resolution=l1_resolution,
            l11_resolution=l11_resolution,
            crs=crs,
            crop_n_jobs=crop_n_jobs,
            available_mem_gib=available_mem_gib,
            chunking=chunking,
            lat_order=lat_order,
            fill_nearest_files=fill_nearest_files,
            mhm_executable=mhm_executable,
            mhm_packages=mhm_packages,
            mhm_args=mhm_args,
            restart_pattern=restart_pattern,
            require_restart=require_restart,
        )

    if int(n_jobs) == 1:
        recreated_results = [_recreate(index) for index in recreate_indices]
    else:
        recreated_results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_recreate)(index) for index in recreate_indices
        )

    updated_results = list(restart_files_by_tile)
    for index, recreated_result in zip(recreate_indices, recreated_results):
        updated_results[index] = recreated_result
    return updated_results


@log_arguments()
def create_mhm_restart_from_setup(  # noqa: PLR0913
    input_path,
    output_path,
    mask_da,
    lon_min,
    lon_max,
    lat_min,
    lat_max,
    mhm_executable,
    l1_resolution,
    l1_increment=20,
    l11_resolution=None,
    crs=None,
    n_jobs=1,
    mhm_n_jobs=None,
    crop_n_jobs=1,
    filename="*.*",
    available_mem_gib=5,
    force_header_creation=True,
    chunking=False,
    output_var=None,
    no_cropping=False,
    lat_order="decreasing",
    output_suffix=None,
    mask_var="mask",
    mhm_packages=None,
    mhm_args=None,
    restart_pattern="**/mHM_restart*.nc",
    restart_output_path=None,
    require_restart=True,
    merge=True,
    merged_restart_file=None,
    fill_nearest_files=None,
    l0_mask_files=None,
    skip_tile_creation=False,
    skip_mhm_run=False,
    recreate_restart=False,
):
    """Create restart files from a setup by tiling, running mHM, and merging output.

    The setup is split into L1-sized tiles, optionally filtered by active mask
    cells, cropped into per-tile setup folders, filled where requested, and
    then passed to mHM. The produced restart files can be collected as-is or
    merged into one final restart file covering the requested domain.
    """
    output_path = Path(output_path)
    mhm_n_jobs = n_jobs if mhm_n_jobs is None else mhm_n_jobs
    mask_ds = (
        mask_da.to_dataset(name=mask_var)
        if isinstance(mask_da, xr.DataArray)
        else mask_da
    )
    tiles = create_setup_tiles(
        lon_min_bound=lon_min,
        lon_max_bound=lon_max,
        lat_min_bound=lat_min,
        lat_max_bound=lat_max,
        l1_resolution=l1_resolution,
        l1_increment=l1_increment,
        output_path=output_path,
        lat_order=lat_order,
    )
    tiles = _filter_tiles_by_mask(tiles, mask_ds, mask_var)
    logger.info(f"Creating mHM restart files for {len(tiles)} setup tiles")
    prepared_tiles = _prepare_tiles_for_mhm(
        tiles=tiles,
        skip_tile_creation=skip_tile_creation,
        n_jobs=n_jobs,
        input_path=input_path,
        mask_ds=mask_ds,
        l1_resolution=l1_resolution,
        l11_resolution=l11_resolution,
        crs=crs,
        filename=filename,
        available_mem_gib=available_mem_gib,
        force_header_creation=force_header_creation,
        chunking=chunking,
        output_var=output_var,
        no_cropping=no_cropping,
        lat_order=lat_order,
        output_suffix=output_suffix,
        mask_var=mask_var,
        crop_n_jobs=crop_n_jobs,
        fill_nearest_files=fill_nearest_files,
        l0_mask_files=l0_mask_files,
    )
    if skip_mhm_run:
        logger.info(
            f"mHM runs disabled; collecting existing restart files for "
            f"{len(prepared_tiles)} tiles."
        )
        restart_files_by_tile = _collect_restart_files_for_tiles(
            prepared_tiles=prepared_tiles,
            restart_pattern=restart_pattern,
            require_restart=require_restart,
            n_jobs=n_jobs,
            setup_path=output_path,
            restart_output_path=restart_output_path,
        )
    else:
        logger.info(f"Runnning mHM on {len(prepared_tiles)} tiles")
        restart_files_by_tile = _run_mhm_for_tiles(
            prepared_tiles=prepared_tiles,
            mhm_n_jobs=mhm_n_jobs,
            mhm_executable=mhm_executable,
            mhm_packages=mhm_packages,
            mhm_args=mhm_args,
            restart_pattern=restart_pattern,
            require_restart=require_restart,
        )

    if recreate_restart:
        restart_files_by_tile = _recreate_missing_restart_results(
            restart_files_by_tile=restart_files_by_tile,
            prepared_tiles=prepared_tiles,
            input_path=input_path,
            l1_resolution=l1_resolution,
            l11_resolution=l11_resolution,
            crs=crs,
            crop_n_jobs=crop_n_jobs,
            available_mem_gib=available_mem_gib,
            chunking=chunking,
            lat_order=lat_order,
            fill_nearest_files=fill_nearest_files,
            mhm_executable=mhm_executable,
            mhm_packages=mhm_packages,
            mhm_args=mhm_args,
            restart_pattern=restart_pattern,
            require_restart=require_restart,
            n_jobs=n_jobs,
        )

    failed_tiles = [
        {
            "tile": result.get("tile", prepared_tiles[index].name),
            "message": result.get("message", ""),
        }
        for index, result in enumerate(restart_files_by_tile)
        if int(result.get("status", result.get("Status", 0))) == 1
    ]
    if skip_mhm_run:
        failed_tiles_file = _summarize_failed_tiles(
            failed_tiles,
            output_path,
            filename="missing_restart_files.txt",
            title="Missing restart files",
        )
    else:
        failed_tiles_file = _summarize_failed_tiles(failed_tiles, output_path)

    restart_files = [
        restart_file
        for result in restart_files_by_tile
        if int(result.get("status", result.get("Status", 0))) != 1
        for restart_file in result.get("restart_files", [])
    ]

    if restart_output_path is not None:
        restart_files = _move_restart_files(
            restart_files,
            setup_path=output_path,
            restart_output_path=restart_output_path,
        )

    merged_restart_path = None
    merged_tile_mask_path = None
    if merge:
        if merged_restart_file is None:
            merged_restart_file = output_path / "mHM_restart_001.nc"
        merged = merge_restart_files(
            restart_file_paths=restart_files,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            l1_resolution=l1_resolution,
            output_file=merged_restart_file,
            mask_ds=mask_ds,
            mask_var=mask_var,
        )
        merged_restart_path = merged_restart_file
        merged_tile_mask_path = merged.attrs.get("merged_tile_mask_file")

    logger.info(f"Created tiled mHM restart files: {restart_files}")
    logger.info(f"{len(failed_tiles)} of {len(tiles)} were missing")
    return {
        "restart_files": restart_files,
        "merged_restart_file": merged_restart_path,
        "merged_tile_mask_file": merged_tile_mask_path,
        "tiles": tiles,
        "failed_tiles": failed_tiles,
        "failed_tiles_file": failed_tiles_file,
    }


def _restart_level_from_native_dim(dim, prefix):
    """Return the restart level suffix encoded in a native mHM dimension."""
    match = re.fullmatch(rf"{prefix}(\d+)", str(dim))
    if match is None:
        return None
    return f"L{match.group(1)}"


def _native_restart_spatial_dims(dataset):
    """Map native restart levels to their column and row dimension names."""
    dims_by_level = {}
    for col_dim in dataset.dims:
        level = _restart_level_from_native_dim(col_dim, "ncols")
        if level is None:
            continue
        row_dim = f"nrows{level[1:]}"
        if row_dim in dataset.dims:
            dims_by_level[level] = (col_dim, row_dim)
    return dims_by_level


def _native_restart_level_sizes(dataset, lon_min, lon_max, lat_min, lat_max):
    """Return target dimension sizes for native restart merge allocation."""
    sizes = {}
    for level, (col_dim, row_dim) in _native_restart_spatial_dims(dataset).items():
        if level != "L1":
            logger.info(
                "Skipping native %s allocation during restart merge; final restart "
                "is written on the L1 lat/lon grid.",
                level,
            )
            continue
        cellsize = float(dataset.attrs[f"cellsize_{level}"])
        sizes[row_dim] = int((float(lon_max) - float(lon_min)) / cellsize + 0.5)
        sizes[col_dim] = int((float(lat_max) - float(lat_min)) / cellsize + 0.5)
    return sizes


def _native_dim_level(dim):
    """Return the native restart level encoded in a row or column dimension."""
    return _restart_level_from_native_dim(
        dim, "ncols"
    ) or _restart_level_from_native_dim(dim, "nrows")


def _has_non_l1_native_spatial_dim(data_array):
    """Return whether a variable uses native spatial dimensions outside L1."""
    return any(
        (level := _native_dim_level(dim)) is not None and level != "L1"
        for dim in data_array.dims
    )


def _fill_value_for_restart_var(data_array):
    """Return the fill value to use when allocating a restart variable."""
    fill_value = data_array.encoding.get(
        "_FillValue", data_array.attrs.get("_FillValue")
    )
    if fill_value is not None:
        return fill_value
    if np.issubdtype(data_array.dtype, np.floating):
        return np.nan
    return 0


def _init_native_restart_merge_dataset(template, lon_min, lon_max, lat_min, lat_max):
    """Initialize the native-dimension dataset used for stitching tile restarts."""
    dim_sizes = _native_restart_level_sizes(
        template, lon_min, lon_max, lat_min, lat_max
    )
    merged = xr.Dataset(attrs=dict(template.attrs))
    for dim, size in template.sizes.items():
        if dim not in dim_sizes:
            dim_sizes[dim] = size
    for coord_name, coord in template.coords.items():
        if all(
            dim_sizes.get(dim, coord.sizes[dim]) == coord.sizes[dim]
            for dim in coord.dims
        ):
            merged = merged.assign_coords({coord_name: coord.copy(deep=True)})

    for data_var in template.data_vars:
        data_array = template[data_var]
        if _has_non_l1_native_spatial_dim(data_array):
            logger.info(
                "Skipping %s during native stitch because it uses non-L1 native "
                "restart dimensions %s.",
                data_var,
                data_array.dims,
            )
            continue
        if not any(
            dim in dim_sizes and dim_sizes[dim] != data_array.sizes[dim]
            for dim in data_array.dims
        ):
            merged[data_var] = data_array.copy(deep=True)
            continue
        shape = [dim_sizes[dim] for dim in data_array.dims]
        merged[data_var] = xr.DataArray(
            np.full(
                shape, _fill_value_for_restart_var(data_array), dtype=data_array.dtype
            ),
            dims=data_array.dims,
            attrs=data_array.attrs.copy(),
            name=data_var,
        )
        merged[data_var].encoding.update(data_array.encoding)

    for level, (col_dim, row_dim) in _native_restart_spatial_dims(template).items():
        if col_dim not in dim_sizes or row_dim not in dim_sizes:
            continue
        merged.attrs[f"xllcorner_{level}"] = lon_min
        merged.attrs[f"yllcorner_{level}"] = lat_min
        merged.attrs[f"ncols_{level}"] = dim_sizes[col_dim]
        merged.attrs[f"nrows_{level}"] = dim_sizes[row_dim]
    return merged


def _native_restart_axis_offset(dataset, level, axis, domain_min):
    """Return the x/y offset of a tile inside the native restart domain."""
    attr = f"{'xllcorner' if axis == 'x' else 'yllcorner'}_{level}"
    cellsize = float(dataset.attrs[f"cellsize_{level}"])
    return int((float(dataset.attrs[attr]) - float(domain_min)) / cellsize + 0.5)


def _native_restart_lat_offset(dataset, level, n_rows, lat_max):
    """Return the north-to-south row offset for a native restart tile."""
    cellsize = float(dataset.attrs[f"cellsize_{level}"])
    tile_north_edge = (
        float(dataset.attrs[f"yllcorner_{level}"]) + int(n_rows) * cellsize
    )
    return int((float(lat_max) - tile_north_edge) / cellsize + 0.5)


def _native_restart_indexers(data_array, dataset, lon_min, lat_max):
    """Build target slices for writing one tile variable into the native merge."""
    indexers = {}
    for dim in data_array.dims:
        col_level = _restart_level_from_native_dim(dim, "ncols")
        row_level = _restart_level_from_native_dim(dim, "nrows")
        if row_level is not None:
            start = _native_restart_axis_offset(dataset, row_level, "x", lon_min)
        elif col_level is not None:
            start = _native_restart_lat_offset(
                dataset,
                col_level,
                data_array.sizes[dim],
                lat_max,
            )
        else:
            continue
        stop = start + data_array.sizes[dim]
        indexers[dim] = slice(start, stop)
    return indexers


def _merge_native_restart_files(restart_file_paths, lon_min, lon_max, lat_min, lat_max):
    """Stitch tile restart files on their native mHM restart dimensions."""
    logger.info("Stitching restart files with native mHM restart dimensions")
    restart_file_paths = sorted(Path(path) for path in restart_file_paths)
    if not restart_file_paths:
        msg = "The list of restart files for merging is empty."
        with ErrorLogger(logger):
            raise ValueError(msg)

    with get_xarray_ds_from_file(restart_file_paths[0]) as first_ds:
        template = first_ds.load()
    merged = _init_native_restart_merge_dataset(
        template=template,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
    )

    for counter, restart_file_path in enumerate(restart_file_paths, start=1):
        logger.info(f"Stitching {counter}/{len(restart_file_paths)} files")
        with get_xarray_ds_from_file(restart_file_path) as cur_ds_in:
            cur_ds = cur_ds_in.load()
        for data_var in cur_ds.data_vars:
            data_array = cur_ds[data_var]
            if data_var not in merged and _has_non_l1_native_spatial_dim(data_array):
                continue
            indexers = _native_restart_indexers(data_array, cur_ds, lon_min, lat_max)
            if not indexers:
                if data_var not in merged:
                    merged[data_var] = data_array.copy(deep=True)
                continue
            target = merged[data_var].isel(indexers)
            if target.shape != data_array.shape:
                msg = (
                    f"Shape mismatch for {data_var} from {restart_file_path}: "
                    f"tile shape {data_array.shape}, target shape {target.shape}, "
                    f"indexers {indexers}."
                )
                with ErrorLogger(logger):
                    raise ValueError(msg)
            merged[data_var][indexers] = data_array.data

    if "L1_domain_mask" in merged:
        mask = np.asarray(merged["L1_domain_mask"].values)
        merged.attrs["nCells_L1"] = int(np.sum(np.isfinite(mask) & (mask > 0)))
    if "L0_domain_mask" in merged:
        mask = np.asarray(merged["L0_domain_mask"].values)
        merged.attrs["nCells_L0"] = int(np.sum(np.isfinite(mask) & (mask > 0)))

    return merged


_FINAL_DIM_RENAMES = {
    "land_cover_period_out": "L1_LandCoverPeriods",
    "land_cover_period": "L1_LandCoverPeriods",
    "L1_LandCoverPeriods": "L1_LandCoverPeriods",
    "horizon_out": "L1_SoilHorizons",
    "L1_SoilHorizons": "L1_SoilHorizons",
    "month_of_year": "L1_LAITimesteps",
    "L1_LAITimesteps": "L1_LAITimesteps",
}

_FINAL_VAR_RENAMES = {
    "land_cover_period_out_bnds": "L1_LandCoverPeriods_bnds",
    "horizon_out_bnds": "L1_SoilHorizons_bnds",
    "month_of_year_bnds": "L1_LAITimesteps_bnds",
    "L1_SealedFraction": "L1_fSealed",
    "L1_Alpha": "L1_alpha",
    "L1_DegDayInc": "L1_degDayInc",
    "L1_DegDayNoPre": "L1_degDayNoPre",
    "L1_DegDayMax": "L1_degDayMax",
    "L1_KarstLoss": "L1_karstLoss",
    "L1_Max_Canopy_Intercept": "L1_maxInter",
    "L1_FastFlow": "L1_kFastFlow",
    "L1_Kperco": "L1_kPerco",
    "L1_SlowFlow": "L1_kSlowFlow",
    "L1_SoilMoistureExponent": "L1_soilMoistExp",
    "L1_FieldCap": "L1_soilMoistFC",
    "L1_PermWiltPoint": "L1_wiltingPoint",
    "L1_SatSoilMoisture": "L1_soilMoistSat",
    "L1_Jarvis_Threshold": "L1_jarvis_thresh_c1",
    "L1_TempThresh": "L1_tempThresh",
    "L1_UnsatThreshold": "L1_unsatThresh",
    "L1_SealedThresh": "L1_sealedThresh",
}


def _cell_bounds_from_centers(values, resolution):
    """Create two-column bounds from 1D cell-center coordinates."""
    values = np.asarray(values, dtype=float)
    half = float(resolution) / 2
    return np.column_stack((values - half, values + half))


def _grid_values(lon_min, lon_max, lat_min, lat_max, resolution):
    """Return longitude and latitude center arrays for the final grid."""
    lon = _global_center_coords(lon_min, lon_max, resolution, increasing=True)
    lat_inc = _global_center_coords(lat_min, lat_max, resolution, increasing=True)
    return lon, lat_inc[::-1], lat_inc


def _final_dim_name(dim):
    """Map native or intermediate dimension names to final restart names."""
    if _restart_level_from_native_dim(dim, "nrows") is not None:
        return "lon"
    if _restart_level_from_native_dim(dim, "ncols") is not None:
        return "lat"
    return _FINAL_DIM_RENAMES.get(dim, dim)


def _finalize_spatial_array(data_array, lon, lat_desc):
    """Rename spatial dimensions and attach final lon/lat coordinates."""
    rename_dims = {
        dim: _final_dim_name(dim)
        for dim in data_array.dims
        if _final_dim_name(dim) != dim
    }
    out = data_array.rename(rename_dims)
    if "lon" in out.dims:
        out = out.assign_coords(lon=lon)
    if "lat" in out.dims:
        out = out.assign_coords(lat=lat_desc)
    if "lat" in out.dims and "lon" in out.dims:
        leading_dims = [dim for dim in out.dims if dim not in ("lat", "lon")]
        out = out.transpose(*leading_dims, "lat", "lon")
        out = out.assign_coords(lat=lat_desc, lon=lon)
    return out


def _coord_or_default(dataset, name, size, values):
    """Return an existing coordinate when compatible, otherwise a default."""
    if name in dataset.coords and dataset[name].size == size:
        return np.asarray(dataset[name].values)
    values = np.asarray(values)
    if values.size == size:
        return values
    return np.arange(size)


def _bounds_or_default(dataset, name, size, default_values):
    """Return existing bounds when compatible, otherwise generated defaults."""
    if name in dataset and dataset[name].shape == (size, 2):
        return np.asarray(dataset[name].values)
    default_values = np.asarray(default_values)
    if default_values.shape == (size, 2):
        return default_values
    if name == "L1_SoilHorizons_bnds" and size == 6:
        np.array(
            [
                [0.0, 50.0],
                [50.0, 150.0],
                [150.0, 300.0],
                [300.0, 600.0],
                [600.0, 1000.0],
                [1000.0, 2000.0],
            ]
        )
    if name == "L1_SoilHorizons_bnds" and size == 3:
        return np.array([[0.0, 300.0], [300.0, 1000.0], [1000.0, 2000.0]])
    edges = np.arange(size + 1, dtype=float)
    return np.column_stack((edges[:-1], edges[1:]))


def _add_final_coords_and_bounds(ds, native, lon, lat, resolution):
    """Add final spatial and auxiliary coordinates with their bounds."""
    ds = ds.assign_coords(lon=("lon", lon), lat=("lat", lat))
    ds["lon_bnds"] = (("lon", "bnds"), _cell_bounds_from_centers(lon, resolution))
    ds["lat_bnds"] = (("lat", "bnds"), _cell_bounds_from_centers(lat, resolution))
    period_size = ds.sizes.get("L1_LandCoverPeriods", 1)
    horizon_size = ds.sizes.get("L1_SoilHorizons", 6)
    lai_size = ds.sizes.get("L1_LAITimesteps", 12)
    ds = ds.assign_coords(
        L1_LandCoverPeriods=(
            "L1_LandCoverPeriods",
            _coord_or_default(native, "land_cover_period_out", period_size, [2000]),
        ),
        L1_SoilHorizons=(
            "L1_SoilHorizons",
            _coord_or_default(
                native, "horizon_out", horizon_size, [50, 150, 300, 600, 1000, 2000]
            ),
        ),
        L1_LAITimesteps=(
            "L1_LAITimesteps",
            _coord_or_default(native, "month_of_year", lai_size, np.arange(lai_size)),
        ),
    )
    ds["L1_LandCoverPeriods_bnds"] = (
        ("L1_LandCoverPeriods", "bnds"),
        _bounds_or_default(
            native,
            "land_cover_period_out_bnds",
            period_size,
            np.tile(np.array([[1900, 2099]], dtype=np.int64), (period_size, 1)),
        ),
    )
    ds["L1_SoilHorizons_bnds"] = (
        ("L1_SoilHorizons", "bnds"),
        _bounds_or_default(
            native,
            "horizon_out_bnds",
            horizon_size,
            np.array(
                [
                    [0.0, 50.0],
                    [50.0, 150.0],
                    [150.0, 300.0],
                    [300.0, 600.0],
                    [600.0, 1000.0],
                    [1000.0, 2000.0],
                ]
            ),
        ),
    )
    ds["L1_LAITimesteps_bnds"] = (
        ("L1_LAITimesteps", "bnds"),
        _bounds_or_default(
            native,
            "month_of_year_bnds",
            lai_size,
            np.array(
                list(zip(range(lai_size), range(1, lai_size + 1))), dtype=np.int64
            ),
        ),
    )
    return ds


def _apply_final_attrs(ds, lon_min, lat_min, resolution, n_cells):
    """Attach CF-style coordinate metadata and mHM restart grid attributes."""
    coord_attrs = {
        "lon": {
            "bounds": "lon_bnds",
            "units": "degrees_east",
            "long_name": "longitude",
            "standard_name": "longitude",
            "axis": "X",
            "missing_value": np.nan,
        },
        "lat": {
            "bounds": "lat_bnds",
            "units": "degrees_north",
            "long_name": "latitude",
            "standard_name": "latitude",
            "axis": "Y",
            "missing_value": np.nan,
        },
        "L1_LandCoverPeriods": {
            "bounds": "L1_LandCoverPeriods_bnds",
            "standard_name": "time period",
            "long_name": "time period",
            "units": "years",
            "positive": "up",
            "axis": "T",
            "missing_value": np.nan,
        },
        "L1_SoilHorizons": {
            "bounds": "L1_SoilHorizons_bnds",
            "units": "mm",
            "positive": "down",
            "long_name": "depth",
            "standard_name": "depth",
            "axis": "Z",
            "missing_value": np.nan,
        },
        "L1_LAITimesteps": {
            "bounds": "L1_LAITimesteps_bnds",
            "units": "month of year",
            "long_name": "time: means within months",
            "standard_name": "month of year",
            "axis": "T",
            "missing_value": np.nan,
        },
    }
    for coord, attrs in coord_attrs.items():
        if coord in ds.coords:
            ds[coord].attrs.update(attrs)
    n_lon = ds.sizes["lon"]
    n_lat = ds.sizes["lat"]
    xll = int(lon_min) if lon_min == int(lon_min) else float(lon_min)
    yll = int(lat_min) if lat_min == int(lat_min) else float(lat_min)
    ds.attrs.update(
        {
            "xllcorner_L1": xll,
            "yllcorner_L1": yll,
            "nrows_L1": int(n_lon),
            "ncols_L1": int(n_lat),
            "cellsize_L1": float(resolution),
            "nCells_L1": int(n_cells),
            "xllcorner_L0": xll,
            "yllcorner_L0": yll,
            "nrows_L0": int(n_lon),
            "ncols_L0": int(n_lat),
            "cellsize_L0": float(resolution),
            "nCells_L0": int(n_cells),
            "institution": "Helmholtz-Centre for Environmental Research - UFZ, Leipzig, Germany",
            "creator": "Robert Schweppe",
            "contact": "stephan.thober@ufz.de",
            "coordinates": "lat_bnds lon_bnds",
        }
    )
    return ds


def _final_mask(mask_ds, mask_var, lon, lat):
    """Return the final active-domain mask on the merged restart grid."""
    target_lon = xr.DataArray(lon, dims=("lon",), name="lon")
    target_lat = xr.DataArray(lat, dims=("lat",), name="lat")
    mask = _combined_restart_mask(mask_ds, mask_var, target_lon, target_lat)
    if mask is None:
        mask = xr.DataArray(
            np.ones((len(lat), len(lon)), dtype=bool),
            dims=("lat", "lon"),
            coords={"lat": lat, "lon": lon},
        )
    return mask.transpose("lat", "lon")


def _add_domain_variables(ds, active_mask, resolution):
    """Add mHM domain mask, coordinates, and cell-area variables."""
    lat_2d = np.broadcast_to(ds["lat"].values[:, None], active_mask.shape)
    lon_2d = np.broadcast_to(ds["lon"].values[None, :], active_mask.shape)
    for level in ("L1", "L0"):
        ds[f"{level}_domain_mask"] = (
            ("lat", "lon"),
            active_mask.values.astype(float),
        )
        ds[f"{level}_domain_lat"] = (("lat", "lon"), lat_2d.astype(float))
        ds[f"{level}_domain_lon"] = (("lat", "lon"), lon_2d.astype(float))
        ds[f"{level}_domain_cellarea"] = (
            ("lat", "lon"),
            np.full(active_mask.shape, float(resolution) ** 2, dtype=float),
        )
    return ds


def _mask_final_spatial_vars(ds, active_mask):
    """Mask final spatial variables outside the active restart domain."""
    for var in list(ds.data_vars):
        if "lat" not in ds[var].dims or "lon" not in ds[var].dims:
            continue
        if var.endswith(("_lat", "_lon", "_cellarea", "_domain_mask")):
            continue
        ds[var] = ds[var].where(active_mask)
    if "L1_maxInter" in ds:
        ds["L1_maxInter"] = ds["L1_maxInter"].where(
            ~(np.isnan(ds["L1_maxInter"]) & active_mask),
            0,
        )
    return ds


def _final_restart_encoding(ds):
    """Build NetCDF encoding for final restart output variables."""
    encoding = {}
    for name in ds.variables:
        if not np.issubdtype(ds[name].dtype, np.floating):
            continue
        encoding[name] = {
            "dtype": "float64",
            "_FillValue": np.nan,
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
        }
    return encoding


def _drop_fill_value_attrs(ds):
    """Remove fill-value attrs before writing values through encoding."""
    ds = ds.copy(deep=False)
    for name in ds.variables:
        ds[name].attrs.pop("_FillValue", None)
    return ds


def _write_final_restart(ds, output_file):
    """Write the final restart dataset to NetCDF."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.is_file():
        output_file.unlink()
    _drop_fill_value_attrs(ds).to_netcdf(
        output_file,
        engine="netcdf4",
        encoding=_final_restart_encoding(ds),
    )


def _tile_mask_output_file(output_file):
    """Return the sidecar output path for the merged tile mask."""
    output_file = Path(output_file)
    return output_file.parent / f"{output_file.stem}_tile_mask{output_file.suffix}"


def _write_merged_tile_mask(restart_file_paths, output_file, mask_var, lon, lat):
    """Write a merged tile-coverage mask beside the final restart file."""
    masks = []
    for restart_file in restart_file_paths:
        mask_file = _tile_mask_file_for_restart(restart_file)
        if mask_file is None:
            continue
        with get_xarray_ds_from_file(mask_file) as mask_ds:
            masks.append(mask_ds.load())
    if not masks:
        return None
    combined = _final_mask(masks, mask_var, lon, lat).astype(float)
    var_name = mask_var or "tile_mask"
    mask_ds = combined.to_dataset(name=var_name)
    mask_output = _tile_mask_output_file(output_file)
    write_xarray_to_file(
        mask_ds,
        mask_output,
        encoding={var_name: {"_FillValue": np.nan, "zlib": True, "complevel": 4}},
    )
    return mask_output


def _convert_native_restart_to_cf(
    native, lon_min, lon_max, lat_min, lat_max, l1_resolution, mask_ds, mask_var
):
    """Convert a native stitched restart dataset to the final CF-style layout."""
    lon, lat_desc, _lat_inc = _grid_values(
        lon_min, lon_max, lat_min, lat_max, l1_resolution
    )
    final = xr.Dataset(coords={"lon": ("lon", lon), "lat": ("lat", lat_desc)})
    for data_var in native.data_vars:
        if data_var.startswith(("L1_domain_", "L0_domain_")):
            continue
        data_array = native[data_var]
        finalized = _finalize_spatial_array(data_array, lon, lat_desc)
        if any(
            _restart_level_from_native_dim(dim, "ncols") is not None
            or _restart_level_from_native_dim(dim, "nrows") is not None
            for dim in finalized.dims
        ):
            logger.debug("Skipping unsupported final restart variable %s.", data_var)
            continue
        final[_FINAL_VAR_RENAMES.get(data_var, data_var)] = finalized

    final = _add_final_coords_and_bounds(final, native, lon, lat_desc, l1_resolution)
    active_mask = _final_mask(mask_ds, mask_var, lon, lat_desc)
    final = _add_domain_variables(final, active_mask, l1_resolution)
    if "L1_fAsp" not in final:
        final["L1_fAsp"] = (("lat", "lon"), np.ones(active_mask.shape, dtype=float))
    if "L1_degDay" not in final:
        final["L1_degDay"] = (("lat", "lon"), np.ones(active_mask.shape, dtype=float))
    final = _mask_final_spatial_vars(final, active_mask)
    return _apply_final_attrs(
        final,
        lon_min=lon_min,
        lat_min=lat_min,
        resolution=l1_resolution,
        n_cells=int(np.sum(active_mask.values)),
    )


def merge_restart_files(
    restart_file_paths,
    lon_min,
    lon_max,
    lat_min,
    lat_max,
    l1_resolution,
    output_file=None,
    mask_ds=None,
    mask_var="mask",
):
    """Merge mHM tile restart files into one final CF-style restart file."""
    logger.info("Merging restart files to final CF lat/lon restart")
    restart_file_paths = [Path(path) for path in restart_file_paths]
    native = _merge_native_restart_files(
        restart_file_paths=restart_file_paths,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
    )
    final = _convert_native_restart_to_cf(
        native=native,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        l1_resolution=l1_resolution,
        mask_ds=mask_ds,
        mask_var=mask_var,
    )
    if output_file is not None:
        output_file = Path(output_file)
        _write_final_restart(final, output_file)
        tile_mask_file = _write_merged_tile_mask(
            restart_file_paths=restart_file_paths,
            output_file=output_file,
            mask_var=mask_var,
            lon=final["lon"].values,
            lat=final["lat"].values,
        )
        if tile_mask_file is not None:
            final.attrs["merged_tile_mask_file"] = str(tile_mask_file)
    logger.info("Merging restart files done")
    return final


_merge_restart_files = merge_restart_files
