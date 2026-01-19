"""Create the mHM restart file."""

import itertools
import logging
import re
import shutil
from pathlib import Path
from subprocess import PIPE, Popen, TimeoutExpired

import numpy as np
import xarray as xr
from crick import TDigest
from joblib import Parallel, delayed

from mhm_tools.common.file_handler import get_xarray_ds_from_file, write_xarray_to_file
from mhm_tools.common.logger import ErrorLogger, log_arguments
from mhm_tools.common.xarray_utils import get_coord_key

logger = logging.getLogger(__name__)


class MorphFiles:
    """A class representing a collection of morphological files.

    Attributes
    ----------
        land_cover (Path): The path to the land cover file.
        bulk_density (Path): The path to the bulk density file.
        sand_content (Path): The path to the sand content file.
        clay_content (Path): The path to the clay content file.
        slope (Path): The path to the slope file.
        slope_emp (Path): The path to the slope_emp file.
        lai (Path): The path to the leaf area index file.
        aspect (Path): The path to the aspect file.
        geology (Path): The path to the geology file.
    """

    def __init__(
        self,
        filepath=None,
        land_cover=None,
        bulk_density=None,
        sand_content=None,
        clay_content=None,
        slope=None,
        slope_emp=None,
        lai=None,
        aspect=None,
        geology=None,
    ):
        self.land_cover = land_cover
        self.bulk_density = bulk_density
        self.sand_content = sand_content
        self.clay_content = clay_content
        self.slope = slope
        self.slope_emp = slope_emp
        self.lai = lai
        self.aspect = aspect
        self.geology = geology

        if filepath is not None:
            self.read_files(filepath)

    def read_files(self, filepath: Path, overwrite=False):
        """Read morph files and assign them to attributes.

        Scans `filepath` for known variables and sets the corresponding instance
        attributes. Existing values are preserved unless `overwrite=True`.

        Parameters
        ----------
        filepath : Path
            Directory containing the files.
        overwrite : bool, default False
            If False, existing attribute values will not be overwritten.
        """
        member_key_synonyms = {
            "bulk_density": ["BLDFIE"],
            "sand_content": ["SNDPPT"],
            "clay_content": ["CLYPPT"],
            "lai": ["LAI"],
            "facc": ["flow_accumulation"],
        }
        if type(filepath) is not Path:
            filepath = Path(filepath)
        logger.info(f"reading morph files from {filepath}")
        for key in self.__dict__:
            logger.debug(f"Looking for {key} file(s)")
            if not overwrite and self.__dict__.get(key, None) is not None:
                continue
            key_files = list(filepath.glob(f"*{key}*.nc"))
            if key == "slope":
                key_files = [file for file in key_files if "emp" not in str(file.name)]
            if len(key_files) == 0:
                if key not in member_key_synonyms:
                    continue  # should raise an error
                for synonym in member_key_synonyms[key]:
                    key_files = list(filepath.glob(f"{synonym}*.nc"))
                    if len(key_files) != 0:
                        break
            logger.debug(f"Found {len(key_files)} {key} files: {key_files}")
            if len(key_files) != 1:
                self.__dict__[key] = [f for f in key_files if f.is_file()]
            else:
                self.__dict__[key] = key_files[0]  # if key_files[0].is_file() else None
            logger.debug(f"Found {key} file(s): {self.__dict__[key]}")
            if self.__dict__[key] is None or not self.__dict__[key]:
                logger.warning(f"Could not find {key} file in {filepath}")
        logger.debug(self.get_files_as_dict())

    def get_file(self, key):
        """Retrieve the file path associated with the given name of the member variable.

        Parameters
        ----------
            key (str): The member-variable name to retrieve the filepath for.

        Returns
        -------
            object: The filepath or list of filepaths associated with the given key, or None if the key is not found.
        """
        return self.__dict__.get(key, None)

    def get_files_as_list(self):
        """Return a list of all files in the object's attributes.

        This method iterates over all attributes of the object and checks if they are lists.
        If an attribute is a list, its elements are added to the file_list.
        If an attribute is not a list, it is directly appended to the file_list.

        Returns
        -------
            list: A list of all files in the object's attributes.
        """
        file_list = []
        for value in self.__dict__.values():
            if isinstance(value, list):
                file_list.extend(value)
            else:
                file_list.append(value)
        return file_list

    def get_files_as_dict(self):
        """Return a dictionary of all files in the object's attributes.

        Returns
        -------
            dict: A dictionary containing all files in the object's attributes.
        """
        return self.__dict__


class LatLon:
    """Represents a latitude-longitude coordinate system.

    Attributes
    ----------
        lat_min (float): The minimum latitude value.
        lon_min (float): The minimum longitude value.
        lat_max (float): The maximum latitude value.
        lon_max (float): The maximum longitude value.
        resolution (float): The resolution of the coordinate system.
    """

    def __init__(
        self,
        lat_min=None,
        lon_min=None,
        lat_max=None,
        lon_max=None,
        resolution=None,
        mask=None,
    ):
        self.lat_min = lat_min
        self.lon_min = lon_min
        self.lat_max = lat_max
        self.lon_max = lon_max
        self.resolution = resolution
        self.mask = mask

    def get_n_lat(self):
        """Return the number of latitude points for the range and resolution.

        Returns
        -------
            int: The number of latitude points.
        """
        # +0.5 to round to nearest integer
        return int((self.lat_max - self.lat_min) / self.resolution + 0.5)

    def get_n_lon(self):
        """Return the number of longitude points for the range and resolution.

        Returns
        -------
        int
            Number of longitude points.
        """
        # +0.5 to round to nearest integer
        return int((self.lon_max - self.lon_min) / self.resolution + 0.5)

    def is_fully_defined(self):
        """Check if all the required attributes are fully defined.

        Returns
        -------
            bool: True if all the required attributes are not None, False otherwise.
        """
        return all(
            [
                self.lat_min is not None,
                self.lon_min is not None,
                self.lat_max is not None,
                self.lon_max is not None,
                self.resolution is not None,
            ]
        )


class Grid:
    """Represents a geographical area for wich morphological data exists.

    This grid is used to run the mPR model. It does not need to contain a whole catchment,
    but can be a subset of it or multiple catchments at once.

    Attributes
    ----------
        file_path (Path): The file path of the grid.
        name (str): The name of the grid.
        latlon_file (str): The file path of the latlon file.
        l0 (LatLon): The lower-left corner of the grid.
        l1 (LatLon): The upper-right corner of the grid.
    """

    def __init__(
        self,
        file_path: Path,
        name=None,
        latlon_file=None,
        l0: LatLon = None,
        l1: LatLon = None,
        land_mask_file=None,
    ):
        file_path = Path(file_path)
        self.morph_files = MorphFiles(filepath=file_path)
        self.name = name
        self.path = file_path
        self.l0 = l0
        self.l1 = l1
        self.restart_file = None
        self.namelist_file = None
        self.land_mask_file = land_mask_file

        if (
            self.l0 is None
            or not self.l0.is_fully_defined()
            or self.l1 is None
            or not self.l1.is_fully_defined()
        ) and latlon_file is not None:
            self.read_latlon(latlon_file)

    def migrate_grid_using_systemlink(self, new_path):
        """Mirgrates the file path by creating a new path and system linking all files there."""
        logger.info(f"Creating system links in {new_path} for all files in {self.path}")
        new_path = Path(new_path)
        new_path.mkdir(parents=True, exist_ok=True)
        for file in self.path.glob("*.*"):
            link_loc = new_path / file.name
            if link_loc.exists():
                link_loc.unlink()
            link_loc.symlink_to(file)
        self.path = new_path
        self.morph_files = MorphFiles(self.path)

    def read_latlon(self, latlon_file: Path):
        """Longitude and Latidute reader.

        Read the latlon file and sets the lower-left (l0) and upper-right
        (l1) corners of the grid as well as the resolution.

        Parameters
        ----------
        latlon_file : Path
            Path to the latlon NetCDF file.

        Returns
        -------
        None
        """
        with get_xarray_ds_from_file(latlon_file) as ds:
            x0 = ds["xc_l0"].to_numpy()
            y0 = ds["yc_l0"].to_numpy()
            self.l0 = LatLon(
                lon_min=x0.min(),
                lon_max=x0.max(),
                lat_min=y0.min(),
                lat_max=y0.max(),
                resolution=x0[1] - x0[0],
            )
            x1 = ds["xc"].to_numpy()
            y1 = ds["yc"].to_numpy()
            self.l1 = LatLon(
                lon_min=x1.min(),
                lon_max=x1.max(),
                lat_min=y1.min(),
                lat_max=y1.max(),
                resolution=x1[1] - x1[0],
            )

    def read_morph_files(self):
        """Read the morph files from the specified path.

        This method uses the `read_files` function from the `MorphFiles` object to read the morph files
        located at the specified path.

        Args:
            self: The instance of the class.

        Returns
        -------
            None
        """
        self.morph_files.read_files(self.path)


class MPRRunner:
    """Class for running the mPR executable."""

    def __init__(self, mpr_executable, mpr_packages=None, mpr_parameter_file=None):
        """Initialize the MPRRunner object.

        Args:
            mpr_executable (str): Path to the mPR executable.
            mpr_packages (str, optional): Packages to load before running mPR. Defaults to None.
            mpr_parameter_file (str, optional): Path to the mPR parameter file. Defaults to None.
        """
        self.mpr_executable = mpr_executable
        self.mpr_packages = mpr_packages
        self.mpr_parameter_file = mpr_parameter_file

    def run_mpr(self, grid: Grid):
        """Run the mPR executable with the given namelist and parameter file.

        Args:
            grid (Grid): The grid object containing the namelist file.

        Raises
        ------
            RuntimeError: If mPR fails with an error.
            TimeoutExpired: If mPR execution times out.
        """
        command = (
            f"module load {self.mpr_packages} \n"
            if self.mpr_packages is not None
            else ""
        )
        command += f"""{self.mpr_executable} -c {grid.namelist_file}"""
        if self.mpr_parameter_file is not None:
            command += f" -p {self.mpr_parameter_file}"
        logger.info(f"Running mPR with: {command}")
        p = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
        try:
            data, error_data = p.communicate()
            logger.debug(data)
            if error_data:
                grid.restart_file = None
                msg = f"MPR failed with STDERR {error_data} for {grid.name} and command {command}"
                with ErrorLogger(logger):
                    raise RuntimeError(msg)
        except TimeoutExpired as err:
            p.kill()
            msg = (
                f"MPR failed with TimeoutExpired for {grid.name} and command {command}"
            )
            with ErrorLogger(logger):
                raise TimeoutExpired(msg) from err
        except RuntimeError as rte:
            msg = f"MPR failed for {grid} and command {command}"
            with ErrorLogger(logger):
                raise RuntimeError(msg) from rte


class MHMRestartFile:
    """A class for creating a restart file for the MHM model.

    This class provides methods to split the grid (if necessary), write the grid namelist,
    call the mpr executable, merge the restart files (if applicable), and delete temporary files (if specified).

    Args:
        input_file_path (Path): The path to the input file.
        nml_template (Path): The path to the namelist template file.
        output_path (Path): The path to the output directory.
        work_path (Path): The path to the working directory.
        l0 (LatLon): The LatLon object representing the high resolution grid.
        l1 (LatLon): The LatLon object representing the low resolution grid.
        mpr (MPRRunner): The MPRRunner object for executing the mpr executable.
        increment_l1 (int, optional): The increment for the low resolution grid. Defaults to 2.
        run_on_whole_domain (bool, optional): Whether to run on the whole domain. Defaults to False.
        use_split_grids (bool, optional): Whether to use split grids. Defaults to False.
        ncpus (int, optional): The number of CPUs to use for parallelization. Defaults to 1.
        clean_temp_files (bool, optional): Whether to clean temporary files. Defaults to False.
        merge (bool, optional): Whether to merge the restart files. Defaults to True.
        merge_only (bool, optional): Whether to only merge the restart files. Defaults to False.

    Attributes
    ----------
        nml_template (Path): The path to the namelist template file.
        output_path (Path): The path to the output directory.
        grid (Grid): The Grid object representing the whole grid.
        subgrids (list): The list of Grid objects representing the subgrids.
        ncpus (int): The number of CPUs to use for parallelization.
        run_on_whole_domain (bool): Whether to run on the whole domain.
        use_split_grids (bool): Whether to use split grids.
        merge_grid (bool): Whether to merge the restart files.
        merge_only (bool): Whether to only merge the restart files.
        clean_temp_files (bool): Whether to clean temporary files.
        mpr (MPRRunner): The MPRRunner object for executing the mpr executable.
        work_dir (str): The working directory.
        increment_l1 (int): The increment for the low resolution grid.
        increment_l0 (int): The increment for the high resolution grid.

    Methods
    -------
        _create_namelist(replace_dict, out_file_path): Create a namelist file by replacing placeholders in the template.
        _write_grid_namelist(grid): Write the grid namelist file.
        _create_latlon(lon_min, lat_min): Create a LatLon object from the given lon_min and lat_min.
        _read_subgrids_from_files(): Read the subgrids from the files on disk.
        _split_grid(): Split the grid into subgrids and write them to disk.
        _split_file(name, file_path): Split a file into subgrids and write them to disk.
        _delete_temp_files(): Delete temporary files.
        _merge_restart_files(): Merge the restart files.
        _correct_restart_file(ds): Correct the restart file parameters.
    """

    def __init__(
        self,
        grid: Grid,
        nml_template: Path,
        output_path: Path,
        mpr: MPRRunner,
        work_path=None,
        increment_l1=2,
        ncpus=1,
        run_on_whole_domain=False,
        use_split_grids=False,
        merge=True,
        merge_only=False,
        clean_temp_files=False,
    ):
        logger.debug(f"Creating MHMRestartFile object with {locals()}")
        self.nml_template = Path(nml_template)
        self.output_path = Path(output_path)
        self.work_path = Path(work_path) if work_path is not None else self.output_path
        self.grid = grid
        self.subgrids = []  # list of grid objects
        self.ncpus = ncpus
        self.run_on_whole_domain = run_on_whole_domain
        self.use_split_grids = use_split_grids
        self.merge_grid = merge
        self.merge_only = merge_only
        self.clean_temp_files = clean_temp_files
        self.mpr = mpr
        self.work_dir = "."
        self.increment_l1 = increment_l1
        self.increment_l0 = (
            int(self.increment_l1 * self.grid.l1.resolution / self.grid.l0.resolution)
            if self.increment_l1 is not None
            else None
        )

    def _create_namelist(self, replace_dict, out_file_path):
        if type(out_file_path) is not Path:
            out_file_path = Path(out_file_path)
        with self.nml_template.open("r") as f:
            nml_data = f.read()
        for replace_key, replace_value in replace_dict.items():
            nml_data = nml_data.replace(str(replace_key), str(replace_value))
        if out_file_path.is_file():
            out_file_path.unlink()
        with out_file_path.open("w") as f:
            f.write(nml_data)
        return out_file_path

    def _write_grid_namelist(self, grid: Grid):
        logger.debug(f"Writing namelist for {grid.name}")
        logger.debug(f"lon_min: {grid.l0.lon_min}, lon_max: {grid.l0.lon_max}")
        replace_dict = {
            "${slicei_j}": grid.name,
            "${output_file}": grid.path / f"output_{grid.name}.nc",
            "${lon_high_start}": f"{grid.l0.lon_min + grid.l0.resolution / 2:.3f}",  # changes to center of cell coordinates
            "${lon_high_res}": f"{grid.l0.resolution:.3f}",
            "${lon_high_n}": f"{grid.l0.get_n_lon()}",
            "${lat_high_start}": f"{grid.l0.lat_min + grid.l0.resolution / 2:.3f}",  # changes to center of cell coordinates
            "${lat_high_res}": f"{grid.l0.resolution:.3f}",
            "${lat_high_n}": f"{grid.l0.get_n_lat()}",
            "${lon_low_start}": f"{grid.l1.lon_min + grid.l1.resolution / 2:.3f}",  # changes to center of cell coordinates
            "${lon_low_res}": f"{grid.l1.resolution:.2f}",
            "${lon_low_n}": f"{grid.l1.get_n_lon()}",
            "${lat_low_start}": f"{grid.l1.lat_min + grid.l1.resolution / 2:.3f}",  # changes to center of cell coordinates
            "${lat_low_res}": f"{grid.l1.resolution:.2f}",
            "${lat_low_n}": f"{grid.l1.get_n_lat()}",
            "${bulk_density}": grid.morph_files.bulk_density,
            "${sand_content}": grid.morph_files.sand_content,
            "${clay_content}": grid.morph_files.clay_content,
            "${slope}": grid.morph_files.slope,
            "${slope_emp}": grid.morph_files.slope_emp,
            "${lai}": grid.morph_files.lai,
            "${aspect}": grid.morph_files.aspect,
            "${geology}": grid.morph_files.geology,
            # "${dem}": grid.morph_files.dem,
            # "${facc}": grid.morph_files.facc,
            # "${karstic}": "0",
            "${land_cover}": grid.morph_files.land_cover,  # this should be a list but the template only has one
        }
        logger.debug(replace_dict)
        grid.restart_file = grid.path / f"output_{grid.name}.nc"
        grid.namelist_file = self._create_namelist(
            replace_dict, grid.path / f"mpr_{grid.name}.nml"
        )
        logger.info(f"Wrote namelist for {grid.name} to {grid.namelist_file}")
        return grid

    def _create_latlon(self, lon_min, lat_min):
        """Create l0 and a l1 resolution LatLon object from a given lon_min and lat_min."""
        lon_max = lon_min + self.increment_l1 * self.grid.l1.resolution
        lon_max = min(lon_max, self.grid.l1.lon_max)
        lat_max = lat_min + self.increment_l1 * self.grid.l1.resolution
        lat_max = min(lat_max, self.grid.l1.lat_max)

        # grid saved in llc coordinates
        l0 = LatLon(  # this is the high resolution grid
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            resolution=self.grid.l0.resolution,
        )
        l1 = LatLon(  # this is the low resolution grid
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            resolution=self.grid.l1.resolution,
        )
        return l0, l1

    def _read_subgrids_from_files(self):
        """Read the subgrids from the files on disk."""
        for i, lon_min in enumerate(
            np.arange(
                self.grid.l1.lon_min,
                self.grid.l1.lon_max,
                self.grid.l1.resolution * self.increment_l1,
            )
        ):
            for j, lat_min in enumerate(
                np.arange(
                    self.grid.l1.lat_min,
                    self.grid.l1.lat_max,
                    self.grid.l1.resolution * self.increment_l1,
                )
            ):
                l0, l1 = self._create_latlon(lon_min, lat_min)
                subgrid_path = self.work_path / f"slice_{i}_{j}"
                logger.debug(f"Reading subgrid {subgrid_path}")
                logger.debug(
                    f"l0: {l0.lon_min}, {l0.lon_max}, {l0.lat_min}, {l0.lat_max}"
                )
                if not subgrid_path.is_dir():
                    logger.error(f"Subgrid {subgrid_path} not found")
                    continue
                grid = Grid(
                    file_path=subgrid_path, name=subgrid_path.name, l0=l0, l1=l1
                )
                grid.read_morph_files()
                self.subgrids.append(grid)

    def _split_grid(self):  # has do addapted to different file types not just .nc
        """Split the grid into subgrids and write them to disk.

        Subgrids are subsets of the original grid with a size of
        increment x increment grid cells.
        """
        logger.info("Splitting grid")
        if self.increment_l0 is None:
            error_message = "Increment for splitting grids is not set"
            with ErrorLogger(logger):
                raise ValueError(error_message)
        for name, file_path in self.grid.morph_files.get_files_as_dict().items():
            if file_path is not None and file_path:
                sub_grid_paths = self._split_file(name, file_path)
            else:
                logger.warning(
                    f"There is no file for {name} that could be split at path {file_path}"
                )
        logger.debug("Creating subgrids")
        self.subgrids = [Grid(file_path=k, **v) for k, v in sub_grid_paths.items()]
        logger.debug("Splitting grid done")

    def _split_cell(self, ds, file_path, i, lon_min, j, lat_min):
        out_dir = Path(self.work_path) / f"slice_{i}_{j}"
        if not out_dir.exists():
            logger.debug(f"Creating {out_dir}")
            out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{file_path.stem}.nc"

        if out_path.is_file():
            out_path.unlink()

        l0, l1 = self._create_latlon(lon_min, lat_min)
        lon_slice = slice(l1.lon_min, l1.lon_max)
        lat_slice = slice(l1.lat_max, l1.lat_min)
        logger.debug(f"slice_{i}_{j}: lat {lat_slice}; lon {lon_slice}")
        ds_cut = ds.sel(
            longitude=lon_slice,
            latitude=lat_slice,
        )
        if "slope" in file_path.name or "geology" in file_path.name:
            ds_cut = ds_cut.sortby("latitude")
        try:
            write_xarray_to_file(ds_cut, out_path)
            logger.debug(f"Written {out_path}")
        except Exception as e:
            logger.error(f"Failed to write {out_path} with {e}")
            logger.debug(f"{l1.lon_min}, {l1.lon_max}, {l1.lat_min}, {l1.lat_max}")
            logger.debug(ds_cut["latitude"].values)
            return None
        return {
            out_dir: {
                "l0": l0,
                "l1": l1,
                "name": f"slice_{i}_{j}",
            }
        }

    def _split_file(self, name, file_path):
        if isinstance(file_path, list) and len(file_path) == 1:
            file_path = file_path[0]
        elif isinstance(file_path, list) and len(file_path) > 1:
            msg = f"There are multiple files with ambigous names in the setup. Please remove all but one of {file_path}"
            with ErrorLogger(logger):
                raise ValueError(msg)
        if not file_path or file_path is None or not file_path.is_file():
            logger.error(f"No file path provided for {name}")
            return None
        if file_path.suffix != ".nc":
            logger.error(f"File {file_path} is not a netCDF file")
            return None
        logger.debug(f"Splitting {file_path}")
        with get_xarray_ds_from_file(file_path) as ds:
            lon_range = np.arange(
                self.grid.l1.lon_min,
                self.grid.l1.lon_max,
                self.grid.l1.resolution * self.increment_l1,
            )
            lat_range = np.arange(
                self.grid.l1.lat_min,
                self.grid.l1.lat_max,
                self.grid.l1.resolution * self.increment_l1,
            )
            iter_product = itertools.product(enumerate(lon_range), enumerate(lat_range))
            sub_grid_paths = Parallel(n_jobs=self.ncpus, backend="loky")(
                delayed(self._split_cell)(ds, file_path, i, lon_min, j, lat_min)
                for (i, lon_min), (j, lat_min) in iter_product
            )
        logger.debug(f"Splitting {file_path} done")
        return {k: v for d in sub_grid_paths for k, v in d.items()}

    def _order_dims(self, dims):
        """Order the dimensions of the data variable.

        Dimensions are sorted by a fixed priority: lower weights come first; any
        unknown dimension defaults to weight 5.

        Parameters
        ----------
        dims : Iterable[str]
            Names of the dimensions.

        Returns
        -------
        list[str]
            Dimensions sorted by priority.
        """
        weight_dims = {
            "land_cover_period_out": 0,
            "month_of_year": 1,
            "horizons": 2,
            "horizon_all": 2,
            "horizon_out": 2,
            "horizon_till": 2,
            "horizon_notill": 2,
            "latitude": 3,
            "lat_out": 3,
            "longitude": 4,
            "lon_out": 4,
        }
        return sorted(dims, key=lambda x: weight_dims.get(x, 5))

    def _merge_restart_files(self):  # noqa: PLR0912, PLR0915
        logger.info("Merging restart files")

        # 1. create an empty file for the whole grid
        ds_whole = xr.Dataset()
        ds_whole["lon_out"] = np.arange(
            self.grid.l1.lon_min + self.grid.l1.resolution / 2,
            self.grid.l1.lon_max
            + self.grid.l1.resolution / 2,  # + since arange omits the last value
            self.grid.l1.resolution,
        )
        ds_whole["lat_out"] = np.arange(
            self.grid.l1.lat_min + self.grid.l1.resolution / 2,
            self.grid.l1.lat_max
            + self.grid.l1.resolution / 2,  # + since arange omits the last value
            self.grid.l1.resolution,
        )
        # Boolean flags: True→1 (trim), False→0 (keep)
        lon_trim = int(ds_whole["lon_out"][-1] >= self.grid.l1.lon_max)
        lat_trim = int(ds_whole["lat_out"][-1] >= self.grid.l1.lat_max)

        # New lengths = old length minus {0 or 1}
        ds_whole["lon_out"] = ds_whole["lon_out"][: len(ds_whole["lon_out"]) - lon_trim]
        ds_whole["lat_out"] = ds_whole["lat_out"][: len(ds_whole["lat_out"]) - lat_trim]

        logger.debug("lat_out")
        logger.debug(ds_whole["lat_out"].shape)
        logger.debug("lon_out")
        logger.debug(ds_whole["lon_out"].shape)
        logger.debug(f"ds_whole: {ds_whole}")

        # 2. create all coordinates in the whole grid
        # TODO: add dimensions to comments to make it more readable

        self.grid.restart_file = self.work_path / "output_whole_grid_restart.nc"

        if self.merge_only:
            restart_file_paths = [
                file
                for dir in self.work_path.glob("slice_*")
                for file in dir.glob("output_*.nc")
            ]
        else:
            restart_file_paths = [subgrid.restart_file for subgrid in self.subgrids]
        logger.debug(f"Restart File Paths: {restart_file_paths}")
        restart_file_paths.sort()
        if not restart_file_paths:
            with ErrorLogger(logger):
                msg = "The list of restart files for merging is empty."
                if self.merge_only:
                    msg += "Try without merge_only flag."
                raise ValueError(msg)
        first_restart_file = next(iter(restart_file_paths))
        logger.info(f"Opening {first_restart_file} als reference")

        with get_xarray_ds_from_file(first_restart_file) as cur_ds:
            for coord in cur_ds.coords:
                if coord not in ds_whole.coords:
                    ds_whole[coord] = cur_ds[coord]
                # init the new bounds (e.g. horizons_out_bnds)
                data_vars = [
                    _
                    for _ in cur_ds.data_vars
                    if _ not in ds_whole.data_vars and _.endswith("_out_bnds")
                ]
                logger.debug(f"Data vars bounds: {data_vars}")
                for data_var in data_vars:
                    if "lat" in data_var:
                        ds_whole[data_var] = np.arange(
                            self.grid.l1.lat_min,
                            self.grid.l1.lat_max,
                            self.grid.l1.resolution,
                        )
                    elif "lon" in data_var:
                        ds_whole[data_var] = np.arange(
                            self.grid.l1.lon_min,
                            self.grid.l1.lon_max,
                            self.grid.l1.resolution,
                        )
                    else:
                        ds_whole[data_var] = cur_ds[data_var]
                # init the new DataArrays (set to nan)
                data_vars = [_ for _ in cur_ds.data_vars if _.startswith("L1_")]
                logger.debug(f"Data vars L1: {data_vars}")
                for data_var in data_vars:
                    dv_coords = list(cur_ds[data_var].coords)
                    for dv_coord in dv_coords:
                        if dv_coord not in ds_whole:
                            logger.info(f"Adding {dv_coord} to coordinates")
                            logger.debug(f"cur_ds[coord] {cur_ds[dv_coord]}")
                            ds_whole[dv_coord] = cur_ds[dv_coord]
                    ds_whole[data_var] = (
                        dv_coords,
                        np.full([len(ds_whole[_]) for _ in dv_coords], np.nan),
                    )

        # 3. iterate over all subgrids and merge them into the whole grid
        for counter, restart_file_path in enumerate(restart_file_paths):
            logger.info(f"Merging {counter}/{len(restart_file_paths)} files")
            logger.debug(
                f"Restart path of the subdomain restart file: {restart_file_path}"
            )
            ints = re.findall(r"\d+", str(restart_file_path))
            isel_start = int(ints[-2])
            jsel_start = int(ints[-1])
            with get_xarray_ds_from_file(restart_file_path) as cur_ds:
                logger.debug(f"Opening {restart_file_path}")
                # reverse the data vars if the latitude is decreasing
                reverse_data_vars = [
                    "L1_latitude",
                    "L1_fAsp",
                    "L1_Max_Canopy_Intercept",
                ]
                for r_data_var in reverse_data_vars:
                    if r_data_var in cur_ds:
                        index_lat = cur_ds[r_data_var].dims.index("lat_out")
                        cur_ds[r_data_var].data = np.flip(
                            cur_ds[r_data_var].data, axis=index_lat
                        )
                        logger.debug(f"{r_data_var} reversed latitudes")
                index_slice = {
                    "lon_out": slice(
                        isel_start * self.increment_l1,
                        (isel_start + 1) * self.increment_l1,
                    ),
                    "lat_out": slice(
                        jsel_start * self.increment_l1,
                        (jsel_start + 1) * self.increment_l1,
                    ),
                }
                for data_var in data_vars:
                    try:
                        if (
                            cur_ds[data_var].shape
                            != ds_whole[data_var][index_slice].shape
                        ):
                            dims = cur_ds[data_var].dims
                            ds_whole[data_var] = ds_whole[data_var].transpose(*dims)
                            if (
                                cur_ds[data_var].shape
                                != ds_whole[data_var][index_slice].shape
                            ):
                                logger.error(
                                    f"Shape mismatch could not be resolved for {data_var} in {restart_file_path}"
                                )
                                logger.debug(
                                    f"shape read in ds: {cur_ds[data_var].shape}; shape in ds_whole: {ds_whole[data_var][index_slice].shape}"
                                )
                                logger.debug(
                                    f"index_slice: lon={ds_whole[index_slice]['lon_out'].data[0]:.3f}, {ds_whole[index_slice]['lon_out'].data[-1]:.3f}; lat={ds_whole[index_slice]['lat_out'].data[0]:.3f}, {ds_whole[index_slice]['lat_out'].data[-1]:.3f}"
                                )
                                logger.debug(
                                    f"extend read in ds: lon={cur_ds['lon_out'].data[0]}, {cur_ds['lon_out'].data[-1]}; lat={cur_ds['lat_out'].data[0]}, {cur_ds['lat_out'].data[-1]}"
                                )
                                continue
                        ds_whole[data_var][index_slice] = cur_ds[data_var].data
                    except KeyError as ke:
                        logger.error(f"Key error in file {restart_file_path}")
                        with ErrorLogger(logger):
                            raise ke
        logger.info("Merging restart files done")
        if "month_of_year_bnds" not in ds_whole.coords:
            logger.info("Adding month_of_year_bnds")
            month_of_year_bnds = [
                [int(m), int(m) + 1] for m in ds_whole["month_of_year"]
            ]
            ds_whole["month_of_year_bnds"] = (
                ["month_of_year", "bnds"],
                month_of_year_bnds,
            )
        # save the intermediate file
        logger.info(f"Writing restart file to {self.grid.restart_file}")
        write_xarray_to_file(ds=ds_whole, file_path=self.grid.restart_file)
        logger.info("Renaming coordinates and data variables")

        return ds_whole

    def _drop_unnecessary_dims(self, ds, *args):
        for arg in args:
            if arg in ds.coords:
                ds = ds.drop(arg)
        return ds

    def _correct_restart_file(self, ds=None):
        if ds is None:
            ds = get_xarray_ds_from_file(self.grid.restart_file)
        ds_mask = get_xarray_ds_from_file(self.grid.land_mask_file)
        logger.debug(
            f"land mask shape before sel: {ds_mask.land_mask.shape} lat_min: {ds_mask.land_mask.lat.min()}, lat_max: {ds_mask.land_mask.lat.max()}"
        )
        ds_mask = ds_mask.sel(
            lon=slice(self.grid.l1.lon_min, self.grid.l1.lon_max),
            lat=slice(self.grid.l1.lat_max, self.grid.l1.lat_min),
        )
        logger.debug(
            f"land mask shape after sel: {ds_mask.land_mask.shape} lat_min: {ds_mask.land_mask.lat.min()}, lat_max: {ds_mask.land_mask.lat.max()}"
        )
        # logger.debug(f'lon: ds {ds['lon_out'].values[0]}-{ds['lon_out'].values[-1]} ; grid {self.grid.l1.lon_min}-{self.grid.l1.lon_max}')
        # logger.debug(f'lat: ds {ds['lat_out'].values[0]}-{ds['lat_out'].values[-1]} ; grid {self.grid.l1.lat_min}-{self.grid.l1.lat_max}')
        # logger.debug(f'mask: {np.shape(ds_mask["land_mask"].data)}')
        # logger.debug(f'mask lon: {ds_mask['lon'].data[0]} to {ds_mask['lon'].data[-1]} with length {np.shape(ds_mask['lon'].data)}')
        # logger.debug(f'ds: {np.shape(ds["L1_SoilMoistureExponent"].data)}')
        try:
            mask_lat_key = get_coord_key(ds_mask, lat=True)
            ds_mask = ds_mask.sortby(mask_lat_key)
        except Exception as e:
            logger.error(f"Could not sort by latitude {e}")
            ds_mask = ds_mask.sortby("lat")
        # ds_mask = xr.open_dataset(
        #     "/work/luedke/land_mask_0p1.nc"
        # ).sortby("latitude")
        ncells = int(ds_mask["land_mask"].sum())
        ds.attrs = {
            "xllcorner_L1": (
                self.grid.l1.lon_min
                if self.grid.l1.lon_min != int(self.grid.l1.lon_min)
                else int(self.grid.l1.lon_min)
            ),
            "yllcorner_L1": (
                self.grid.l1.lat_min
                if self.grid.l1.lat_min != int(self.grid.l1.lat_min)
                else int(self.grid.l1.lat_min)
            ),
            "nrows_L1": self.grid.l1.get_n_lon(),
            "ncols_L1": self.grid.l1.get_n_lat(),
            "cellsize_L1": self.grid.l1.resolution,
            "nCells_L1": ncells,
            "xllcorner_L0": (
                self.grid.l1.lon_min
                if self.grid.l1.lon_min != int(self.grid.l1.lon_min)
                else int(self.grid.l1.lon_min)
            ),
            "yllcorner_L0": (
                self.grid.l1.lat_min
                if self.grid.l1.lat_min != int(self.grid.l1.lat_min)
                else int(self.grid.l1.lat_min)
            ),
            "nrows_L0": self.grid.l1.get_n_lon(),
            "ncols_L0": self.grid.l1.get_n_lat(),
            "cellsize_L0": self.grid.l1.resolution,
            "nCells_L0": ncells,
        }
        ds["L1_domain_mask"] = (
            ("lat_out", "lon_out"),
            ds_mask["land_mask"].data.astype(int),
        )
        ds["L1_domain_lat"] = (
            ("lat_out", "lon_out"),
            np.stack([ds["lat_out"].data] * len(ds["lon_out"]), axis=1),
        )
        ds["L1_domain_lon"] = (
            ("lat_out", "lon_out"),
            np.stack([ds["lon_out"].data] * len(ds["lat_out"]), axis=0),
        )
        ds["L1_domain_cellarea"] = (
            ("lat_out", "lon_out"),
            np.full_like(
                ds_mask["land_mask"].data, self.grid.l1.resolution**2, dtype=float
            ),
        )
        ds["L0_domain_mask"] = (
            ("lat_out", "lon_out"),
            ds_mask["land_mask"].data.astype(int),
        )
        ds["L0_domain_lat"] = (
            ("lat_out", "lon_out"),
            np.stack([ds["lat_out"].data] * len(ds["lon_out"]), axis=1),
        )
        ds["L0_domain_lon"] = (
            ("lat_out", "lon_out"),
            np.stack([ds["lon_out"].data] * len(ds["lat_out"]), axis=0),
        )
        ds["L0_domain_cellarea"] = (
            ("lat_out", "lon_out"),
            np.full_like(
                ds_mask["land_mask"].data, self.grid.l1.resolution**2, dtype=float
            ),
        )
        ds["L1_fAsp"] = (
            ("lat_out", "lon_out"),
            np.ones((len(ds["lat_out"]), len(ds["lon_out"]))),
        )
        ds["L1_degDay"] = (
            ("lat_out", "lon_out"),
            np.ones((len(ds["lat_out"]), len(ds["lon_out"]))),
        )

        BNDS_VALUES = {
            "L1_SoilHorizons_bnds": np.array([[0.0, 300], [300, 1000], [1000, 2000]]),
            "L1_LandCoverPeriods_bnds": np.array([[1900, 2099]]),
            "L1_LAITimesteps_bnds": np.array(list(zip(range(12), range(1, 13)))),
            "lon_bnds": np.array(
                list(
                    zip(
                        np.linspace(
                            self.grid.l1.lon_min,
                            self.grid.l1.lon_max - self.grid.l1.resolution,
                            self.grid.l1.get_n_lon(),
                        ),
                        np.linspace(
                            self.grid.l1.lon_min + self.grid.l1.resolution,
                            self.grid.l1.lon_max,
                            self.grid.l1.get_n_lon(),
                        ),
                    )
                )
            ),
            "lat_bnds": np.array(
                list(
                    zip(
                        np.linspace(
                            self.grid.l1.lat_min,
                            self.grid.l1.lat_max - self.grid.l1.resolution,
                            self.grid.l1.get_n_lat(),
                        ),
                        np.linspace(
                            self.grid.l1.lat_min + self.grid.l1.resolution,
                            self.grid.l1.lat_max,
                            self.grid.l1.get_n_lat(),
                        ),
                    )
                )
            ),
        }

        logger.debug(f"BNDS_VALUES: {BNDS_VALUES}")

        ds["horizon_out"] = (
            ("horizon_out",),
            BNDS_VALUES["L1_SoilHorizons_bnds"][:, 1],
        )

        # for coord in ds.coords:
        #     ds[coord].encoding["missing_value"] = np.nan
        ds = ds.rename(
            {
                "horizon_out": "L1_SoilHorizons",
                "horizon_out_bnds": "L1_SoilHorizons_bnds",
                "land_cover_period_out": "L1_LandCoverPeriods",
                "land_cover_period_out_bnds": "L1_LandCoverPeriods_bnds",
                "month_of_year": "L1_LAITimesteps",
                "month_of_year_bnds": "L1_LAITimesteps_bnds",
                "lat_out": "lat",
                "lon_out": "lon",
                "lat_out_bnds": "lat_bnds",
                "lon_out_bnds": "lon_bnds",
                # something to make compat with 22-couple-with_mpr-branch only
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
                # "L1_PET_LAI_correction_factor": "L1_petLAIcorFactor",
                # "L1_Aerodyn_resist": "L1_aeroResist",
                # "L1_Bulk_Surface_Resist": "L1_surfResist",
            }
        ).squeeze("horizon_all", drop=True)

        ds = self._drop_unnecessary_dims(
            ds, "horizon_all_bnds", "land_cover_period", "longitude", "latitude"
        )

        mask = np.stack(
            [np.isnan(ds["L1_maxInter"].data)[0, :, :] & ds_mask["land_mask"].data]
            * ds["L1_maxInter"].shape[0],
            axis=0,
        )
        ds["L1_maxInter"] = xr.where(mask, 0, ds["L1_maxInter"])

        BNDS_DIMS = {
            "L1_SoilHorizons_bnds": ("L1_SoilHorizons", "bnds"),
            "L1_LandCoverPeriods_bnds": ("L1_LandCoverPeriods", "bnds"),
            "L1_LAITimesteps_bnds": ("L1_LAITimesteps", "bnds"),
            "lat_bnds": ("lat", "bnds"),
            "lon_bnds": ("lon", "bnds"),
        }

        ds["lat_bnds"] = (BNDS_DIMS["lat_bnds"], BNDS_VALUES["lat_bnds"])
        ds["lon_bnds"] = (BNDS_DIMS["lon_bnds"], BNDS_VALUES["lon_bnds"])

        # apply the final mask on parameters
        for data_var in ds.data_vars:
            if data_var in BNDS_DIMS:
                ds[data_var] = (BNDS_DIMS[data_var], BNDS_VALUES[data_var])
                logger.debug(
                    f"Setting {data_var} to {(BNDS_DIMS[data_var], BNDS_VALUES[data_var])}"
                )
                # logger.info(data_var, ds[data_var])
                continue
            if not ("lat" in ds[data_var].dims and "lon" in ds[data_var].dims):
                continue
            # those grids need to be defined in its entirety
            if data_var.endswith(("_lon", "_lat", "_cellarea")):
                continue
            slicer = []
            for dim in ds[data_var].dims:
                if dim == "lat":
                    slicer.append(...)
                elif dim != "lon":
                    slicer.append(None)
            logger.debug(f"Masking {data_var}")
            ds[data_var] = ds[data_var].where(
                np.broadcast_to(
                    ds_mask["land_mask"].data[tuple(slicer)], ds[data_var].shape
                )
            )
            if "L1_SoilHorizons" in ds[data_var].dims:
                ds[data_var] = ds[data_var].transpose(
                    "L1_LandCoverPeriods", "L1_SoilHorizons", "lat", "lon"
                )

        COORD_ATTRS = {
            "L1_SoilHorizons": {
                "bounds": "L1_SoilHorizons_bnds",
                "units": "mm",
                "positive": "down",
                "long_name": "depth",
                "standard_name": "depth",
                "axis": "Z",
            },
            "L1_LandCoverPeriods": {
                "bounds": "L1_LandCoverPeriods_bnds",
                "units": "years",
                "long_name": "time period",
                "standard_name": "time period",
                "axis": "T",
            },
            "L1_LAITimesteps": {
                "bounds": "L1_LAITimesteps_bnds",
                "units": "month of year",
                "long_name": "time: means within months",
                "standard_name": "month of year",
                "axis": "T",
            },
            "lat": {
                "bounds": "lat_bnds",
                "units": "degrees_north",
                "long_name": "latitude",
                "standard_name": "latitude",
                "axis": "Y",
            },
            "lon": {
                "bounds": "lon_bnds",
                "units": "degrees_east",
                "long_name": "longitude",
                "standard_name": "longitude",
                "axis": "X",
            },
        }

        GLOBAL_ATTRS = {
            "institution": "Helmholtz-Centre for Environmental Research - UFZ, Leipzig, Germany",
            "creator": "Robert Schweppe",
            "contact": "stephan.thober@ufz.de",
        }

        for coord_name, attrs in COORD_ATTRS.items():
            ds[coord_name].attrs.update(attrs)
        ds.attrs.update(GLOBAL_ATTRS)

        ds = ds.sortby("lat", ascending=False)

        self.grid.restart_file = (
            self.grid.restart_file.parent
            / f"mHM_restart_001{self.grid.restart_file.suffix}"
        )
        logger.info(f"Writing renamed restart file to {self.grid.restart_file}")
        encoding = {}
        for data_var in ds.data_vars:
            encoding[data_var] = {
                "dtype": "float32",
                "_FillValue": -9999.0,
                "zlib": True,
                "complevel": 4,
            }
        for coord in ds.coords:
            encoding[coord] = {
                "dtype": "float32",
                "_FillValue": -9999.0,
                "zlib": True,
                "complevel": 4,
            }
        if not self.output_path.is_dir():
            self.output_path.mkdir(parents=True)
        output_file = self.output_path / self.grid.restart_file.name
        write_xarray_to_file(ds, output_file, encoding=encoding)
        self.grid.restart_file = output_file

    def _delete_temp_files(self):
        logger.info("Deleting temporary files")
        # remove all temporary files meaning all files in the morph_files of each subgrid
        for subgrid in self.subgrids:
            for file_path in subgrid.morph_files.get_files_as_list():
                if isinstance(file_path, list):
                    for f in file_path:
                        # f.unlink() # ofnly deletes the file not the dir and not the namelist_file
                        shutil.rmtree(f.parent)
                else:
                    file_path.unlink()

    def _prepare_slope_emp(self, n=10000):
        logger.info("Preparing slope_emp")
        td = TDigest(compression=n)
        if self.run_on_whole_domain:
            if (
                self.grid.morph_files.slope_emp is None
                or not self.grid.morph_files.slope_emp.is_file()
            ):
                with get_xarray_ds_from_file(self.grid.morph_files.slope) as ds_slope:
                    data = ds_slope["slope"]
                    flattened = data.values.flatten()
                    flattened_no_nan = flattened[~np.isnan(flattened)]
                    td.update(flattened_no_nan)
                    cdf = td.cdf(data.values)
                    cdf = xr.DataArray(cdf, dims=["latitude", "longitude"])
                    ds_slope["slope"] = cdf
                    ds_slope_emp = ds_slope.rename({"slope": "slope_emp"})
                    write_xarray_to_file(ds_slope_emp, self.grid.path / "slope_emp.nc")
                    self.grid.morph_files.slope_emp = self.grid.path / "slope_emp.nc"
        else:
            for sgrid in self.subgrids:
                with get_xarray_ds_from_file(sgrid.morph_files.slope) as ds_slope:
                    data = ds_slope["slope"]
                    flattened = data.values.flatten()
                    flattened_no_nan = flattened[~np.isnan(flattened)]
                    td.update(flattened_no_nan)
            for sgrid in self.subgrids:
                with get_xarray_ds_from_file(sgrid.morph_files.slope) as ds_slope:
                    data = ds_slope["slope"]
                    cdf = td.cdf(data.values)
                    cdf = xr.DataArray(cdf, dims=["latitude", "longitude"])
                    ds_slope["slope"] = cdf
                    ds_slope_emp = ds_slope.rename({"slope": "slope_emp"})
                    write_xarray_to_file(ds_slope_emp, sgrid.path / "slope_emp.nc")
                    sgrid.morph_files.slope_emp = sgrid.path / "slope_emp.nc"

    def _create_restart_for_grid(self, grid):
        logger.debug(
            f"Processing subgrid {grid.name}, {grid.path}, {grid.l0.lon_min}, {grid.l0.lon_max}, {grid.l0.lat_min}, {grid.l0.lat_max}"
        )
        grid = self._write_grid_namelist(grid)
        self.mpr.run_mpr(grid)
        return grid

    @log_arguments()
    def create_restart_file(self):
        """Create a restart file for the MHM model.

        This method creates a restart file by splitting the grid (if necessary),
        writing the grid namelist, calling the mpr executable, merging the restart
        files (if applicable), and deleting temporary files (if specified).

        Returns
        -------
            None

        Raises
        ------
            Any exceptions that occur during the execution of the method.
        """
        logger.info("Creating restart file")
        if self.run_on_whole_domain:
            logger.info("grid will be processed as a whole")
            self._prepare_slope_emp()
            self._create_restart_for_grid(self.grid)
            self._correct_restart_file()
        else:
            logger.info(
                f"grid will be split and processed in parallel on {self.ncpus} cores"
            )
            if not self.merge_only:
                if self.use_split_grids:
                    self._read_subgrids_from_files()
                else:
                    self._split_grid()
                    self._prepare_slope_emp()
                logger.info(
                    f"The grid has been split into {len(self.subgrids)} subgrids."
                )
                logger.info("Creating namelists and running MPR.")
                subgrids = Parallel(n_jobs=self.ncpus, backend="loky")(
                    delayed(self._create_restart_for_grid)(subgrid)
                    for subgrid in self.subgrids
                )
                self.subgrids = subgrids
            if self.merge_grid:
                # merge the restart files
                dataset = self._merge_restart_files()
                # change the coordinates and data variables to the correct names
                self._correct_restart_file(dataset)
            if self.clean_temp_files:
                self._delete_temp_files()
        logger.info("Script finished successfully.")
