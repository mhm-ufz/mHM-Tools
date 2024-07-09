"""Create the mHM restart file."""

import logging
import re
import shutil
from pathlib import Path
from subprocess import PIPE, Popen, TimeoutExpired

import numpy as np
import xarray as xr
from joblib import Parallel, delayed

from mhm_tools.common.constants import LOG_LEVELS

logging.basicConfig(format="%(asctime)s - %(levelname)-8s - %(message)s")
logger = logging.getLogger(__name__)


class MorphFiles:
    """
    A class representing a collection of morphological files.

    Attributes
    ----------
        land_cover (Path): The path to the land cover file.
        bulk_density (Path): The path to the bulk density file.
        sand_content (Path): The path to the sand content file.
        clay_content (Path): The path to the clay content file.
        slope (Path): The path to the slope file.
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
        lai=None,
        aspect=None,
        geology=None,
    ):
        self.land_cover = land_cover
        self.bulk_density = bulk_density
        self.sand_content = sand_content
        self.clay_content = clay_content
        self.slope = slope
        self.lai = lai
        self.aspect = aspect
        self.geology = geology

        if filepath is not None:
            self.read_files(filepath)

    def read_files(self, filepath: Path, overwrite=False):
        """
        Read files from the specified filepath and assigns them to the corresponding attributes.

        Args:
            filepath (Path): The path to the directory containing the files.
            overwrite (bool, optional): If False, existing attribute values will not be overwritten.
                Defaults to False.
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
        """
        Retrieve the file path associated with the given name of the member variable.

        Parameters
        ----------
            key (str): The member-variable name to retrieve the filepath for.

        Returns
        -------
            object: The filepath or list of filepaths associated with the given key, or None if the key is not found.
        """
        return self.__dict__.get(key, None)

    def get_files_as_list(self):
        """
        Return a list of all files in the object's attributes.

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
        """
        Return a dictionary of all files in the object's attributes.

        Returns
        -------
            dict: A dictionary containing all files in the object's attributes.
        """
        return self.__dict__


class LatLon:
    """
    Represents a latitude-longitude coordinate system.

    Attributes
    ----------
        lat_min (float): The minimum latitude value.
        lon_min (float): The minimum longitude value.
        lat_max (float): The maximum latitude value.
        lon_max (float): The maximum longitude value.
        resolution (float): The resolution of the coordinate system.
    """

    def __init__(
        self, lat_min=None, lon_min=None, lat_max=None, lon_max=None, resolution=None
    ):
        self.lat_min = lat_min
        self.lon_min = lon_min
        self.lat_max = lat_max
        self.lon_max = lon_max
        self.resolution = resolution

    def get_n_lat(self):
        """
        Calculate the number of latitude points based on the given latitude range and resolution.

        Returns
        -------
            int: The number of latitude points.
        """
        # print('nlat',self.lat_max, self.lat_min, self.resolution, (self.lat_max - self.lat_min) / self.resolution, int((self.lat_max - self.lat_min) / self.resolution), flush=True)
        return int(
            (self.lat_max - self.lat_min) / self.resolution + 0.5
        )  # + 0.5 to round up

    def get_n_lon(self):
        """
        Calculate the number of longitude points based on the given longitude range and resolution.

        Returns
        -------
            int: The number of longitude points.
        """
        # print('nlon', self.lon_max, self.lon_min, self.resolution, (self.lon_max - self.lon_min) / self.resolution, int((self.lon_max - self.lon_min) / self.resolution), flush=True)
        return int(
            (self.lon_max - self.lon_min) / self.resolution + 0.5
        )  # + 0.5 to round up

    def is_fully_defined(self):
        """
        Check if all the required attributes are fully defined.

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
    """
    Represents a geographical area for wich morphological data exists.

    This grid is used to run the mPR model. It does not need to contain a whole catchment, but can be a subset of it or multiple catchments at once.

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
    ):
        file_path = Path(file_path)
        self.morph_files = MorphFiles(filepath=file_path)
        self.name = name
        self.path = file_path
        self.l0 = l0
        self.l1 = l1
        self.restart_file = None
        self.namelist_file = None
        if (
            self.l0 is None
            or not self.l0.is_fully_defined()
            or self.l1 is None
            or not self.l1.is_fully_defined()
        ) and latlon_file is not None:
            self.read_latlon(latlon_file)

        def set_restart_file(self, restart_file):
            self.restart_file = restart_file

        def set_namelist_file(self, namelist_file):
            self.namelist_file = namelist_file

    def read_latlon(self, latlon_file: Path):
        """
        Read the latlon file and sets the lower-left (l0) and upper-right (l1) corners of the grid as well as the resolution.

        Args:
            latlon_file (Path): The file path of the latlon file.

        Returns
        -------
            None
        """
        with xr.open_dataset(latlon_file) as ds:
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
        """
        Read the morph files from the specified path.

        This method uses the `read_files` function from the `MorphFiles` object to read the morph files
        located at the specified path.

        Args:
            self: The instance of the class.

        Returns
        -------
            None
        """
        self.morph_files.read_files(self.path)


class MHMRestartFile:
    """
    A class for creating a restart file for the MHM model.

    This class provides methods to split the grid (if necessary), write the grid namelist,
    call the mpr executable, merge the restart files (if applicable), and delete temporary files (if specified).

    Parameters
    ----------
        input_file_path (Path): The path to the input file.
        nml_template (Path): The path to the namelist template file.
        output_path (Path): The path to the output directory.
        mpr_executable (str, optional): The path to the mpr executable. Defaults to None.
        mpr_parameter_file (str, optional): The path to the mpr parameter file. Defaults to None.
        lon_min_target_grid (float, optional): The minimum longitude of the target grid. Defaults to None.
        lon_max_target_grid (float, optional): The maximum longitude of the target grid. Defaults to None.
        lat_min_target_grid (float, optional): The minimum latitude of the target grid. Defaults to None.
        lat_max_target_grid (float, optional): The maximum latitude of the target grid. Defaults to None.
        l0_resolution (float, optional): The resolution of the high-resolution grid. Defaults to None.
        l1_resolution (float, optional): The resolution of the low-resolution grid. Defaults to None.
        increment_l1 (int, optional): The increment for splitting the grid. Defaults to 2.
        split_grid (bool, optional): Whether to split the grid. Defaults to False.
        ncpus (int, optional): The number of CPUs to use for parallelization. Defaults to 1.
        clean_temp_files (bool, optional): Whether to clean temporary files. Defaults to False.
        log_level (int, optional): The log level. Defaults to logging.DEBUG.
        mpr_packages (str, optional): The mpr packages to load. Defaults to None.

    Attributes
    ----------
        nml_template (Path): The path to the namelist template file.
        output_path (Path): The path to the output directory.
        grid (Grid): The grid object.
        subgrids (list): The list of subgrid objects.
        ncpus (int): The number of CPUs to use for parallelization.
        run_on_whole_domain (bool): Whether to run on whole domain or split.
        clean_temp_files (bool): Whether to clean temporary files.
        mpr_executable (str): The path to the mpr executable.
        mpr_packages (str): The mpr packages to load.
        parameter_file (str): The path to the mpr parameter file.
        work_dir (str): The working directory.
        increment_l1 (int): The increment for splitting the grid.
        increment_l0 (int): The increment for the high-resolution grid.

    Methods
    -------
        _create_namelist(replace_dict, out_file_path, overwrite=False): Create a namelist file with the given replace dictionary.
        _write_grid_namelist(grid): Write the namelist for a grid.
        _split_grid(): Split the grid into subgrids and write them to disk.
        _split_file(name, file_path): Split a file into subgrids.
        _call_mpr(namelist): Call the mpr executable with the given namelist and parameter file.
    """

    def __init__(
        self,
        input_file_path: Path,
        nml_template: Path,
        output_path: Path,
        mpr_executable=None,
        mpr_parameter_file=None,
        lon_min_target_grid=None,
        lon_max_target_grid=None,
        lat_min_target_grid=None,
        lat_max_target_grid=None,
        l0_resolution=None,
        l1_resolution=None,
        increment_l1=2,
        run_on_whole_domain=False,
        use_split_grids=False,
        ncpus=1,
        clean_temp_files=False,
        log_level=logging.DEBUG,
        mpr_packages=None,
        merge=True,
        merge_only=False,
    ):
        logger.setLevel(
            LOG_LEVELS[log_level] if log_level in LOG_LEVELS else logging.INFO
        )
        logger.debug(f"Creating MHMRestartFile object with {locals()}")
        self.nml_template = Path(nml_template)
        self.output_path = Path(output_path)
        grid_latlon_l0 = LatLon(
            lat_min=lat_min_target_grid,
            lon_min=lon_min_target_grid,
            lat_max=lat_max_target_grid,
            lon_max=lon_max_target_grid,
            resolution=l0_resolution,
        )
        grid_latlon_l1 = LatLon(
            lat_min=lat_min_target_grid,
            lon_min=lon_min_target_grid,
            lat_max=lat_max_target_grid,
            lon_max=lon_max_target_grid,
            resolution=l1_resolution,
        )
        self.grid = Grid(
            file_path=Path(input_file_path),
            name="whole grid",
            latlon_file=None,
            l0=grid_latlon_l0,
            l1=grid_latlon_l1,
        )
        self.subgrids = []  # list of grid objects
        self.ncpus = ncpus
        self.run_on_whole_domain = run_on_whole_domain
        self.use_split_grids = use_split_grids
        self.merge_grid = merge
        self.merge_only = merge_only
        self.clean_temp_files = clean_temp_files
        self.mpr_executable = mpr_executable
        self.mpr_packages = mpr_packages
        self.parameter_file = mpr_parameter_file
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
        replace_dict = {
            "${slicei_j}": grid.name,
            "${output_file}": grid.path / f"output_{grid.name}.nc",
            "${lon_high_start}": f"{grid.l0.lon_min + grid.l0.resolution / 2:.3f}",  # changes to center of cell coordinates
            "${lon_high_res}": f"{grid.l0.resolution:.3f}",
            "${lon_high_n}": f"{grid.l0.get_n_lon()}",
            "${lat_high_start}": f"{grid.l0.lat_min + grid.l0.resolution / 2:.3f}",  # changes to center of cell coordinates
            "${lat_high_res}": f"{grid.l0.resolution:.3f}",
            "${lat_high_n}": f"{grid.l0.get_n_lat()}",
            "${lon_low_start}": f"{grid.l1.lon_min + grid.l1.resolution / 2:.2f}",  # changes to center of cell coordinates
            "${lon_low_res}": f"{grid.l1.resolution:.2f}",
            "${lon_low_n}": f"{grid.l1.get_n_lon()}",
            "${lat_low_start}": f"{grid.l1.lat_min + grid.l1.resolution / 2:.2f}",  # changes to center of cell coordinates
            "${lat_low_res}": f"{grid.l1.resolution:.2f}",
            "${lat_low_n}": f"{grid.l1.get_n_lat()}",
            "${bulk_density}": grid.morph_files.bulk_density,
            "${sand_content}": grid.morph_files.sand_content,
            "${clay_content}": grid.morph_files.clay_content,
            "${slope}": grid.morph_files.slope,
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
    
    def _read_subgrids_from_files(self):
        for subgrid_path in self.output_path.glob("slice_*"):
            logger.debug(f"Reading subgrid {subgrid_path}")
            grid = Grid(file_path=subgrid_path, name=subgrid_path.name, l0=self.grid.l0, l1=self.grid.l1)
            grid.read_morph_files()
            self.subgrids.append(grid)

    def _split_grid(self):  # has do addapted to different file types not just .nc
        """
        Split the grid into subgrids and write them to disk.

        Subgrids are subsets of the original grid with a size of increment x increment grid cells.
        """
        logger.info("Splitting grid")
        if self.increment_l0 is None:
            error_message = "Increment for splitting grids is not set"
            raise ValueError(error_message)
        sub_grid_paths = Parallel(n_jobs=self.ncpus, backend="loky")(
            delayed(self._split_file)(name, file_path)
            for name, file_path in self.grid.morph_files.get_files_as_dict().items()
        )  # move the parallelization into the split_file function to improve performance
        logger.debug("Creating subgrids")
        self.subgrids = [Grid(file_path=k, **v) for k, v in sub_grid_paths[0].items()]
        logger.debug("Splitting grid done")

    def _split_file(self, name, file_path):
        sub_grid_paths = {}
        if file_path is None:
            logger.error(f"No file path provided for {name}")
            return None
        logger.debug(f"Splitting {file_path}")
        with xr.open_dataset(file_path) as ds:
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
                    out_dir = Path(self.output_path) / f"slice_{i}_{j}"
                    if not out_dir.exists():
                        logger.debug(f"Creating {out_dir}")
                        out_dir.mkdir(parents=True, exist_ok=True)

                    out_path = out_dir / f"{file_path.stem}.nc"

                    if out_path.is_file():
                        out_path.unlink()

                    lon_max = lon_min + self.increment_l1 * self.grid.l1.resolution
                    lat_max = lat_min + self.increment_l1 * self.grid.l1.resolution
                    ds_cut = ds.sel(
                        longitude=slice(lon_min, lon_max),
                        latitude=slice(lat_max, lat_min),
                    )
                    try:
                        ds_cut.to_netcdf(out_path, "w")
                        logger.debug(f"Written {out_path}")
                    except Exception as e:
                        logger.error(f"Failed to write {out_path} with {e}")
                        logger.debug(f"{lon_min}, {lon_max}, {lat_min}, {lat_max}")
                        logger.debug(ds_cut["latitude"].values)
                        return None

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
                    sub_grid_paths[out_dir] = {
                        "l0": l0,
                        "l1": l1,
                        "name": f"slice_{i}_{j}",
                    }
        logger.debug(f"Splitting {file_path} done")
        return sub_grid_paths

    def _call_mpr(self, grid: Grid):
        """Call the mpr executable with the given namelist and parameter file."""
        # tmpdir = Path.cwd()
        # os.chdir(self.work_dir)
        command = (
            f"module load {self.mpr_packages} \n"
            if self.mpr_packages is not None
            else ""
        )
        # command = f"""module load iomkl/2020b netCDF-Fortran/4.5.3
        command += f"""{self.mpr_executable} -c {grid.namelist_file}"""
        if self.parameter_file is not None:
            command += f" -p {self.parameter_file}"
        logger.info(f"Running mPR with: {command}")

        p = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
        try:
            data, error_data = p.communicate()
            if error_data:
                grid.restart_file = None
                # logger.error(f"Failed with STDERR {error_data}")
                msg = f"MPR failed with STDERR {error_data} for {grid.name} and command {command}"
                raise RuntimeError(msg)
        except TimeoutExpired:
            # logger.error("Timeout expired")
            p.kill()
            msg = "MPR failed with TimeoutExpired for {grid.name} and command {command}"
            raise TimeoutExpired("MPR timeout expired")
        # finally:
        # os.chd    ir(tmpdir)

    def _merge_restart_files(self):
        logger.info("Merging restart files")

        # 1. create an empty file for the whole grid
        ds_whole = xr.Dataset()
        ds_whole["longitude"] = np.arange(
            self.grid.l0.lon_min, self.grid.l0.lon_max, self.grid.l0.resolution
        )
        ds_whole["latitude"] = np.arange(
            self.grid.l0.lat_min, self.grid.l0.lat_max, self.grid.l0.resolution
        )
        ds_whole["lon_out"] = np.arange(
            self.grid.l1.lon_min + self.grid.l1.resolution / 2,
            self.grid.l1.lon_max + self.grid.l1.resolution / 2, # + since arange omits the last value
            self.grid.l1.resolution,
        )
        ds_whole["lat_out"] = np.arange(
            self.grid.l1.lat_min + self.grid.l1.resolution / 2,
            self.grid.l1.lat_max + self.grid.l1.resolution / 2, # + since arange omits the last value
            self.grid.l1.resolution,
        )

        # 2. create all coordinates in the whole grid
        # TODO: add dimensions to comments to make it more readable
        if self.grid.restart_file is None:
            self.grid.restart_file = self.output_path / "output_whole_grid_restart.nc"
            logger.warning(
                f"No restart file for the whole grid setting it to {self.grid.restart_file}"
            )
        else:
            logger.info(f"Creating whole grid with {self.grid.restart_file}")
        if self.merge_only:
            restart_file_paths = [
                file
                for dir in self.output_path.glob("slice_*")
                for file in dir.glob("output_*.nc")
            ]
        else:
            restart_file_paths = [subgrid.restart_file for subgrid in self.subgrids]
        restart_file_paths.sort()
        logger.info(f"Opening {restart_file_paths[0]} als reference")
        if not restart_file_paths[0].is_file():
            logger.error(f"Could not open {restart_file_paths[0]}")
        with xr.open_dataset(restart_file_paths[0]) as cur_ds:
            for coord in cur_ds.coords:
                if coord not in ds_whole.coords:
                    ds_whole[coord] = cur_ds[coord]
                # init the new bounds (e.g. horizons_out_bnds)
                data_vars = [
                    _
                    for _ in cur_ds.data_vars
                    if _ not in ds_whole.data_vars and _.endswith("_out_bnds")
                ]
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
                for data_var in data_vars:
                    coords = [_ for _ in cur_ds[data_var].coords]
                    # print(data_var, coords)
                    for coord in coords:
                        if coord not in ds_whole:
                            logger.info(f"Adding {coord} to {data_var}")
                            logger.debug(f"cur_ds[coord] {cur_ds[coord]}")
                            ds_whole[coord] = cur_ds[coord]
                    ds_whole[data_var] = (
                        coords,
                        np.full([len(ds_whole[_]) for _ in coords], np.nan),
                    )

        # 3. iterate over all subgrids and merge them into the whole grid
        for restart_file_path in restart_file_paths:
            ints = re.findall(r"\d+", str(restart_file_path))
            isel_start = int(ints[-2])
            jsel_start = int(ints[-1])
            with xr.open_dataset(restart_file_path) as cur_ds:
                # logger.warning(f"Could not open {restart_file}")
                # continue
                for data_var in data_vars:
                    index_slice = dict(
                        lon_out=slice(
                            isel_start * self.increment_l1,
                            (isel_start + 1) * self.increment_l1,
                        ),
                        lat_out=slice(
                            jsel_start * self.increment_l1,
                            (jsel_start + 1) * self.increment_l1,
                        ),
                    )
                    if cur_ds[data_var].shape != ds_whole[data_var][index_slice].shape:
                        logger.warning(
                            f"Shape mismatch for {data_var} in {restart_file_path}"
                        )
                        dims = cur_ds[data_var].dims
                        ds_whole[data_var] = ds_whole[data_var].transpose(*dims)
                        if (
                            cur_ds[data_var].shape
                            != ds_whole[data_var][index_slice].shape
                        ):
                            logger.error(
                                f"Shape mismatch could not be resolved for {data_var} in {restart_file_path}"
                            )
                            continue
                    ds_whole[data_var][index_slice] = cur_ds[data_var].data
        rename_dict = {'lon_out': 'lon', 'lat_out': 'lat', 'lon_out_bnds': 'lon_bnds', 'lat_out_bnds': 'lat_bnds', 'month_of_year': 'L1_LAITimesteps', 'horizon_out': 'L1_SoilHorizons', '    horizon_out_bnds': 'L1_SoilHorizons_bnds'}
        rename_dict = {k: v for k, v in rename_dict.items() if k in ds_whole.coords} # make sure that all keys are in the dataset
        ds_whole = ds_whole.rename(rename_dict)
        ds_whole.to_netcdf(self.grid.restart_file)
        logger.info("Merging restart files done")

    def _delete_temp_files(self):
        logger.info("Deleting temporary files")
        # remove all temporary files meaning all files in the morph_files of each subgrid
        for subgrid in self.subgrids:
            for file_path in subgrid.morph_files.get_files_as_list():
                if isinstance(file_path, list):
                    for f in file_path:
                        # f.unlink() # only deletes the file not the dir and not the namelist_file
                        shutil.rmtree(f.parent)
                else:
                    file_path.unlink()

    def _create_restart_for_grid(self, grid):
        logger.debug(
            f"Processing subgrid {grid.name}, {grid.path}, {grid.l0.lon_min}, {grid.l0.lon_max}, {grid.l0.lat_min}, {grid.l0.lat_max}"
        )
        grid = self._write_grid_namelist(grid)
        self._call_mpr(grid)
        return grid

    def create_restart_file(self):
        """
        Create a restart file for the MHM model.

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
            self._create_restart_for_grid(self.grid)
        else:
            logger.info(
                f"grid will be split and processed in parallel on {self.ncpus} cores"
            )
            if not self.merge_only:
                if self.use_split_grids:
                    self._read_subgrids_from_files()
                else:
                    self._split_grid()
                subgrids = Parallel(n_jobs=self.ncpus, backend="loky")(
                    delayed(self._create_restart_for_grid)(subgrid)
                    for subgrid in self.subgrids
                )
                self.subgrids = subgrids
            if self.merge_grid:
                self._merge_restart_files()
            if self.clean_temp_files:
                self._delete_temp_files()
            
        logger.info("Restart file created")
