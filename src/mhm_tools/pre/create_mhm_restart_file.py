"""Create the mHM restart file."""

import logging
import os
import sys
from pathlib import Path
from subprocess import PIPE, Popen, TimeoutExpired

import numpy as np
import xarray as xr

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
        logger.debug(self.get_files_as_dir())

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

    def get_files_as_dir(self):
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
        return int((self.lat_max - self.lat_min) / self.resolution + 0.5) # + 0.5 to round up

    def get_n_lon(self):
        """
        Calculate the number of longitude points based on the given longitude range and resolution.

        Returns
        -------
            int: The number of longitude points.
        """
        # print('nlon', self.lon_max, self.lon_min, self.resolution, (self.lon_max - self.lon_min) / self.resolution, int((self.lon_max - self.lon_min) / self.resolution), flush=True)
        return int((self.lon_max - self.lon_min) / self.resolution + 0.5) # + 0.5 to round up

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

        if (
            self.l0 is None
            or not self.l0.is_fully_defined()
            or self.l1 is None
            or not self.l1.is_fully_defined()
        ) and latlon_file is not None:
            self.read_latlon(latlon_file)

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

    Attributes
    ----------
    input_file_path : Path
        The path to the directory containing the input files.
    nml_template : Path
        The path to the namelist template file.
    output_path : Path
        The path to the output directory.
    latlon_file : Optional[Path]
        The path to the latlon file.
    split_grid : bool
        Whether to split the grid into subgrids.
    clean_temp_files : bool
        Whether to clean temporary files.
    increment_l1 : int
        The increment for splitting the grid in number of coarse grid (l1) cells.
    lon_min_target_grid : Optional[float]
        The minimum longitude of the target grid.
    lon_max_target_grid : Optional[float]
        The maximum longitude of the target grid.
    lat_min_target_grid : Optional[float]
        The minimum latitude of the target grid.

    Methods
    -------
    split_grid_if_necessary()
        Split the grid into subgrids if necessary.
    write_grid_namelist()
        Write the grid namelist file.
    call_mpr_executable()
        Call the mpr executable.
    merge_restart_files()
        Merge the restart files if applicable.
    delete_temporary_files()
        Delete temporary files if specified.
    """

    def __init__(
        self,
        input_file_path: Path,
        nml_template: Path,
        output_path: Path,
        split_grid=False,
        clean_temp_files=False,
        increment_l1=2,
        lon_min_target_grid=None,
        lon_max_target_grid=None,
        lat_min_target_grid=None,
        lat_max_target_grid=None,
        l0_resolution=None,
        l1_resolution=None,
        log_level=logging.DEBUG,
        mpr_executable=None,
    ):
        logger.setLevel(log_level)
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
        self.split_grid = split_grid
        self.clean_temp_files = clean_temp_files
        self.mpr_executable = mpr_executable
        self.parameter_file = None
        self.work_dir = "."
        self.increment_l1 = increment_l1
        self.increment_l0 = (
            int(self.increment_l1 * self.grid.l1.resolution / self.grid.l0.resolution)
            if self.increment_l1 is not None
            else None
        )

    def _create_namelist(self, replace_dict, out_file_path, overwrite=False):
        if type(out_file_path) is not Path:
            out_file_path = Path(out_file_path)
        if not out_file_path.exists() or overwrite:
            with self.nml_template.open("r") as f:
                nml_data = f.read()
            for replace_key, replace_value in replace_dict.items():
                nml_data = nml_data.replace(str(replace_key), str(replace_value))
            with out_file_path.open("w") as f:
                f.write(nml_data)
        return out_file_path

    def _write_grid_namelist(self, grid: Grid):
        replace_dict = {
            "${slicei_j}": grid.name,
            "${output_file}": grid.path / f"output_{grid.name}.nc",
            "${lon_high_start}": f"{grid.l0.lon_min:.3f}", # does this need be changed to center of the cell?
            "${lon_high_res}": f"{grid.l0.resolution:.3f}",
            "${lon_high_n}": f"{grid.l0.get_n_lon()}",
            "${lat_high_start}": f"{grid.l0.lat_min:.3f}",# does this need be changed to center of the cell?
            "${lat_high_res}": f"{grid.l0.resolution:.3f}",
            "${lat_high_n}": f"{grid.l0.get_n_lat()}",
            "${lon_low_start}": f"{grid.l1.lon_min:.2f}",# does this need be changed to center of the cell?
            "${lon_low_res}": f"{grid.l1.resolution:.2f}",
            "${lon_low_n}": f"{grid.l1.get_n_lon()}",
            "${lat_low_start}": f"{grid.l1.lat_min:.2f}",# does this need be changed to center of the cell?
            "${lat_low_res}": f"{grid.l1.resolution:.2f}",
            "${lat_low_n}": f"{grid.l1.get_n_lat()}",
            "${bulk_density}": grid.morph_files.bulk_density,
            "${sand_content}": grid.morph_files.sand_content,
            "${clay_content}": grid.morph_files.clay_content,
            "${slope}": grid.morph_files.slope,
            "${lai}": grid.morph_files.lai,
            "${aspect}": grid.morph_files.aspect,
            "${geology}": grid.morph_files.geology,
            "${land_cover}": grid.morph_files.land_cover,  # this should be a list but the template only has one
        }
        logger.info(f"Writing namelist for {grid.name} to {grid.path / 'mpr.nml'}")
        logger.debug(replace_dict)
        return self._create_namelist(replace_dict, grid.path / "mpr.nml")

    def _split_grid(self):  # has do addapted to different file types not just .nc
        """
        Split the grid into subgrids and write them to disk.

        Subgrids are subsets of the original grid with a size of increment x increment grid cells.
        """
        logger.info("Splitting grid")
        if self.increment_l0 is None:
            error_message = "Increment for splitting grids is not set"
            raise ValueError(error_message)
        sub_grid_paths = {}
        for file_path in self.grid.morph_files.get_files_as_list():
            if file_path is None:
                continue  # should raise error
            logger.info(f"Splitting {file_path}")
            self._split_file(file_path, sub_grid_paths)
            logger.debug(f"Splitting {file_path} done")
        logger.debug("Creating subgrids")
        self.subgrids = [Grid(file_path=k, **v) for k, v in sub_grid_paths.items()]
        logger.debug("Splitting grid done")

    def _split_file(self, file_path, sub_grid_paths):
        logger.debug(f"Splitting {file_path}")
        # logger.debug(f"{self.grid.l0.get_n_lon()}, {self.increment_l0}, {self.grid.l0.get_n_lon() // self.increment_l0}")
        # logger.debug(f"{self.grid.l0.get_n_lat()}, {self.increment_l0}, {self.grid.l0.get_n_lat() // self.increment_l0}")
        with xr.open_dataset(file_path) as ds:
            for i, lon_min in enumerate(
                np.arange(
                    self.grid.l1.lon_min,
                    self.grid.l1.lon_max,
                    self.grid.l1.resolution * self.increment_l1
                )
            ):
                for j, lat_min in enumerate(
                    np.arange(
                        self.grid.l1.lat_min,
                        self.grid.l1.lat_max,
                        self.grid.l1.resolution * self.increment_l1
                    )
                ):
                    out_dir = Path(self.output_path) / f"slice_{i}_{j}"
                    if not out_dir.exists():
                        logger.debug(f"Creating {out_dir}")
                    out_dir.mkdir(parents=True, exist_ok=True)

                    out_path = out_dir / f"{file_path.stem}.nc"

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
                        return
                    
                    # grid saved in llc coordinates
                    l0 = LatLon( # this is the high resolution grid
                        lon_min=lon_min,
                        lon_max=lon_max,
                        lat_min=lat_min,
                        lat_max=lat_max,
                        resolution=self.grid.l0.resolution,
                    )
                    l1 = LatLon( # this is the low resolution grid
                        lon_min=lon_min,
                        lon_max=lon_max,
                        lat_min=lat_min,
                        lat_max=lat_max,
                        resolution=self.grid.l1.resolution,
                    )
                    if out_dir not in sub_grid_paths:
                        sub_grid_paths[out_dir] = {
                            "l0": l0,
                            "l1": l1,
                            "name": f"slice_{i}_{j}",
                        }

    def _call_mpr(self, namelist):
        """Call the mpr executable with the given namelist and parameter file."""
        # tmpdir = Path.cwd()
        # os.chdir(self.work_dir)
        command = f"{self.mpr_executable} -c {namelist}"  # -p {self.parameter_file}"
        logger.info(f"Running mPR with: {command}")

        p = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
        try:
            data, error_data = p.communicate()
            if error_data:
                logger.error(f"Failed with STDERR {error_data}")
        except TimeoutExpired:
            logger.error("Timeout expired")
            p.kill()
        # finally:
            # os.chd    ir(tmpdir)

    def _merge_restart_files(self):
        logger.info("Merging restart files")
        logger.info("Not implemented yet")

    def _delete_temp_files(self):
        logger.info("Deleting temporary files")
        # remove all temporary files meaning all files in the morph_files of each subgrid
        for subgrid in self.subgrids:
            for file_path in subgrid.morph_files.get_files_as_list():
                if isinstance(file_path, list):
                    for f in file_path:
                        f.unlink()
                else:
                    file_path.unlink()

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
        if self.split_grid:
            logger.info("grid will be split and processed in parallel")
            self._split_grid()
            for subgrid in self.subgrids:  # parallelize this
                logger.debug(
                    f"Processing subgrid {subgrid.name}, {subgrid.path}, {subgrid.l0.lon_min}, {subgrid.l0.lon_max}, {subgrid.l0.lat_min}, {subgrid.l0.lat_max}"
                )
                nml = self._write_grid_namelist(subgrid)
                self._call_mpr(nml)
            self._merge_restart_files()
            if self.clean_temp_files:
                self._delete_temp_files()
        else:
            logger.info("grid will be processed as a whole")
            nml = self._write_grid_namelist(self.grid)
            self._call_mpr(nml)
        logger.info("Restart file created")
