"""Create the mHM restart file."""

import logging
import os
from pathlib import Path
from subprocess import PIPE, Popen, TimeoutExpired

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
        }
        if type(filepath) is not Path:
            filepath = Path(filepath)
        logger.info(f'reading morph files from {filepath}')
        for key in self.__dict__:
            if not overwrite and self.__dict__.get(key, None) is not None:
                continue
            key_files = list(filepath.glob(f"{key}*.nc"))
            if len(key_files) == 0:
                if key not in member_key_synonyms:
                    continue  # should raise an error
                for synonym in member_key_synonyms[key]:
                    key_files = list(filepath.glob(f"{synonym}*.nc"))
                    if len(key_files) != 0:
                        break
            if len(key_files) != 1:
                self.__dict__[key] = [f if f.is_file() else None for f in key_files]
            else:
                self.__dict__[key] = key_files[0] if key_files[0].is_file() else None

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
        return int((self.lat_max - self.lat_min) / self.resolution)

    def get_n_lon(self):
        """
        Calculate the number of longitude points based on the given longitude range and resolution.

        Returns
        -------
            int: The number of longitude points.
        """
        return int((self.lon_max - self.lon_min) / self.resolution)

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


class Domain:
    """
    Represents a geographical area for wich morphological data exists.

    This domain is used to run the mPR model. It does not need to contain a whole catchment, but can be a subset of it or multiple catchments at once.

    Attributes
    ----------
        file_path (Path): The file path of the domain.
        name (str): The name of the domain.
        latlon_file (str): The file path of the latlon file.
        l0 (LatLon): The lower-left corner of the domain.
        l1 (LatLon): The upper-right corner of the domain.
    """

    def __init__(
        self,
        file_path: Path,
        name=None,
        latlon_file=None,
        l0: LatLon = None,
        l1: LatLon = None,
    ):
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
        Read the latlon file and sets the lower-left (l0) and upper-right (l1) corners of the domain as well as the resolution.

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

    This class provides methods to split the domain (if necessary), write the domain namelist,
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
    split_domain : bool
        Whether to split the domain into subdomains.
    clean_temp_files : bool
        Whether to clean temporary files.
    increment_l1 : int
        The increment for splitting the domain in number of coarse grid (l1) cells.
    lon_min_target_grid : Optional[float]
        The minimum longitude of the target grid.
    lon_max_target_grid : Optional[float]
        The maximum longitude of the target grid.
    lat_min_target_grid : Optional[float]
        The minimum latitude of the target grid.

    Methods
    -------
    split_domain_if_necessary()
        Split the domain into subdomains if necessary.
    write_domain_namelist()
        Write the domain namelist file.
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
        latlon_file=None,
        split_domain=False,
        clean_temp_files=True,
        increment_l1=2,
        lon_min_target_grid=None,
        lon_max_target_grid=None,
        lat_min_target_grid=None,
        lat_max_target_grid=None,
        l0_resolution=None,
        l1_resolution=None,
        log_level=logging.DEBUG,
        mpr_executable=None
    ):
        logger.setLevel(log_level)
        self.nml_template = nml_template
        self.output_path = output_path
        domain_latlon_l0 = LatLon(
            lat_min=lat_min_target_grid,
            lon_min=lon_min_target_grid,
            lat_max=lat_max_target_grid,
            lon_max=lon_max_target_grid,
            resolution=l0_resolution,
        )
        domain_latlon_l1 = LatLon(
            lat_min=lat_min_target_grid,
            lon_min=lon_min_target_grid,
            lat_max=lat_max_target_grid,
            lon_max=lon_max_target_grid,
            resolution=l1_resolution,
        )
        self.domain = Domain(
            input_file_path,
            latlon_file=latlon_file,
            l0=domain_latlon_l0,
            l1=domain_latlon_l1,
        )
        self.subdomains = []  # list of Domain objects
        self.split_domain = split_domain
        self.clean_temp_files = clean_temp_files
        self.mpr_executable = mpr_executable
        self.parameter_file = None
        self.work_dir = None
        self.increment_l1 = increment_l1
        self.increment_l0 = (
            int(
                self.increment_l1
                * self.domain.l1.resolution
                / self.domain.l0.resolution
            )
            if self.increment_l1 is not None
            else None
        )

    def _create_namelist(self, replace_dict, template, out_file_path, overwrite=False):
        if not out_file_path.exists() or overwrite:
            with template.open("r") as f:
                nml = f.read()
            for replace_key, replace_value in replace_dict.items():
                nml = nml.replace(replace_key, replace_value)
            with out_file_path.open("w") as f:
                f.write(nml)
        return out_file_path

    def _write_domain_namelist(self, domain: Domain):
        replace_dict = {
            "${slicei_j}": domain.name,
            "${output_file}": domain.path / f"output_{domain.name}.nc",
            "${lon_high_start}": f"{domain.l0.lon_min:.3f}",
            "${lon_high_res}": f"{domain.l0.resolution:.3f}",
            "${lon_high_n}": f"{domain.l0.get_n_lon()}",
            "${lat_high_start}": f"{domain.l0.lat_min:.3f}",
            "${lat_high_res}": f"{-domain.l0.resolution:.3f}",
            "${lat_high_n}": f"{domain.l0.get_n_lat()}",
            "${lon_low_start}": f"{domain.l1.lon_min:.2f}",
            "${lon_low_res}": f"{domain.l1.resolution:.2f}",
            "${lon_low_n}": f"{domain.l1.get_n_lon()}",
            "${lat_low_start}": f"{domain.l1.lat_min:.2f}",
            "${lat_low_res}": f"{-domain.l1.resolution:.2f}",
            "${lat_low_n}": f"{domain.l1.get_n_lat()}",
            "${bulk_density}": domain.morph_files.bulk_density,
            "${sand_content}": domain.morph_files.sand_content,
            "${clay_content}": domain.morph_files.clay_content,
            "${slope}": domain.morph_files.slope,
            "${lai}": domain.morph_files.lai,
            "${aspect}": domain.morph_files.aspect,
            "${geology}": domain.morph_files.geology,
            "${land_cover}": domain.morph_files.land_cover,  # this should be a list but the template only has one
        }
        return self._create_namelist(
            replace_dict, self.nml_template, domain.output_path / "mpr.nml"
        )

    def _split_domain(self):  # has do addapted to different file types not just .nc
        """
        Split the domain into subdomains and write them to disk.

        Subdomains are subsets of the original domain with a size of increment x increment grid cells.
        """
        if self.increment_l0 is None:
            error_message = "Increment for splitting domains is not set"
            raise ValueError(error_message)
        sub_domain_paths = {}
        for file_path in self.domain.morph_files.get_files_as_list():
            if file_path is None:
                continue  # should raise error
            logger.info(f"Splitting {file_path}")
            ds = xr.open_dataset(file_path)
            logger.debug(f"0, {self.domain.l0.get_n_lon()}, {self.increment_l0}")
            for i, isel_start in enumerate(
                range(0, self.domain.l0.get_n_lon(), self.increment_l0)
            ):
                for j, jsel_start in enumerate(
                    range(0, self.domain.l0.get_n_lat(), self.increment_l0)
                ):
                    out_dir = Path(self.output_path) / f"slice_{i}_{j}"
                    out_dir.mkdir(parents=True, exist_ok=True)

                    out_path = out_dir / f"{file_path.stem}.nc"
                    ds_cut = ds.isel(
                        longitude=slice(isel_start, isel_start + self.increment_l0),
                        latitude=slice(jsel_start, jsel_start + self.increment_l0),
                    )
                    try:
                        ds_cut.to_netcdf(out_path)
                    except Exception:
                        logger.debug(
                            f"{jsel_start}, {jsel_start + self.increment_l0}"
                        )
                        logger.debug(ds_cut["latitude"].values)
                        return
                    l0 = LatLon(
                        lon_min=ds_cut.longitude.min(),
                        lon_max=ds_cut.longitude.max(),
                        lat_min=ds_cut.latitude.min(),
                        lat_max=ds_cut.latitude.max(),
                        resolution=self.domain.l0.resolution,
                    )
                    l1 = LatLon(
                        lon_min=ds_cut.longitude.min(),
                        lon_max=ds_cut.longitude.max(),
                        lat_min=ds_cut.latitude.min(),
                        lat_max=ds_cut.latitude.max(),
                        resolution=self.domain.l1.resolution,
                    )
                    if out_dir not in sub_domain_paths:
                        sub_domain_paths[out_dir] = {
                            "l0": l0,
                            "l1": l1,
                            "name": f"slice_{i}_{j}",
                        }
            logger.debug(f"Splitting {file_path} done")
        self.subdomains = [
            Domain(file_path=k, **v) for k, v in sub_domain_paths.items()
        ]

    def _call_mpr(self, namelist):
        """Call the mpr executable with the given namelist and parameter file."""
        tmpdir = Path.cwd()
        os.chdir(self.work_dir)
        command = f"{self.mpr_executable} -c {namelist} -p {self.parameter_file}"
        logger.info(f"Running mPR with: {command}")

        p = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
        try:
            data, error_data = p.communicate()
            if error_data:
                logger.error(f"Failed with STDERR {error_data}")
        except TimeoutExpired:
            p.kill()
        finally:
            os.chdir(tmpdir)

    def _merge_restart_files(self):
        logger.info("Merging restart files")
        logger.info("Not implemented yet")

    def _delete_temp_files(self):
        logger.info("Deleting temporary files")
        # remove all temporary files meaning all files in the morph_files of each subdomain
        for subdomain in self.subdomains:
            for file_path in subdomain.morph_files.get_files_as_list():
                if isinstance(file_path, list):
                    for f in file_path:
                        f.unlink()
                else:
                    file_path.unlink()

    def create_restart_file(self):
        """
        Create a restart file for the MHM model.

        This method creates a restart file by splitting the domain (if necessary),
        writing the domain namelist, calling the mpr executable, merging the restart
        files (if applicable), and deleting temporary files (if specified).

        Returns
        -------
            None

        Raises
        ------
            Any exceptions that occur during the execution of the method.
        """
        logger.info("Creating restart file")
        if self.split_domain:
            self._split_domain()
            for subdomain in self.subdomains:  # parallelize this
                nml = self._write_domain_namelist(subdomain)
                self._call_mpr(nml)
            self._merge_restart_files()
            if self.clean_temp_files:
                self._delete_temp_files()
        else:
            nml = self._write_domain_namelist(self.domain)
            self._call_mpr(nml)
        logger.info("Restart file created")
