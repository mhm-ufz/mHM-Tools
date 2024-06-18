from pathlib import Path
import multiprocessing as mp
from subprocess import Popen, PIPE, TimeoutExpired
import os
import xarray as xr
import logging


class MorphFiles:
    def __init__(
        self,
        filepath: Path = None,
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
        # self.member_dir = {
        #     "land_cover": self.land_cover,
        #     "bulk_density": self.bulk_density,
        #     "sand_content": self.sand_content,
        #     "clay_content": self.clay_content,
        #     "slope": self.slope,
        #     "lai": self.lai,
        #     "aspect": self.aspect,
        #     "geology": self.geology,
        # }
        
        if filepath is not None:
            self.read_files(filepath)

    def read_files(self, filepath: Path, overwrite=False):
        member_key_synonyms = {
                "bulk_density": ["BLDFIE"],
                "sand_content": ["SNDPPT"],
                "clay_content": ["CLYPPT"],
            }
        for key in self.__dict__.keys():
            if not overwrite and self.__dict__.get(key, None) is not None:
                continue
            key_files = list(filepath.glob(f"{key}*.nc"))
            if len(key_files) == 0:
                if not key in member_key_synonyms.keys():
                    continue # should raise an error
                for synonym in member_key_synonyms[key]:
                    key_files = list(filepath.glob(f"{synonym}*.nc"))
                    if len(key_files) != 0:
                        break
            if len(key_files) != 1:
                self.__dict__[key] = [f if f.is_file() else None for f in key_files]
            else:
                self.__dict__[key] = key_files[0] if key_files[0].is_file() else None
    
    def get_file(self, key):
        return self.__dict__.get(key, None)

    def get_files_as_list(self):
        file_list = []
        for value in self.__dict__.values():
            if isinstance(value, list):
                file_list.extend(value)
            else:
                file_list.append(value)
        return file_list

    def get_files_as_dir(self):
        return self.__dict__


class LatLon:
    def __init__(
        self, lat_min=None, lon_min=None, lat_max=None, lon_max=None, resolution=None
    ):
        self.lat_min = lat_min
        self.lon_min = lon_min
        self.lat_max = lat_max
        self.lon_max = lon_max
        self.resolution = resolution

    def get_n_lat(self):
        return int((self.lat_max - self.lat_min) / self.resolution)

    def get_n_lon(self):
        return int((self.lon_max - self.lon_min) / self.resolution)

    def is_fully_defined(self):
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
    def __init__(self, file_path: Path, name=None, latlon_file=None, l0: LatLon=None, l1: LatLon = None):
        self.morph_files = MorphFiles(filepath=file_path)
        self.name = name
        self.path = file_path
        self.l0 = l0
        self.l1 = l1

        if (self.l0 is None or not self.l0.is_fully_defined() or self.l1 is None or not self.l1.is_fully_defined()) and latlon_file is not None:
            self.read_latlon(latlon_file)

    def read_latlon(self, latlon_file: Path):
        with xr.open_dataset(latlon_file) as ds:
            x0 = ds['xc_l0'].to_numpy()
            y0 = ds['yc_l0'].to_numpy()
            self.l0 = LatLon(lon_min=x0.min(), lon_max=x0.max(), lat_min=y0.min(), lat_max=y0.max(), resolution=x0[1] - x0[0])
            x1 = ds['xc'].to_numpy()
            y1 = ds['yc'].to_numpy()
            self.l1 = LatLon(lon_min=x1.min(), lon_max=x1.max(), lat_min=y1.min(), lat_max=y1.max(), resolution=x1[1] - x1[0])

        pass  # read latlon file and set self.l0 and self.l1

    def read_morph_files(self):
        self.morph_files.read_files(self.path)


class MHMRestartFile:
    logging.basicConfig(format="%(asctime)s - %(levelname)-8s - %(message)s")
    logger = logging.getLogger(__name__)
    def __init__(
        self,
        input_file_path: Path,
        nml_template: Path,
        output_path: Path,
        latlon_file: Path = None,
        split_domain=False,
        clean_temp_files=True,
        increment_l1=2,
        lon_min_target_grid=None,
        lon_max_target_grid=None,
        lat_min_target_grid=None,
        lat_max_target_grid=None,
        l0_resolution=None,
        l1_resolution=None,
        log_level=logging.DEBUG
    ):
        self.logger.setLevel(logging.DEBUG)
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
        self.domain = Domain(input_file_path, latlon_file=latlon_file, l0=domain_latlon_l0, l1=domain_latlon_l1)
        self.subdomains = []  # list of Domain objects
        self.split_domain = split_domain
        self.clean_temp_files = clean_temp_files
        self.mpr_executable = None
        self.parameter_file = None
        self.work_dir = None
        self.increment_l1 = increment_l1
        self.increment_l0 = int(
            self.increment_l1 * self.domain.l1.resolution / self.domain.l0.resolution
        ) if self.increment_l1 is not None else None

    def _create_namelist(self, replace_dict, template, out_file_path, overwrite=False):
        if not out_file_path.exists() or overwrite:
            with open(template) as f:
                nml = f.read()
            for replace_key, replace_value in replace_dict.items():
                nml = nml.replace(replace_key, replace_value)
            with open(out_file_path, "w") as f:
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
            "${land_cover}": domain.morph_files.land_cover, # this should be a list but the template only has one
            
        }
        return self._create_namelist(
            replace_dict, self.nml_template, domain.output_path / "mpr.nml"
        )

    def _split_domain(self): # has do addapted to different file types not just .nc
        """
        Split the domain into subdomains and write them to disk

        Subdomains are subsets of the original domain with a size of increment x increment grid cells.
        """
        if self.increment_l0 is None:
            raise ValueError("Increment for splitting domains is not set")
        sub_domain_paths = {}
        for file_path in self.domain.morph_files.get_files_as_list():
            if file_path is None:
                continue # should raise error
            self.logger.info(f"Splitting {file_path}")
            ds = xr.open_dataset(file_path)
            self.logger.debug(f'0, {self.domain.l0.get_n_lon()}, {self.increment_l0}')
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
                    except Exception as e:
                        self.logger.debug(f"{jsel_start}, {jsel_start + self.increment_l0}")
                        self.logger.debug(ds_cut['latitude'].values)
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
                        resolution = self.domain.l1.resolution
                    )
                    if out_dir not in sub_domain_paths.keys():
                        sub_domain_paths[out_dir] = {'l0':l0, 'l1':l1, 'name':f"slice_{i}_{j}"}
            self.logger.debug(f"Splitting {file_path} done")
        self.subdomains = [Domain(file_path=k, **v) for k, v in sub_domain_paths.items()]

    def _call_mpr(self, namelist):
        tmpdir = os.getcwd()
        os.chdir(self.work_dir)
        command = f"{self.mpr_executable} -c {namelist} -p {self.parameter_file}"
        print(f"Running command: {command}", flush=True)

        p = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
        try:
            data, error_data = p.communicate()
            if error_data:
                print("Failed with STDERR %s", error_data, flush=True)
        except TimeoutExpired:
            p.kill()
        finally:
            os.chdir(tmpdir)

    def _merge_restart_files(self):
        pass

    def _clean_temp_files(self):
        # remove all temporary files meaning all files in the morph_files of each subdomain
        for subdomain in self.subdomains:
            for file_type, file_path in subdomain.morph_files.get_files_as_dir():
                if isinstance(file_path, list):
                    for f in file_path:
                        f.unlink()
                else:
                    file_path.unlink()

    def create_restart_file(self):
        if self.split_domain:
            self._split_domain()
            for subdomain in self.subdomains:  # parallelize this
                nml = self._write_domain_namelist(subdomain)
                self._call_mpr(subdomain, nml)
            self._merge_restart_files()
            if self.clean_temp_files:
                self._clean_temp_files()
        else:
            nml = self._write_domain_namelist(self.domain)
            self._call_mpr(self.domain, nml)


if __name__ == "__main__":
    split_domain = False
    restart_creator = MHMRestartFile(
        input_file_path=Path(
            "/work-local/ottor/ulysses/"
        ),  # path to all the input files
        output_path=Path(
            "/work-local/ottor/ulysses/MPR_namelists"
        ),  # path where the output files will be written
        nml_template=Path(
            "/home/ottor/projects/htessel_mpr/05_mhm_global_param/nml/mpr_mhm_global_sel3_slice_template.nml"
        ),  # path to the namelist template
        latlon_file=Path(
            "/work-local/ottor/ulysses/latlon.nc"
        ),  # path to the latlon file maybe not needed
        split_domain=split_domain,  # split the domain into subdomains or not
        clean_temp_files=True,
    )  # clean temporary files or not

    restart_creator.create_restart_file()
