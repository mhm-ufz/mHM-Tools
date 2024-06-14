from pathlib import Path
import multiprocessing as mp
import pathlib
from subprocess import Popen, PIPE, TimeoutExpired
import os

class MorphFiles:
    def __init__(self, filepath: Path = None, 
                 land_cover = None, bulk_density = None, 
                 sand_content = None, clay_content = None, 
                 slope = None, lai = None, aspect = None, 
                 geology = None):
        self.land_cover = None
        self.bulk_density = None
        self.sand_content = None
        self.clay_content = None
        self.slope = None
        self.lai = None
        self.aspect = None
        self.geology = None
        self.member_dir = {'land_cover': self.land_cover,
                           'bulk_density': self.bulk_density,
                           'sand_content': self.sand_content,
                           'clay_content': self.clay_content,
                           'slope': self.slope,
                           'lai': self.lai,
                           'aspect': self.aspect,
                           'geology': self.geology}
        self.member_key_synonyms = {'bulk_density': ['BLDFIE'], 'sand_content': ['SNDPPT'], 'clay_content': ['CLYPPT']},
        if filepath is not None:
            self.read_files(filepath)
    
    def read_files(self, filepath: Path):
        for key in self.member_dir.keys():
            key_files = filepath.glob(f'{key}*.nc')
            if len(key_files) == 0:
                for synonym in self.member_key_synonyms[key]:
                    key_files = filepath.glob(f'{synonym}*.nc')
                    if len(key_files) != 0:
                        break
            if len(key_files) != 1:
                self.member_dir[key] = [f for f in key_files]
            else:
                self.member_dir[key] = key_files[0]
    
    def get_files_as_list(self):
        file_list = []
        for key, value in self.member_dir.items():
            if isinstance(value, list):
                file_list.extend(value)
            else:
                file_list.append(value)
        return file_list
    
    def get_files_as_dir(self):
        return self.member_dir.values()
    
class LatLon:
    def __init__(self, lat_min = None, lon_min = None, lat_max = None, lon_max = None, resolution = None):
        self.lat_min = lat_min
        self.lon_min = lon_min
        self.lat_max = lat_max
        self.lon_max = lon_max
        self.resolution = resolution
    
    def get_n_lat(self):
        return int((self.lat_max - self.lat_min) / self.resolution)

    def get_n_lon(self):
        return int((self.lon_max - self.lon_min) / self.resolution)
        

class Domain:
    def __init__(self, file_path: Path, name = None, latlon_file = None):
        self.morph_files = MorphFiles(filepath=file_path)
        self.name = name
        self.path = file_path
        self.l0 = LatLon()
        self.l1 = LatLon()

        if latlon_file is not None:
            self.read_latlon(latlon_file)

    def read_latlon(self, latlon_file: Path):
        pass # read latlon file and set self.l0 and self.l1


class MHMRestartFile:
    def __init__(self, input_file_path: Path, nml_template: Path, output_path: Path, latlon_file: Path = None, split_domain = False, clean_temp_files = True, increment_l1 = 2):
        self.nml_template = nml_template
        self.output_path = output_path
        self.domain = Domain(input_file_path, latlon_file=latlon_file)
        self.subdomains = [] # list of Domain objects
        self.split_domain = split_domain
        self.clean_temp_files = clean_temp_files
        self.mpr_executable = None
        self.parameter_file = None
        self.work_dir = None
        self.increment_l1 = increment_l1
        self.increment_l0 = self.increment_l1 * self.domain.l0.resolution / self.domain.l1.resolution

    def _create_namelist(self, replace_dict, template, out_file_path, overwrite=False):
        if not out_file_path.exists() or overwrite:
            with open(template) as f:
                nml = f.read()
            for replace_key, replace_value in replace_dict.items():
                nml = nml.replace(replace_key, replace_value)
            with open(out_file_path, 'w') as f:
                f.write(nml)
        return out_file_path 

    def _write_domain_namelist(self, domain: Domain):
        replace_dict = {
            '${slicei_j}': domain.name,
            '${output_file}': domain.path / f'output_{domain.name}.nc',

            '${lon_high_start}': f'{domain.l0.lon_min:.3f}',
            '${lon_high_res}': f'{domain.l0.resolution:.3f}',
            '${lon_high_n}': f'{domain.l0.get_n_lon()}',

            '${lat_high_start}': f'{domain.l0.lat_min:.3f}',
            '${lat_high_res}': f'{-domain.l0.resolution:.3f}',
            '${lat_high_n}': f'{domain.l0.get_n_lat()}',

            '${lon_low_start}': f'{domain.l1.lon_min:.2f}',
            '${lon_low_res}': f'{domain.l1.resolution:.2f}',
            '${lon_low_n}': f'{domain.l1.get_n_lon()}',

            '${lat_low_start}': f'{domain.l1.lat_min:.2f}',
            '${lat_low_res}': f'{-domain.l1.resolution:.2f}',
            '${lat_low_n}': f'{domain.l1.get_n_lat()}',
        }
        return self._create_namelist(replace_dict, self.nml_template, domain.output_path / 'mpr.nml')

    def _split_domain(self): # INCREMENT, X_COUNT, Y_COUNT, OUTPUT_FOLDER
        """
        Split the domain into subdomains and write them to disk
        
        Subdomains are subsets of the original domain with a size of increment x increment grid cells.
        """
        sub_domain_paths = []
        for file_path in self.domain.morph_files.get_files_as_list():
            ds = xr.open_dataset(file_path)
            for i, isel_start in enumerate(range(0, self.domain.l0.get_n_lon(), self.increment_l0)):
                for j, jsel_start in enumerate(range(0, self.domain.l0.get_n_lon(), self.increment_l0)):
                    out_path = Path(self.output_path) / f'slice_{i}_{j}'
                    out_path.mkdir(parents=True, exist_ok=True)
                    # mkdir outpath / slice{i}_{j}
                    
                    file_path = out_path / f'{file_path.name}.nc'
                    ds.isel(longitude=slice(isel_start, isel_start + self.increment_l0),
                            latitude=slice(jsel_start, jsel_start + self.increment_l0)).\
                        to_netcdf(out_path)
                    if out_path not in sub_domain_paths:
                        sub_domain_paths.append(out_path)
        self.subdomains = [Domain(p, p.name) for p in sub_domain_paths]
        
    def _call_mpr(self, namelist):
        tmpdir = os.getcwd()
        os.chdir(self.work_dir)
        command = f'{self.mpr_executable} -c {namelist} -p {self.parameter_file}'
        print('Running command: %s' % command, flush=True)

        p = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
        try:
            data, error_data = p.communicate()
            if error_data:
                print('Failed with STDERR %s', error_data, flush=True)
        except TimeoutExpired:
            p.kill()
        finally:
            os.chdir(tmpdir)

    def _merge_restart_files(self):
        pass

    def clean_temp_files(self):
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
            for subdomain in self.subdomains: # parallelize this
                nml = self._write_domain_namelist(subdomain)
                self._call_mpr(subdomain, nml)
            self._merge_restart_files()
            if self.clean_temp_files:
                self.clean_temp_files()
        else:
            nml = self._write_domain_namelist(self.domain)
            self._call_mpr(self.domain, nml)
    

if __name__=='__main__':
    split_domain = False
    restart_creator = MHMRestartFile(input_file_path=Path('/work-local/ottor/ulysses/'), # path to all the input files
                                     output_path=Path('/work-local/ottor/ulysses/MPR_namelists'), # path where the output files will be written
                                     nml_template=Path('/home/ottor/projects/htessel_mpr/05_mhm_global_param/nml/mpr_mhm_global_sel3_slice_template.nml'), # path to the namelist template
                                     latlon_file=Path('/work-local/ottor/ulysses/latlon.nc'), # path to the latlon file maybe not needed
                                     split_domain=split_domain, # split the domain into subdomains or not
                                     clean_temp_files=True) # clean temporary files or not
    restart_creator.create_restart_file()
    

