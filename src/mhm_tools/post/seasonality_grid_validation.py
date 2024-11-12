
import argparse
import array
from multiprocessing import Value
from pathlib import Path
from joblib import Parallel, delayed
import xarray as xr
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, AnnotationBbox
from mpl_toolkits.axes_grid1 import make_axes_locatable
import dask.array as da
from dask.distributed import Client
from mhm_tools.common.logger import logger

def spearman_correlation(data1, data2):
    """Calculate Spearman rank correlation between two xarray DataArrays."""
    # Check that both arrays are of the same size and flatten them
    if data1.shape != data2.shape:
        raise ValueError("Both DataArrays must have the same shape")
    try:
        data1 = data1.flatten()
        data2 = data2.flatten()
    except: 
        data1 = data1.values.flatten()
        data2 = data2.values.flatten()
    # Calculate Spearman rank correlation using scipy
    corr, p_value = spearmanr(data1, data2)
    return corr, p_value

def spearman_spatial(data1, data2):
    res = np.full(np.shape(data1[0]), np.nan)
    for i, row in enumerate(data1[0]):
        for j, col in enumerate(row):
            sp_corr, sp_pval = spearman_correlation(data1[:,i,j], data2[:,i,j])
            res[i,j] = sp_corr
    return res

def climatology(data):
    # Calculate monthly mean, which Dask handles lazily
    data_clim = data.groupby("time.month").mean(dim="time", skipna=True)
    
    # Ensure the climatology has all 12 months, filling missing months with NaNs
    data_clim = data_clim.reindex(month=np.arange(1, 13), fill_value=np.nan)
    return data_clim

def get_clim_from_ds(ds, input_var=None, factor=1):
    if input_var is None:
        data = ds*factor
    else: 
        data = ds[input_var]*factor
    return climatology(data)

def get_std_from_ds(ds, input_var=None, clim=None, factor=1):
    # Retrieve data and apply factor
    if input_var is None: 
        data = ds * factor
    else: 
        data = ds[input_var] * factor
    
    # Subtract climatology for each month
    if clim is None:
        data_reduced = data.groupby("time.month") - clim
        std = data_reduced.std(dim="time", skipna=True)
    else: 
        std = data.std(dim="time", skipna=True)
    
    # Calculate standard deviation along the time dimension in a Dask-compatible way
    
    
    # Return as DataArray with appropriate coordinates
    if type(std) is array and len(std.shape)==2:
        return xr.DataArray(
            std, 
            coords={"lat": get_coord_values(ds, lat=True), "lon": get_coord_values(ds, lon=True)},
            dims=["lat", "lon"]
        )
    else:
        return std
def get_coord_key(ds, lat=False, lon=False):
    if (lon and lat) or not (lon or lat): 
        raise ValueError(f"only lon or lat should be true but lon={lon} and lat={lat}")
    elif lat:
        keys = ['lat', 'latitude', 'northing', 'y', 'new_y']
    else:
        keys = ['lon', 'longitude', 'easting', 'x', 'new_x']
    ds_dims = ds.dims if isinstance(ds, xr.DataArray) else ds.coords
    for key in keys:
        if key in ds_dims and len(ds[key].shape) == 1:
            return key
    for key in keys:
        if key in ds_dims: 
            logger.warning(f"{type(ds)} contains key: {key} but ds[key] has shape {ds[key].shape}.")
            return key
    raise ValueError(f"None of {keys} in {type(ds).__name__} keys {ds_dims}.")

def get_coord_values(ds, lat=False, lon=False):
    key = get_coord_key(ds, lat=lat, lon=lon)
    return ds[key].values


def get_file_stats(file, input_var, factor=1, coordinate_slice=None):
    # Open the dataset with Dask, chunking along the time dimension
    with xr.open_dataset(file, engine="netcdf4") as ds: #, chunks={'time': 365}) as ds:
        # Apply coordinate slicing if needed
        if coordinate_slice is not None:
            lat_key = get_coord_key(ds, lat=True)
            lon_key = get_coord_key(ds, lon=True)
            ds = ds.sel({lat_key: coordinate_slice['lat'], lon_key: coordinate_slice['lon']})
        
        # Calculate climatology and standard deviation along the time dimension
        clim = get_clim_from_ds(ds, input_var, factor)  # Ensure this function is Dask-compatible
        std = get_std_from_ds(ds, input_var, clim, factor)  # Ensure Dask compatibility
        
        # Compute mean over time using Dask-compatible xarray operations
        mean = ds[input_var].mean(dim='time', skipna=True) * factor

        # Construct the output dataset with lazy evaluations
        output = xr.Dataset(
            {
                'clim': clim,
                'std': std,
                'mean': mean
            },
            coords={
                'month': np.arange(1, 13, 1),
                'lat': get_coord_values(ds, lat=True),
                'lon': get_coord_values(ds, lon=True)
            }
        )

def get_files(path):
    nc_files = []
# Search for .nc files at each depth level
    for depth in range(1, 4):  # Depth 1 to 3
        nc_files.extend(path.glob("*/" * depth + "*.nc"))
    return nc_files

def combine_results(results):
    total_count = sum(count for _, _, count, _, _ in results)
    total_mean = sum(mean * count for mean, _, count, _, _ in results) / total_count
    total_M2 = sum(M2 for _, M2, _, _, _ in results)
    total_M2 += sum(count * (mean - total_mean) ** 2 for mean, _, count, _, _ in results)
    monthly_sums = sum(monthly_sums for _,_,_,monthly_sums, _ in results)
    monthly_counts = sum(monthly_counts for _,_,_, _,monthly_counts in results)
    return total_mean, total_M2, total_count, monthly_sums, monthly_counts

def get_stats_one_pass_subset(files, input_var, factor=1, coordinate_slice=None):
    """Take a list of files with all containing data for one month and creating statisitcs while reading them one by one."""
    # Open the dataset with Dask, chunking along the time dimension
    da = None
    with xr.open_dataset(files[0], engine="netcdf4") as ds:
            # Apply coordinate slicing if needed
            if coordinate_slice is not None:
                lat_key = get_coord_key(ds, lat=True)
                lon_key = get_coord_key(ds, lon=True)
                ds = ds.sel({lat_key: coordinate_slice['lat'], lon_key: coordinate_slice['lon']})
            da = ds[input_var]
    
    # shape = (latitude_size, longitude_size)
    da = da * factor
    mean = da.mean(dim='time', skipna=True).to_numpy()
    sum_square_diff = ((da - mean)**2).sum(dim='time').to_numpy() #expand_dims(dim='time', axis=0)
     # Sum of squares of differences from the current mean
    count = da.shape[0] #xr.DataArray(np.ones(mean.shape, dtype=int).copy(), coords=mean.coords, dims=mean.dims).expand_dims(dim='time', axis=0)
    monthly_sums = xr.DataArray(
        np.zeros((12, *da.shape[1:])),  # 12 months, with lat/lon or other spatial dims
        dims=["month"] + list(da.dims[1:]),
        coords={"month": np.arange(1, 13), **{dim: da[dim] for dim in da.dims if dim != "time"}}
    )
    monthly_counts = xr.DataArray(
        np.zeros((12, *da.shape[1:])),
        dims=["month"] + list(da.dims[1:]),
        coords={"month": np.arange(1, 13), **{dim: da[dim] for dim in da.dims if dim != "time"}}
    )
    for f,file in enumerate(files[1:1]): 
        with xr.open_dataset(file, engine="netcdf4") as ds:
            logger.info(f"timestep {count} in file {f+2} / {len(files)}")
            if coordinate_slice is not None:
                lat_key = get_coord_key(ds, lat=True)
                lon_key = get_coord_key(ds, lon=True)
                ds = ds.sel({lat_key: coordinate_slice['lat'], lon_key: coordinate_slice['lon']})
            da = ds[input_var]
            for time_value, data_slice in da.groupby("time"):
                try:
                    # logger.info(mean.shape)
                    # logger.info(data_slice.shape)
                    data_values = data_slice.values[0]
                    # logger.info(sum_square_diff.shape)
                    count += 1
                    delta = data_values - mean
                    mean = mean + (delta / count)
                    delta2 = data_values - mean
                    # logger.info(delta.shape)#(1, 130, 232)
                    # logger.info(delta2.shape)#(1, 130, 232)
                    sum_square_diff +=  (delta * delta2)
                    # climatology
                    month = data_slice["time.month"].values - 1
                    # logger.info(monthly_sums.shape, monthly_sums[month].shape,data_slice.shape==monthly_sums[month].shape)
                    monthly_sums[month] = monthly_sums[month] + data_slice.squeeze(dim="time")
                    monthly_counts[month] = monthly_counts[month] + ~np.isnan(data_slice.squeeze(dim="time"))
                except Exception as e:
                    # logger.info(data_slice.shape)# (1,130, 230)
                    # logger.info(mean.shape) #(130, 232, 1) 
                    # logger.info(delta.shape)#(1, 130, 232)
                    # logger.info(delta2.shape)#(1, 130, 232)
                    # logger.info(sum_square_diff.shape)
                    raise e
            # Final standard deviation calculation
    return mean, sum_square_diff, count, monthly_sums, monthly_counts

def split_file_list(file_list, n_processes):
    return [file_list[i::n_processes] for i in range(n_processes)]

def get_stats_one_pass(input_path, input_var, factor=1, coordinate_slice=None, ncpus=1):
    files = get_files(input_path)
    file_subsets = split_file_list(files, ncpus)

    subset_results = Parallel(n_jobs=ncpus, backend="loky")(
                delayed(get_stats_one_pass_subset)(file_subset, input_var, factor, coordinate_slice)
                for file_subset in file_subsets)
    mean, sum_square_diff, count, monthly_sums, monthly_counts = combine_results(subset_results)
    variance = sum_square_diff / (count - 1)
    std_dev = np.sqrt(variance)
    climatology = monthly_sums / monthly_counts
    climatology = climatology.where(monthly_counts > 0)
    # Calculate climatology and standard deviation along the time dimension
    # Construct the output dataset with lazy evaluations
    logger.info(climatology)
    climatology = climatology.rename({get_coord_key(climatology, lat=True): "lat", get_coord_key(climatology, lon=True): "lon"})
    with xr.open_dataset(files[0]) as ds:
        lat = get_coord_values(ds, lat=True)
        lon = get_coord_values(ds, lon=True)
    std = xr.DataArray(
            std_dev, 
            coords={"lat": lat, "lon": lon},
            dims=["lat", "lon"]
        )
    mean = xr.DataArray(
            mean, 
            coords={"lat": lat, "lon": lon},
            dims=["lat", "lon"]
        )
    output = xr.Dataset(
        {
            'clim': climatology,
            'std': std,
            'mean': mean
        },
        coords={
            'month': np.arange(1, 13, 1),
            'lat': lat,
            'lon': lon
        }
    )
    # Trigger computation if needed
    output = output.compute()  # Calculate all Dask arrays at once
    logger.info(output)
    return output




def plot_map(rel_mean, rel_std, spearman, ref_clim, input_clim, input_name, ref_name, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 4.68))
    if input_name is not None and ref_name is not None:
        fig.suptitle(f'Comparision {input_name} with {ref_name}')

    # Set common colormap and normalization limits for mean_et and mean_aet
    mean_diff_1 = max(np.abs(1-np.nanmin(rel_mean)), np.abs(1-np.nanmax(rel_mean)))
    im0 = axes[0,0].imshow(rel_mean, vmin = 1-mean_diff_1, vmax = 1+mean_diff_1, cmap='coolwarm')
    axes[0,0].set_title(f'Relative temporal Mean - {np.nanmean(rel_mean):.2f}')

    std_diff_1 = max(np.abs(1-np.nanmin(rel_std)), np.abs(1-np.nanmax(rel_std)))
    im1 = axes[0,1].imshow(rel_std, vmin = 1-std_diff_1, vmax = 1+std_diff_1, cmap='coolwarm')
    axes[0,1].set_title(f'Relative temporal Standarddeviation')

    im2 = axes[1,0].imshow(spearman, vmin=np.nanmin(spearman), vmax=1, cmap='viridis_r')
    axes[1,0].set_title(f'Spearman Correlation - {np.nanmean(spearman):.2f}')
    ab = AnnotationBbox(TextArea(f"{np.nanmean(rel_std):.2f}"), (2000, 2000), xybox=(1000, 1000), xycoords='data',
                    boxcoords="offset points")


    # Plot for the seasonality
    months = np.arange(1,13,1)
    bar_width = 0.4
    # im3 = axes[1,1].errorbar(months, np.nanmean(input_clim, axis=(1,2)), label=input_name, color='#79A3E6', fmt='o', alpha=0.8, markersize=3)
    # im3 = axes[1,1].errorbar(months, np.nanmean(ref_clim, axis=(1,2)), label=ref_name, color='#008176', fmt='s', alpha=0.8, markersize=3)
    axes[1,1].bar(months - bar_width/2, np.nanmean(ref_clim, axis=(1,2)), width=bar_width, color='#008176', label=ref_name, alpha=0.8)
    axes[1,1].bar(months + bar_width/2, np.nanmean(input_clim, axis=(1,2)), width=bar_width, color='#79A3E6', label=input_name, alpha=0.8)

    ax_twy = axes[1,1].twinx()
    rel_clim = np.nanmean(input_clim, axis=(1,2)) / np.nanmean(ref_clim, axis=(1,2))
    rel_clim_diff_1 = max(np.abs(1-np.nanmin(rel_clim)), np.abs(1-np.nanmax(rel_clim)))
    im4 = ax_twy.errorbar(months, rel_clim, label=f'{input_name}/{ref_name}', color='#0000A7', fmt='--')
    ax_twy.axhline(y=1, color='#0000A7', linewidth=0.5)
    axes[1,1].set_xlabel('month of year')
    handles, labels = [], []
    for ax in [axes[1,1], ax_twy]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.append(handle)
            labels.append(label)

    axes[1,1].legend(handles, labels, loc='upper right')
    axes[1,1].set_title('Seasonality of evapotranspiration')
    axes[1,1].set_ylabel("ET [mm/day]")
    axes[1,1].tick_params(axis='y', labelcolor='black')
    axes[1,1].set_xlim(1-(1.1*bar_width),12+(1.1*bar_width))
    axes[1,1].set_xticks(months)
    axes[1,1].set_xticklabels(months)
    ax_twy.set_ylim(max(0,1-rel_clim_diff_1*1.05), 1+rel_clim_diff_1*1.05)  # Example range for the ratio
    ax_twy.set_ylabel("Ratio (Input / Reference)", color="#0000A7")
    ax_twy.tick_params(axis='y', labelcolor='#0000A7')


    divider0 = make_axes_locatable(axes[0,0])
    cax = divider0.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im0, cax=cax, label='')

    divider1 = make_axes_locatable(axes[0,1])
    cax2 = divider1.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im1, cax=cax2, label='')

    divider2 = make_axes_locatable(axes[1,0])
    cax = divider2.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im2, cax=cax, label='')

    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
    axes[1,1].spines['top'].set_linewidth(0.25)
    for ax in [axes[0,0],axes[1,0],axes[0,1]]:
        ax.set_yticks([])
        ax.set_xticks([])
        ax.yaxis.labelpad = 0
        for spine in ax.spines.values():
            spine.set_linewidth(0.25)
    plt.tight_layout()

    plt.savefig(output_path / 'et_map.png', dpi=1000)
    logger.info('created et_map')

def create_map_from_output(output_path, input_name, ref_name):
    with xr.open_dataset(get_ref_file(output_path, ref_name)) as ds:
        rel_std = ds['rel_std']
        rel_mean = ds['rel_mean']
        spearman = ds['spearman']
        input_clim = ds[f'{input_name}_clim'] if f'{input_name}_clim' in ds else ds['input_clim']
        ref_clim = ds[f'{ref_name}_clim'] if f'{ref_name}_clim' in ds else ds['ref_clim']
    plot_map(rel_std=rel_std, rel_mean=rel_mean, spearman=spearman, ref_clim=ref_clim, input_clim=input_clim, input_name=input_name, ref_name=ref_name, output_path=output_path)

def compare_input_with_ref(input_path, input_var, output_path , ref_path, ref_var, input_name=None, ref_name=None, input_factor=1, ref_factor=1, coordinate_slice=None):
    if input_var is not None:
        if input_path.is_file():
            output_ds = get_file_stats(input_path, input_var, input_factor, coordinate_slice)
        elif input_path.is_dir():
            output_ds = get_stats_one_pass(input_path, input_var, input_factor, coordinate_slice)
        else:
            raise ValueError()
    else: 
        with xr.open_dataset(input_path, engine="netcdf4") as ds_input:
            if coordinate_slice is not None:
                ds_input = ds_input.sel({get_coord_key(ds_input, lat=True): coordinate_slice['lat'], get_coord_key(ds_input, lon=True): coordinate_slice['lon']})
            if 'clim' in ds_input and 'std' in ds_input and 'mean' in ds_input:
                input = ds_input
            else: 
                raise KeyError('Wrong input file-')
    #     elif input_var is not None and input_var in ds_ref:
    if ref_var is not None:
        if input_path.is_file():
            output_ds = get_file_stats(ref_path, ref_var, ref_factor, coordinate_slice)
        elif input_path.is_dir():
            output_ds = get_stats_one_pass(ref_path, ref_var, ref_factor, coordinate_slice)
        else:
            raise ValueError()
    else: 
        with xr.open_dataset(ref_path, engine="netcdf4") as ds_ref:
            if coordinate_slice is not None:
                ds_ref = ds_ref.sel({get_coord_key(ds_ref, lat=True): coordinate_slice['lat'], get_coord_key(ds_ref, lon=True): coordinate_slice['lon']})
            if 'clim' in ds_ref and 'std' in ds_ref and 'mean' in ds_ref:
                ref = ds_ref
            else: 
                raise KeyError('Wrong ref file-')
    rel_mean = input['mean'].values / ref['mean'].values
    rel_std = input['std'].values / ref['std'].values
    spearman = spearman_spatial(input['clim'], ref['clim'])
    rel_mean = xr.DataArray(rel_mean, coords={"lat": get_coord_values(input, lat=True), "lon": get_coord_values(input, lon=True)}, dims=["lat", "lon"])
    rel_std = xr.DataArray(rel_std, coords={"lat": get_coord_values(input, lat=True), "lon": get_coord_values(input, lon=True)}, dims=["lat", "lon"])
    spearman = xr.DataArray(spearman, coords={"lat": get_coord_values(input, lat=True), "lon": get_coord_values(input, lon=True)}, dims=["lat", "lon"])
    output = xr.Dataset(
            {
                f'spearman': spearman,
                f'rel_std': rel_std,
                f'rel_mean': rel_mean
            },
            coords={
                'month': np.arange(1,13,1),
                'lat': get_coord_values(input, lat=True),
                'lon': get_coord_values(input, lon=True)
            }
        )
    file_name = 'relative_stats'
    if input_name is not None:
        file_name += f'_{input_name}'
        output[f'{input_name}_clim'] = input['clim']
    else:
        output[f'input_clim'] = input['clim']
    if ref_name is not None:
        file_name += f'_{ref_name}'
        output[f'{ref_name}_clim'] = ref['clim']
    else:
        output[f'ref_clim'] = ref['clim']
    output.to_netcdf(output_path / f'{file_name}.nc')
    plot_map(rel_std=rel_std, rel_mean=rel_mean, spearman=spearman, ref_clim=ref['clim'], input_clim=input['clim'], input_name=input_name, ref_name=ref_name, output_path=output_path)

def get_ref_file(output_path, ref_name):
    return output_path / f'{ref_name}_stats.nc' if ref_name is not None else output_path / 'stats.nc'


def seasonality_grid_validation(input_path, input_var, output_path, ref_file, ref_var, input_name=None, ref_name=None, input_factor=1, ref_factor=1, only_plot=False, coordinate_slice=None, n_cpus=1):
    # client = Client(n_workers=n_cpus, timeout=f"{60*3}s", memory_limit='25GB')
    output_path = Path(output_path)
    input_path = Path(input_path)
    if not output_path.is_dir():
        output_path.mkdir(parents=True)
    if only_plot and get_ref_file(output_path, ref_name).is_file():
        create_map_from_output(output_path=output_path, input_name=input_name, ref_name=ref_name)
    elif ref_file is None:
        output_name = f'{input_name}_stats.nc' if input_name is not None else 'stats.nc'
        if input_path.is_file():
            output_ds = get_file_stats(input_path, input_var, input_factor, coordinate_slice)
        elif input_path.is_dir():
            output_ds = get_stats_one_pass(input_path, input_var, input_factor, coordinate_slice)
        else:
            raise ValueError()
        logger.info(output_path / output_name)
        output_ds.to_netcdf(output_path / output_name)
    else:
        ref_path = Path(ref_path)
        compare_input_with_ref(input_path, input_var, output_path, ref_file, ref_var, input_name, ref_name, input_factor, ref_factor, coordinate_slice)
    # client.close()