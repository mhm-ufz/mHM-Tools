
import argparse
from multiprocessing import Value
from pathlib import Path
import xarray as xr
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, AnnotationBbox
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
    data_clim = data.groupby("time.month").mean()
    if len(data_clim.month) < 12:
        clim = np.full((12,*np.shape(data.values[0])), np.nan)
        for i, month in enumerate(data_clim.month):
            clim[month-1] = data_clim.values[i]
        return clim
        # pass # data.month gives the values of the months (from 1 to 12)
    return data_clim.values

def get_clim_from_ds(ds, input_var, factor=1):
        data = ds[input_var]*factor
        clim = climatology(data)
        return xr.DataArray(clim, coords={"month": range(1, 13), "lat": get_coord_values(ds, lat=True), "lon": get_coord_values(ds, lon=True)}, dims=["month", "lat", "lon"])

def get_std_from_ds(ds, input_var, clim=None, factor=1):
    data = ds[input_var]
    data_reduced = data.values*factor - clim.sel(month=data["time.month"])
    std = np.nanstd(data_reduced.values, axis=0)
    return xr.DataArray(std, coords={"lat": get_coord_values(ds, lat=True), "lon": get_coord_values(ds, lon=True)}, dims=["lat", "lon"])
# def get_coord_values(ds, *keys):
#     for key in keys:
#         if key in ds and len(ds[key].shape) == 1:
#             return ds[key].values
#     return None

# def get_lat_values(ds):
#     return get_coord_values(ds, 'lat', 'latitude', 'northing')
# def get_lon_values(ds):
#     return get_coord_values(ds, 'lon', 'longitude', 'easting')

def get_coord_key(ds, lat=False, lon=False):
    if (lon and lat) or not (lon or lat): 
        raise ValueError(f"only lon or lat should be true but lon={lon} and lat={lat}")
    elif lat:
        keys = ['lat', 'latitude', 'northing']
    else:
        keys = ['lon', 'longitude', 'easting']
    for key in keys:
        if key in ds and len(ds[key].shape) == 1:
            return key
    raise ValueError(f"None of {keys} in dataset.")

def get_coord_values(ds, lat=False, lon=False):
    key = get_coord_key(ds, lat=lat, lon=lon)
    return ds[key].values

def get_file_stats(file, input_var, factor=1, coordiante_slice=None):
    with xr.open_dataset(file, engine="netcdf4") as ds:
        if coordiante_slice is not None: 
            ds = ds.sel({get_coord_key(ds, lat=True): coordiante_slice['lat'], get_coord_key(ds, lon=True): coordiante_slice['lon']})
        clim = get_clim_from_ds(ds, input_var, factor)
        std = get_std_from_ds(ds, input_var, clim, factor)
        mean = xr.DataArray(np.nanmean(ds[input_var], axis=0) * factor, coords={"lat": get_coord_values(ds, lat=True), "lon": get_coord_values(ds, lon=True)}, dims=["lat", "lon"])
        output = xr.Dataset(
            {
                f'clim': clim,
                f'std': std,
                f'mean': mean
            },
            coords={
                'month': np.arange(1,13,1),
                'lat': get_coord_values(ds, lat=True),
                'lon': get_coord_values(ds, lon=True)
            }
        )
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
    print('created et_map')

def create_map_from_output(output_path, input_name, ref_name):
    with xr.open_dataset(get_ref_file(output_path, ref_name)) as ds:
        rel_std = ds['rel_std']
        rel_mean = ds['rel_mean']
        spearman = ds['spearman']
        input_clim = ds[f'{input_name}_clim'] if f'{input_name}_clim' in ds else ds['input_clim']
        ref_clim = ds[f'{ref_name}_clim'] if f'{ref_name}_clim' in ds else ds['ref_clim']
    plot_map(rel_std=rel_std, rel_mean=rel_mean, spearman=spearman, ref_clim=ref_clim, input_clim=input_clim, input_name=input_name, ref_name=ref_name, output_path=output_path)

def compare_input_with_ref(input_file, input_var, output_path , ref_file, ref_var, input_name=None, ref_name=None, input_factor=1, ref_factor=1, coordiante_slice=None):
    if input_var is not None:
        input = get_file_stats(input_file, input_var, input_factor)
    else: 
        with xr.open_dataset(input_file, engine="netcdf4") as ds_input:
            if coordiante_slice is not None:
                ds_input = ds_input.sel({get_coord_key(ds_input, lat=True): coordiante_slice['lat'], get_coord_key(ds_input, lon=True): coordiante_slice['lon']})
            if 'clim' in ds_input and 'std' in ds_input and 'mean' in ds_input:
                input = ds_input
            else: 
                raise KeyError('Wrong input file-')
    #     elif input_var is not None and input_var in ds_ref:
    if ref_var is not None:
        ref = get_file_stats(ref_file, ref_var, ref_factor)
    else: 
        with xr.open_dataset(ref_file, engine="netcdf4") as ds_ref:
            if coordiante_slice is not None:
                ds_ref = ds_ref.sel({get_coord_key(ds_ref, lat=True): coordiante_slice['lat'], get_coord_key(ds_ref, lon=True): coordiante_slice['lon']})
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


def seasonality_grid_validation(input_file, input_var, output_path, ref_file, ref_var, input_name=None, ref_name=None, input_factor=1, ref_factor=1, only_plot=False, coordinate_slice=None):
    output_path = Path(output_path)
    if not output_path.is_dir():
        output_path.mkdir(parents=True)
    if only_plot and get_ref_file(output_path, ref_name).is_file():
        create_map_from_output(output_path=output_path, input_name=input_name, ref_name=ref_name)
    elif ref_file is None:
        output_name = f'{input_name}_stats.nc' if input_name is not None else 'stats.nc'
        get_file_stats(input_file, input_var, input_factor, coordinate_slice).to_netcdf(output_path / output_name)
    else:
        compare_input_with_ref(input_file, input_var, output_path, ref_file, ref_var, input_name, ref_name, input_factor, ref_factor, coordinate_slice)
