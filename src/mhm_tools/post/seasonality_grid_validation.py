import array
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from matplotlib.offsetbox import AnnotationBbox, TextArea
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mhm_tools.common.logger import log_arguments, logger
from scipy.stats import spearmanr

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
    """Calculate maps of Spearman rank correlation between two xarray DataArrays of shape(12,n,m)."""
    if len(np.shape(data1)) != len(np.shape(data2)) or len(np.shape(data1)) != 3:
        raise ValueError("Wrong shape for spatial spearman correlation!")
    res = np.full(np.shape(data1[0]), np.nan)
    for i, row in enumerate(data1[0]):
        for j, col in enumerate(row):
            sp_corr, sp_pval = spearman_correlation(data1[:, i, j], data2[:, i, j])
            res[i, j] = sp_corr
    return res


def climatology(data):
    """Calculate the climatology from a xarray DataArray."""
    data_clim = data.groupby("time.month").mean(dim="time", skipna=True)

    # Ensure the climatology has all 12 months, filling missing months with NaNs
    data_clim = data_clim.reindex(month=np.arange(1, 13), fill_value=np.nan)
    return data_clim


def get_clim_from_ds(ds, input_var=None, factor=1):
    """Calculate climatology from DataSet with variable or DataArray while mulitplying with a provided factor."""
    if input_var is None:
        data = ds * factor
    else:
        data = ds[input_var] * factor
    return climatology(data)


def get_std_from_ds(ds, input_var=None, clim=None, factor=1):
    """Calculate maps of temporal standard deviation from an DataArray.

    If a climatology is provided the timeseries can be detrended by seasonality.
    """
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

    # Return as DataArray with appropriate coordinates
    if type(std) is array and len(std.shape) == 2:
        return xr.DataArray(
            std,
            coords={
                "lat": get_coord_values(ds, lat=True),
                "lon": get_coord_values(ds, lon=True),
            },
            dims=["lat", "lon"],
        )
    return std


def get_coord_key(ds, lat=False, lon=False):
    if (lon and lat) or not (lon or lat): 
        raise ValueError(f"only lon or lat should be true but lon={lon} and lat={lat}")
    if lat:
        keys = ["lat", "latitude", "northing", "y", "new_y"]
    else:
        keys = ["lon", "longitude", "easting", "x", "new_x"]
    ds_dims = ds.dims if isinstance(ds, xr.DataArray) else ds.coords
    for key in keys:
        if key in ds_dims and len(ds[key].shape) == 1:
            return key
    for key in keys:
        if key in ds_dims:
            logger.warning(
                f"{type(ds)} contains key: {key} but ds[key] has shape {ds[key].shape}."
            )
            return key
    raise ValueError(f"None of {keys} in {type(ds).__name__} keys {ds_dims}.")


def get_coord_values(ds, lat=False, lon=False):
    "Get latitude or longitude values from DataSet."
    key = get_coord_key(ds, lat=lat, lon=lon)
    return ds[key].values


def get_file_stats(file, input_var, factor=1, coordinate_slice=None, output_path=None):
    with xr.open_dataset(file, engine="netcdf4") as ds: 
        # Apply coordinate slicing if needed
        if coordinate_slice is not None:
            lat_key = get_coord_key(ds, lat=True)
            lon_key = get_coord_key(ds, lon=True)
            ds = ds.sel(
                {lat_key: coordinate_slice["lat"], lon_key: coordinate_slice["lon"]}
            )

        # Calculate climatology and standard deviation along the time dimension
        clim = get_clim_from_ds(ds, input_var, factor) 
        std = get_std_from_ds(ds, input_var, clim, factor)  
        
        mean = ds[input_var].mean(dim='time', skipna=True) * factor

        # Construct the output dataset with lazy evaluations
        output = xr.Dataset(
            {"clim": clim, "std": std, "mean": mean},
            coords={
                "month": np.arange(1, 13, 1),
                "lat": get_coord_values(ds, lat=True),
                "lon": get_coord_values(ds, lon=True),
            },
        )
    if output_path is not None:
        output.to_netcdf(output_path)
    return output


def get_files(path, n_bootstrap_years=None):
    nc_files = []
    # Search for .nc files at each depth level
    if n_bootstrap_years is not None:
        # needs fixed folder structure of y/m/file
        years = [y for y in path.glob('*/') if y.is_dir()]
        selected_years = random.choices(years, k=n_bootstrap_years)
        for year in selected_years:
            for depth in range(0, 3):  # Depth 0 to 2
                nc_files.extend(year.glob("*/" * depth + "*.nc"))
    else:
        for depth in range(1, 4):  # Depth 1 to 3
            nc_files.extend(path.glob("*/" * depth + "*.nc"))
    return nc_files


def combine_results(results):
    total_count = sum(count for _, _, count, _, _ in results)
    total_mean = sum(mean * count for mean, _, count, _, _ in results) / total_count
    total_M2 = sum(M2 for _, M2, _, _, _ in results)
    total_M2 += sum(
        count * (mean - total_mean) ** 2 for mean, _, count, _, _ in results
    )
    monthly_sums = sum(ms for _, _, _, ms, _ in results)
    monthly_counts = sum(mc for _, _, _, _, mc in results)
    return total_mean, total_M2, total_count, monthly_sums, monthly_counts

def get_stats_one_pass_subset(files, input_var, factor=1, coordinate_slice=None):
    """Take a list of files with all containing data for one month and creating statisitcs while reading them one by one."""
    da = None
    with xr.open_dataset(files[0], engine="netcdf4") as ds:
        # Apply coordinate slicing if needed
        if coordinate_slice is not None:
            lat_key = get_coord_key(ds, lat=True)
            lon_key = get_coord_key(ds, lon=True)
            ds = ds.sel({lat_key: coordinate_slice['lat'], lon_key: coordinate_slice['lon']})
        da = ds[input_var]
    count = 0 #xr.DataArray(np.ones(mean.shape, dtype=int).copy(), coords=mean.coords, dims=mean.dims).expand_dims(dim='time', axis=0)
    mean = np.zeros(da.shape[1:])
    sum_square_diff = np.zeros(da.shape[1:])
    monthly_sums = np.zeros((12, *da.shape[1:]))
    monthly_counts = np.zeros((12, *da.shape[1:]))
    for f, file in enumerate(files): 
        with xr.open_dataset(file, engine="netcdf4") as ds:
            logger.info(f"timestep {count} in file {f+1} / {len(files)} from {file}")
            if coordinate_slice is not None:
                lat_key = get_coord_key(ds, lat=True)
                lon_key = get_coord_key(ds, lon=True)
                ds = ds.sel({lat_key: coordinate_slice['lat'], lon_key: coordinate_slice['lon']})
            da = ds[input_var]
            for time_value, data_slice in da.groupby("time"):
                try:
                    data_values = data_slice.values[0]*factor
                    logger.debug(f"{count} - {np.shape(data_values)}")
                    count += 1
                    delta = data_values - mean
                    mean += delta / count
                    delta2 = data_values - mean
                    sum_square_diff += delta * delta2
                    # climatology
                    month = int(data_slice["time.month"].values[0] - 1)
                    monthly_sums[month] += data_slice.fillna(0).squeeze(dim="time").values * factor
                    monthly_counts[month] += ~np.isnan(data_slice.squeeze(dim="time").values)
                except Exception as e:
                    raise e
    logger.debug(f"{np.nanmean(mean)}, {np.nanmean(sum_square_diff)}, {count}, {np.nanmean(monthly_sums)}, {np.nanmean(monthly_counts)}")
    return mean, sum_square_diff, count, monthly_sums, monthly_counts


def split_file_list(file_list, n_processes):
    return [file_list[i::n_processes] for i in range(n_processes)]

def get_stats_one_pass(input_path, input_var, factor=1, coordinate_slice=None, ncpus=1, n_bootstrap_years=None, bootstrap_index=None, output_path=None):
    files = get_files(input_path, n_bootstrap_years=n_bootstrap_years)
    file_subsets = split_file_list(files, ncpus)
    logger.info('creating statistics...')
    subset_results = Parallel(n_jobs=ncpus, backend="loky")(
                delayed(get_stats_one_pass_subset)(file_subset, input_var, factor, coordinate_slice)
                for file_subset in file_subsets)
    logger.info('combining results...')
    mean, sum_square_diff, count, monthly_sums, monthly_counts = combine_results(subset_results)
    logger.debug(f"{mean.mean()}, {sum_square_diff.mean()}, {count}, {monthly_sums.mean()}, {monthly_counts.mean()}")
    variance = sum_square_diff / (count - 1)
    std_dev = np.sqrt(variance)
    monthly_sums = np.where(monthly_counts>0, monthly_sums, np.nan)
    monthly_counts = np.where(monthly_counts>0, monthly_counts, np.nan)
    climatology = monthly_sums / monthly_counts
    climatology =  np.where(monthly_counts > 0, climatology, np.nan)
    with xr.open_dataset(files[0], engine="netcdf4") as ds:
        # Apply coordinate slicing if needed
        if coordinate_slice is not None:
            lat_key = get_coord_key(ds, lat=True)
            lon_key = get_coord_key(ds, lon=True)
            ds = ds.sel(
                {lat_key: coordinate_slice["lat"], lon_key: coordinate_slice["lon"]}
            )
            lat = get_coord_values(ds, lat=True)
            lon = get_coord_values(ds, lon=True)
    # Calculate climatology and standard deviation along the time dimension
    # Construct the output dataset with lazy evaluations
    # climatology = climatology.rename({get_coord_key(climatology, lat=True): "lat", get_coord_key(climatology, lon=True): "lon"})
    
    std = xr.DataArray(
            std_dev, 
            coords={"lat": lat, "lon": lon},
            dims=["lat", "lon"]
        )
    clim = xr.DataArray(
        climatology,
        coords={"month": np.arange(1, 13, 1), "lat": lat, "lon": lon},
        dims=["month", "lat", "lon"],
    )
    mean = xr.DataArray(mean, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"])
    output = xr.Dataset(
        {"clim": clim, "std": std, "mean": mean},
        coords={"month": np.arange(1, 13, 1), "lat": lat, "lon": lon},
    )
    # Trigger computation if needed
    if output_path is not None:
        output_file = output_path.parent / f"{output_path.stem}_{bootstrap_index}.nc" if bootstrap_index is not None else output_path
        logger.info(f'Writing output to {output_file}')
        output.to_netcdf(output_file)
    return output


def plot_map(
    rel_mean, rel_std, spearman, ref_clim, input_clim, input_name, ref_name, output_path
):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 4.68))
    if input_name is not None and ref_name is not None:
        fig.suptitle(f"Comparision {input_name} with {ref_name}")

    # Set common colormap and normalization limits for mean_et and mean_aet
    mean_diff_1 = max(np.abs(1 - np.nanmin(rel_mean)), np.abs(1 - np.nanmax(rel_mean)))
    im0 = axes[0, 0].imshow(
        rel_mean, vmin=1 - mean_diff_1, vmax=1 + mean_diff_1, cmap="coolwarm"
    )
    axes[0, 0].set_title(f"Relative temporal Mean - {np.nanmean(rel_mean):.2f}")

    std_diff_1 = max(np.abs(1 - np.nanmin(rel_std)), np.abs(1 - np.nanmax(rel_std)))
    im1 = axes[0, 1].imshow(
        rel_std, vmin=1 - std_diff_1, vmax=1 + std_diff_1, cmap="coolwarm"
    )
    axes[0, 1].set_title("Relative temporal Standarddeviation")

    im2 = axes[1, 0].imshow(
        spearman, vmin=np.nanmin(spearman), vmax=1, cmap="viridis_r"
    )
    axes[1, 0].set_title(f"Spearman Correlation - {np.nanmean(spearman):.2f}")
    ab = AnnotationBbox(
        TextArea(f"{np.nanmean(rel_std):.2f}"),
        (2000, 2000),
        xybox=(1000, 1000),
        xycoords="data",
        boxcoords="offset points",
    )

    # Plot for the seasonality
    months = np.arange(1, 13, 1)
    bar_width = 0.4
    # im3 = axes[1,1].errorbar(months, np.nanmean(input_clim, axis=(1,2)), label=input_name, color='#79A3E6', fmt='o', alpha=0.8, markersize=3)
    # im3 = axes[1,1].errorbar(months, np.nanmean(ref_clim, axis=(1,2)), label=ref_name, color='#008176', fmt='s', alpha=0.8, markersize=3)
    axes[1, 1].bar(
        months - bar_width / 2,
        np.nanmean(ref_clim, axis=(1, 2)),
        width=bar_width,
        color="#008176",
        label=ref_name,
        alpha=0.8,
    )
    axes[1, 1].bar(
        months + bar_width / 2,
        np.nanmean(input_clim, axis=(1, 2)),
        width=bar_width,
        color="#79A3E6",
        label=input_name,
        alpha=0.8,
    )

    ax_twy = axes[1, 1].twinx()
    rel_clim = np.nanmean(input_clim, axis=(1, 2)) / np.nanmean(ref_clim, axis=(1, 2))
    rel_clim_diff_1 = max(
        np.abs(1 - np.nanmin(rel_clim)), np.abs(1 - np.nanmax(rel_clim))
    )
    im4 = ax_twy.errorbar(
        months, rel_clim, label=f"{input_name}/{ref_name}", color="#0000A7", fmt="--"
    )
    ax_twy.axhline(y=1, color="#0000A7", linewidth=0.5)
    axes[1, 1].set_xlabel("month of year")
    handles, labels = [], []
    for ax in [axes[1, 1], ax_twy]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.append(handle)
            labels.append(label)

    axes[1, 1].legend(handles, labels, loc="upper right")
    axes[1, 1].set_title("Seasonality of evapotranspiration")
    axes[1, 1].set_ylabel("ET [mm/day]")
    axes[1, 1].tick_params(axis="y", labelcolor="black")
    axes[1, 1].set_xlim(1 - (1.1 * bar_width), 12 + (1.1 * bar_width))
    axes[1, 1].set_xticks(months)
    axes[1, 1].set_xticklabels(months)
    ax_twy.set_ylim(
        max(0, 1 - rel_clim_diff_1 * 1.05), 1 + rel_clim_diff_1 * 1.05
    )  # Example range for the ratio
    ax_twy.set_ylabel("Ratio (Input / Reference)", color="#0000A7")
    ax_twy.tick_params(axis="y", labelcolor="#0000A7")

    divider0 = make_axes_locatable(axes[0, 0])
    cax = divider0.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im0, cax=cax, label="")

    divider1 = make_axes_locatable(axes[0, 1])
    cax2 = divider1.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im1, cax=cax2, label="")

    divider2 = make_axes_locatable(axes[1, 0])
    cax = divider2.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im2, cax=cax, label="")

    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
    axes[1, 1].spines["top"].set_linewidth(0.25)
    for ax in [axes[0, 0], axes[1, 0], axes[0, 1]]:
        ax.set_yticks([])
        ax.set_xticks([])
        ax.yaxis.labelpad = 0
        for spine in ax.spines.values():
            spine.set_linewidth(0.25)
    plt.tight_layout()

    plt.savefig(output_path / "et_map.png", dpi=1000)
    logger.info("created et_map")


def create_map_from_output(output_path, input_name, ref_name):
    with xr.open_dataset(get_rel_stat_file(output_path, ref_name)) as ds:
        rel_std = ds['rel_std']
        rel_mean = ds['rel_mean']
        spearman = ds['spearman']
        input_clim = ds[f'{input_name}_clim'] if f'{input_name}_clim' in ds else ds['input_clim']
        ref_clim = ds[f'{ref_name}_clim'] if f'{ref_name}_clim' in ds else ds['ref_clim']
    plot_map(rel_std=rel_std, rel_mean=rel_mean, spearman=spearman, ref_clim=ref_clim, input_clim=input_clim, input_name=input_name, ref_name=ref_name, output_path=output_path)

def get_stats(path, var, factor, coordinate_slice, n_bootstrap_years, ncpus, output_file):
    if var is not None:
        if path.is_file():
            stats_ds = get_file_stats(path, var, factor, coordinate_slice, output_path=output_file)
        elif path.is_dir():
            stats_ds = get_stats_one_pass(path, var, factor, coordinate_slice, n_bootstrap_years=n_bootstrap_years, ncpus=ncpus, output_path=output_file)
        else:
            raise ValueError()
    else: 
        with xr.open_dataset(path, engine="netcdf4") as ds_input:
            if coordinate_slice is not None:
                ds_input = ds_input.sel({get_coord_key(ds_input, lat=True): coordinate_slice['lat'], get_coord_key(ds_input, lon=True): coordinate_slice['lon']})
            if 'clim' in ds_input and 'std' in ds_input and 'mean' in ds_input:
                stats_ds = ds_input
            else: 
                raise KeyError('Wrong statisitcs file. If you want to create new statistics you have to provide a var.')
    return stats_ds

def compare_input_with_ref(input_path, input_var, output_path , ref_path, ref_var, input_name=None, ref_name=None, input_factor=1, ref_factor=1, coordinate_slice=None, n_bootstrap_years=None, bootstrap_index=None, ncpus=1):
    if bootstrap_index is not None: 
        random.seed(bootstrap_index)
    # get input statistics
    input_stats_file = None # output_path / f"{input_name}_stats.nc" if input_name is not None else "input_stats.nc"
    ref_stats_file = None #output_path / f"{ref_name}_stats.nc" if ref_name is not None else "ref_stats.nc"
    # TODO: Add index to file names
    input = get_stats(path=input_path, var=input_var, factor=input_factor, coordinate_slice=coordinate_slice, n_bootstrap_years=n_bootstrap_years, ncpus=ncpus, output_file=input_stats_file)
    # get output statistics

    ref = get_stats(path=ref_path, var=ref_var, factor=ref_factor, coordinate_slice=coordinate_slice, n_bootstrap_years=n_bootstrap_years, ncpus=ncpus, output_file=ref_stats_file)
    # compare and save statistics
    rel_mean = input['mean'].values / ref['mean'].values
    rel_std = input['std'].values / ref['std'].values
    spearman = spearman_spatial(input['clim'], ref['clim'])
    rel_mean = xr.DataArray(rel_mean, coords={"lat": get_coord_values(input, lat=True), "lon": get_coord_values(input, lon=True)}, dims=["lat", "lon"])
    rel_std = xr.DataArray(rel_std, coords={"lat": get_coord_values(input, lat=True), "lon": get_coord_values(input, lon=True)}, dims=["lat", "lon"])
    spearman = xr.DataArray(spearman, coords={"lat": get_coord_values(input, lat=True), "lon": get_coord_values(input, lon=True)}, dims=["lat", "lon"])
    input_clim = xr.DataArray(input['clim'].values, coords={"month":np.arange(1,13,1), "lat": get_coord_values(input, lat=True), "lon": get_coord_values(input, lon=True)}, dims=["month", "lat", "lon"])
    ref_clim = xr.DataArray(ref['clim'].values, coords={"month":np.arange(1,13,1), "lat": get_coord_values(input, lat=True), "lon": get_coord_values(input, lon=True)}, dims=["month", "lat", "lon"])
    output = xr.Dataset(
        {"spearman": spearman, "rel_std": rel_std, "rel_mean": rel_mean},
        coords={
            "month": np.arange(1, 13, 1),
            "lat": get_coord_values(input, lat=True),
            "lon": get_coord_values(input, lon=True),
        },
    )
    file_name = "relative_stats"
    if input_name is not None:
        file_name += f'_{input_name}'
        output[f'{input_name}_clim'] = input_clim
    else:
        output[f'input_clim'] = input_clim
    if ref_name is not None:
        file_name += f'_{ref_name}'
        output[f'{ref_name}_clim'] = ref_clim
    else:
        output[f'ref_clim'] = ref_clim
    if bootstrap_index is not None:
        file_name = output_path / f'{file_name}_{bootstrap_index}.nc'
        output.to_netcdf(file_name)
        return file_name
    else: 
        output.to_netcdf(output_path / f"{file_name}.nc")
        plot_map(rel_std=rel_std, rel_mean=rel_mean, spearman=spearman, ref_clim=ref['clim'], input_clim=input['clim'], input_name=input_name, ref_name=ref_name, output_path=output_path)

def get_rel_stat_file(output_path, input_name, ref_name):
    file_name = 'relative_stats'
    if input_name is not None:
        file_name += f'_{input_name}'
    if ref_name is not None:
        file_name += f'_{ref_name}'
    return output_path / f"{file_name}.nc"
    
def evaluate_boostraping_stat_files(stat_files, input_name, ref_name):
    """Evaluate bootstrapped statistics and compute median across bootstrap iterations."""
    # Open the first file to initialize dimensions and weights
    with xr.open_dataset(stat_files[0]) as first_file:
        shape = first_file["rel_mean"].shape
        n_bootstrap = len(stat_files)

        # Determine keys for climatology fields
        input_clim_key = f"{input_name}_clim" if f"{input_name}_clim" in first_file else "input_clim"
        ref_clim_key = f"{ref_name}_clim" if f"{ref_name}_clim" in first_file else "ref_clim"

        # Preallocate arrays for all variables
        mean = np.empty((n_bootstrap, *shape))
        std = np.empty((n_bootstrap, *shape))
        spearman = np.empty((n_bootstrap, *shape))
        input_clim = np.empty((n_bootstrap, 12, *shape))  # Month x lat x lon
        ref_clim = np.empty((n_bootstrap, 12, *shape))    # Month x lat x lon

    # Fill the preallocated arrays with bootstrap data
    for i, file in enumerate(stat_files):
        with xr.open_dataset(file) as ds:
            mean[i] = ds["rel_mean"].values
            std[i] = ds["rel_std"].values
            spearman[i] = ds["spearman"].values
            input_clim[i] = ds[input_clim_key].values
            ref_clim[i] = ds[ref_clim_key].values

    # Convert the arrays into xarray DataArrays
    mean_da = xr.DataArray(mean, dims=["bootstrap", "lat", "lon"])
    std_da = xr.DataArray(std, dims=["bootstrap", "lat", "lon"])
    spearman_da = xr.DataArray(spearman, dims=["bootstrap", "lat", "lon"])
    input_clim_da = xr.DataArray(input_clim, dims=["bootstrap", "month", "lat", "lon"])
    ref_clim_da = xr.DataArray(ref_clim, dims=["bootstrap", "month", "lat", "lon"])

    # Combine results into an xarray Dataset
    return {
            "rel_mean": mean_da.median(dim="bootstrap"),
            "rel_std": std_da.median(dim="bootstrap"),
            "spearman": spearman_da.median(dim="bootstrap"),
            "input_clim": input_clim_da.median(dim="bootstrap"),
            "ref_clim": ref_clim_da.median(dim="bootstrap"),
        }

def get_dataset_from_path(path):
    if path.is_file() and path.suffix == 'nc':
        return xr.open_dataset(path)
    elif path.is_dir():
        file_list = get_files(path)
        logger.debug(file_list)
        logger.debug('combining files by coords ...')
        return xr.open_mfdataset(
            file_list,
            combine="by_coords"  # Ensures files are combined based on shared coordinates
        )


def direct_comparison(input_path, ref_path, input_var, ref_var, input_name, ref_name, input_factor, ref_factor, coordinate_slice, output_path):
    if ref_path is None:
        raise ValueError('ref_path must be given for direct comparison.')
    logger.info("Start direct comparison.")
    input = get_dataset_from_path(input_path)
    ref = get_dataset_from_path(ref_path)
    logger.info('crop the data spatially') 
    input = input.sel({get_coord_key(input, lat=True): coordinate_slice["lat"], get_coord_key(input, lon=True): coordinate_slice["lon"]})
    ref = ref.sel({get_coord_key(ref, lat=True): coordinate_slice["lat"], get_coord_key(ref, lon=True): coordinate_slice["lon"]})
    logger.info('crop to overlapping time')
    start_time = max(input.time.min(), ref.time.min())
    end_time = min(input.time.max(), ref.time.max())
    logger.info(f'start_time {start_time}; end_time {end_time}')
    input = input.sel(time=slice(start_time, end_time))
    ref = ref.sel(time=slice(start_time, end_time))

    logger.info('regridd to the lower temporal resolution using average')
    if len(ref.time) < len(input.time):
        low_res_time = ref.time
        input = (input.groupby_bins("time", bins=low_res_time).mean(dim="time")
        )
    elif len(input.time) > len(ref.time):
        low_res_time = input.time
        ref = (ref.groupby_bins("time", bins=low_res_time).mean(dim="time")
        )
    
    logger.info('calculate climatologies')
    input_clim = get_clim_from_ds(input, input_var, input_factor)
    ref_clim = get_clim_from_ds(ref, ref_var, ref_factor)

    logger.info('calculate relative standard deviation')
    # with clim given here the seasonality is removed from the data 
    input_std = get_std_from_ds(input, input_var, input_clim, input_factor)
    ref_std = get_std_from_ds(ref, ref_var, ref_clim, ref_factor)
    rel_std = (input_std / ref_std).values

    input = input[input_var] * input_factor
    ref = ref[ref_var] * ref_factor

    logger.info("calculate rel mean")
    rel_mean = input.mean(dim='time', skipna=True).values / ref.mean(dim='time', skipna=True).values
    logger.info("calculate spearman spatial ")
    spearman = spearman_spatial(input, ref).values

    rel_mean = xr.DataArray(rel_mean, coords={"lat": get_coord_values(input, lat=True), "lon": get_coord_values(input, lon=True)}, dims=["lat", "lon"])
    rel_std = xr.DataArray(rel_std, coords={"lat": get_coord_values(input, lat=True), "lon": get_coord_values(input, lon=True)}, dims=["lat", "lon"])
    spearman = xr.DataArray(spearman, coords={"lat": get_coord_values(input, lat=True), "lon": get_coord_values(input, lon=True)}, dims=["lat", "lon"])
    output = xr.Dataset(
        {"spearman": spearman, "rel_std": rel_std, "rel_mean": rel_mean},
        coords={
            "lat": get_coord_values(input, lat=True),
            "lon": get_coord_values(input, lon=True),
        },
    )
    logger.info(f"Save the results to {output_path/ f"{input_name}_{ref_name}_direct_comp.nc"}")
    output.to_netcdf(output_path/ f"{input_name}_{ref_name}_direct_comp.nc")
    plot_map(rel_mean, rel_std, spearman, ref_clim, input_clim, input_name, ref_name, output_path)


@log_arguments()
def seasonality_grid_validation(input_path, input_var, output_path, ref_path, ref_var, input_name=None, ref_name=None, input_factor=1, ref_factor=1, only_plot=False, coordinate_slice=None, n_cpus=1, n_bootstrap_years=None, n_bootstrap_selections=None, direct_comp=True):
    output_path = Path(output_path)
    input_path = Path(input_path)
    ref_path = Path(ref_path) if ref_path is not None else None
    if direct_comp:
        direct_comparison(input_path, ref_path, input_var, ref_var, input_name, ref_name, input_factor, ref_factor, coordinate_slice, output_path)

    if not output_path.is_dir():
        output_path.mkdir(parents=True)
    if only_plot and get_rel_stat_file(output_path, input_name, ref_name).is_file():
        create_map_from_output(output_path=output_path, input_name=input_name, ref_name=ref_name)
    elif ref_path is None:
        output_name = f'{input_name}_stats.nc' if input_name is not None else 'stats.nc'
        if input_path.is_file():
            get_file_stats(input_path, input_var, input_factor, coordinate_slice, output=output_path/output_name)
        elif n_bootstrap_years is not None and n_bootstrap_selections > 0 and input_path.is_dir():
            stat_files = Parallel(n_jobs=n_cpus)(delayed(get_stats_one_pass)(ref_path, ref_var, ref_factor, coordinate_slice, n_bootstrap_years, s, output_path/output_name) for s in range(n_bootstrap_selections))
        elif input_path.is_dir():
            get_stats_one_pass(input_path, input_var, input_factor, coordinate_slice, output_path=output_path/output_name)
        else:
            raise ValueError("input path does not exist")
    else:
        if n_bootstrap_years is not None and n_bootstrap_selections > 0 and input_path.is_dir() and ref_path.is_dir():
            if only_plot:
                stat_files = list(output_path.glob(f'relative_stats_{input_name}_{ref_name}_*'))
            else:
                ref_path = Path(ref_path)
                stat_files = Parallel(n_jobs=n_cpus)(delayed(compare_input_with_ref)(input_path, input_var, output_path, ref_path, ref_var, input_name, ref_name, input_factor, ref_factor, coordinate_slice, n_bootstrap_years, s) for s in range(n_bootstrap_selections))
            results = evaluate_boostraping_stat_files(stat_files, input_name=input_name, ref_name=ref_name)
            plot_map(**results, output_path=output_path, input_name=input_name, ref_name=ref_name)
        else:
            compare_input_with_ref(input_path, input_var, output_path, ref_path, ref_var, input_name, ref_name, input_factor, ref_factor, coordinate_slice, ncpus=n_cpus)