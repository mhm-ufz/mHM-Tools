"""Helpers for plotting metric medians on catchment geometries."""

import logging
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt

from mhm_tools.common.xarray_utils import get_coord_key, get_single_data_var

logger = logging.getLogger(__name__)


def create_match_id(value):
    """Create a stable string ID for file matching.

    Parameters
    ----------
    value : object
        ID-like value to normalize.

    Returns
    -------
    str
        Normalized ID string.
    """
    if value is None:
        return ""
    if isinstance(value, float) and np.isfinite(value) and value.is_integer():
        return str(int(value))
    value_str = str(value).strip()
    if value_str.endswith(".0"):
        try:
            return str(int(float(value_str)))
        except ValueError:
            return value_str
    return value_str


def find_matching_geometry_file(folder, match_id, suffix):
    """Find a geometry file whose name contains an ID.

    Parameters
    ----------
    folder : str or Path
        Directory containing geometry files.
    match_id : object
        ID used for filename matching.
    suffix : str
        File suffix to match.

    Returns
    -------
    Path or None
        Matching geometry file, if found.
    """
    if folder is None:
        return None
    geometry_dir = Path(folder)
    if not geometry_dir.is_dir():
        return None
    normalized_id = create_match_id(match_id)
    if not normalized_id:
        return None
    matches = sorted(geometry_dir.glob(f"*{normalized_id}*{suffix}"))
    if not matches:
        return None
    if len(matches) > 1:
        logger.warning(
            f"Multiple {suffix} files matched ID {normalized_id} in "
            f"{geometry_dir}. Using {matches[0].name}."
        )
    return matches[0]


def read_shape_geometry(shape_file, geometry_id=None):
    """Read one shapefile geometry in EPSG:4326.

    Parameters
    ----------
    shape_file : str or Path
        Shapefile path.
    geometry_id : object, optional
        ID assigned to the output geometry.

    Returns
    -------
    geopandas.GeoDataFrame
        One-row GeoDataFrame with a dissolved geometry.
    """
    import geopandas as gpd

    gdf = gpd.read_file(shape_file)
    if gdf.empty:
        return gpd.GeoDataFrame(columns=["id", "geometry"], geometry="geometry")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif str(gdf.crs) != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    geometry = _geometry_union(gdf.geometry)
    return gpd.GeoDataFrame(
        {"id": [create_match_id(geometry_id)], "geometry": [geometry]},
        geometry="geometry",
        crs="EPSG:4326",
    )


def read_mask_geometry(mask_file, mask_var=None, geometry_id=None):
    """Read one NetCDF mask as a dissolved EPSG:4326 geometry.

    Parameters
    ----------
    mask_file : str or Path
        NetCDF mask file.
    mask_var : str, optional
        Mask variable name. Defaults to "mask" when available.
    geometry_id : object, optional
        ID assigned to the output geometry.

    Returns
    -------
    geopandas.GeoDataFrame
        One-row GeoDataFrame with the valid mask geometry.
    """
    import geopandas as gpd
    from rasterio import features
    from rasterio.transform import from_origin
    from shapely.geometry import shape

    with xr.open_dataset(mask_file) as mask_ds:
        mask_da = _get_mask_data_array(mask_ds, mask_file, mask_var=mask_var)
        lat_key = get_coord_key(mask_da, lat=True)
        lon_key = get_coord_key(mask_da, lon=True)
        for dim in mask_da.dims:
            if dim not in [lat_key, lon_key]:
                mask_da = mask_da.isel({dim: 0}, drop=True)
        mask_da = mask_da.load()

    lat_values = np.asarray(mask_da[lat_key].values, dtype=float)
    lon_values = np.asarray(mask_da[lon_key].values, dtype=float)
    values = np.asarray(mask_da.values)
    if lat_values.size < 2 or lon_values.size < 2:
        msg = f"Mask file {mask_file} must contain at least two lat/lon cells."
        raise ValueError(msg)
    if lat_values[0] < lat_values[-1]:
        values = np.flipud(values)
        lat_values = lat_values[::-1]
    if lon_values[0] > lon_values[-1]:
        values = np.fliplr(values)
        lon_values = lon_values[::-1]

    lat_res = float(np.nanmedian(np.abs(np.diff(lat_values))))
    lon_res = float(np.nanmedian(np.abs(np.diff(lon_values))))
    transform = from_origin(
        float(np.nanmin(lon_values) - lon_res / 2.0),
        float(np.nanmax(lat_values) + lat_res / 2.0),
        lon_res,
        lat_res,
    )
    valid_mask = np.isfinite(values) & (values > 0)
    shapes = features.shapes(
        valid_mask.astype(np.uint8),
        mask=valid_mask,
        transform=transform,
        connectivity=8,
    )
    geometries = [shape(geometry) for geometry, value in shapes if value == 1]
    if not geometries:
        return gpd.GeoDataFrame(columns=["id", "geometry"], geometry="geometry")
    geometry = _geometry_union(gpd.GeoSeries(geometries, crs="EPSG:4326"))
    return gpd.GeoDataFrame(
        {"id": [create_match_id(geometry_id)], "geometry": [geometry]},
        geometry="geometry",
        crs="EPSG:4326",
    )


def _get_mask_data_array(mask_ds, mask_file, mask_var=None):
    """Get the mask data array used for geometry creation.

    Parameters
    ----------
    mask_ds : xarray.Dataset
        Open mask dataset.
    mask_file : str or Path
        Source mask file path used in error messages.
    mask_var : str, optional
        Explicit mask variable name.

    Returns
    -------
    xarray.DataArray
        Selected mask data array.
    """
    if mask_var is not None:
        if mask_var not in mask_ds.data_vars:
            msg = f"Mask variable {mask_var!r} not found in {mask_file}."
            raise ValueError(msg)
        return mask_ds[mask_var]
    if "mask" in mask_ds.data_vars:
        return mask_ds["mask"]
    mask_var_name = get_single_data_var(mask_ds)
    if mask_var_name is None:
        msg = (
            f"Could not select a mask variable from {mask_file}. "
            "Use --mask-var to choose one explicitly."
        )
        raise ValueError(msg)
    return mask_ds[mask_var_name]


def calculate_metric_medians(metric_df, variables=None, id_col="id", default_id=None):
    """Calculate median metric rows, optionally grouped by ID.

    Parameters
    ----------
    metric_df : pandas.DataFrame
        Metric rows.
    variables : Sequence[str], optional
        Metric columns to aggregate.
    id_col : str, optional
        ID column used for grouping.
    default_id : object, optional
        ID used when id_col is absent.

    Returns
    -------
    pandas.DataFrame
        Median metric rows with an ID column.
    """
    variables = get_metric_variables(metric_df, variables=variables, id_col=id_col)
    if not variables:
        return pd.DataFrame(columns=[id_col])
    data = metric_df.copy()
    for variable in variables:
        data[variable] = pd.to_numeric(data[variable], errors="coerce")
    if id_col in data.columns:
        rows = data.groupby(id_col, dropna=True)[variables].median().reset_index()
    else:
        row = {id_col: create_match_id(default_id)}
        for variable in variables:
            row[variable] = data[variable].median()
        rows = pd.DataFrame([row])
    return rows.replace([np.inf, -np.inf], np.nan)


def get_metric_variables(metric_df, variables=None, id_col="id"):
    """Get numeric metric columns from a metric table.

    Parameters
    ----------
    metric_df : pandas.DataFrame
        Metric table.
    variables : Sequence[str], optional
        Explicit metric columns.
    id_col : str, optional
        ID column to exclude from defaults.

    Returns
    -------
    list[str]
        Metric column names.
    """
    if variables is not None:
        return [variable for variable in variables if variable in metric_df.columns]
    skip_columns = {id_col, "x", "y", "lon", "lat", "name", "index"}
    metric_variables = []
    for column in metric_df.columns:
        if column in skip_columns:
            continue
        values = pd.to_numeric(metric_df[column], errors="coerce")
        if values.notna().any():
            metric_variables.append(column)
    return metric_variables


def create_catchment_metric_geodataframe(
    metric_rows,
    shape_folder=None,
    mask_folder=None,
    mask_var=None,
    shape_files_by_id=None,
    mask_files_by_id=None,
    id_col="id",
):
    """Attach catchment geometries to median metric rows.

    Parameters
    ----------
    metric_rows : pandas.DataFrame
        Median metric rows with IDs.
    shape_folder : str or Path, optional
        Directory containing ID-matched shapefiles.
    mask_folder : str or Path, optional
        Directory containing ID-matched NetCDF mask files.
    mask_var : str, optional
        Mask variable name.
    shape_files_by_id : Mapping[object, Path], optional
        Explicit shape files by row ID.
    mask_files_by_id : Mapping[object, Path], optional
        Explicit mask files by row ID.
    id_col : str, optional
        ID column name.

    Returns
    -------
    geopandas.GeoDataFrame
        Metric rows with catchment geometries.
    """
    import geopandas as gpd

    rows = []
    geometries = []
    for _, row in metric_rows.iterrows():
        row_id = row[id_col]
        match_id = create_match_id(row_id)
        geometry_gdf = _read_geometry_for_id(
            match_id=match_id,
            shape_folder=shape_folder,
            mask_folder=mask_folder,
            mask_var=mask_var,
            shape_files_by_id=shape_files_by_id,
            mask_files_by_id=mask_files_by_id,
        )
        if geometry_gdf is None or geometry_gdf.empty:
            logger.warning(f"No catchment geometry found for ID {match_id}.")
            continue
        geometry = geometry_gdf.geometry.iloc[0]
        if geometry is None or geometry.is_empty:
            logger.warning(f"Empty catchment geometry for ID {match_id}.")
            continue
        rows.append(row.to_dict())
        geometries.append(geometry)

    if not rows:
        return gpd.GeoDataFrame(columns=[*list(metric_rows.columns), "geometry"])
    gdf = gpd.GeoDataFrame(rows, geometry=geometries, crs="EPSG:4326")
    return sort_geodataframe_by_area_desc(gdf)


def sort_geodataframe_by_area_desc(gdf):
    """Sort geometries by area from largest to smallest.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame to sort.

    Returns
    -------
    geopandas.GeoDataFrame
        Area-sorted GeoDataFrame.
    """
    if gdf.empty:
        return gdf
    try:
        area_values = gdf.to_crs("EPSG:6933").geometry.area
    except Exception:
        area_values = gdf.geometry.area
    sorted_gdf = gdf.assign(_plot_area=area_values).sort_values(
        "_plot_area", ascending=False
    )
    return sorted_gdf.drop(columns=["_plot_area"])


def write_catchment_median_maps(
    metric_df,
    output_dir,
    variables=None,
    shape_folder=None,
    mask_folder=None,
    mask_var=None,
    shape_files_by_id: Optional[Mapping[object, Path]] = None,
    mask_files_by_id: Optional[Mapping[object, Path]] = None,
    id_col="id",
    output_prefix="catchment_map",
    dpi=200,
    title_context=None,
):
    """Write catchment median maps for metric rows.

    Parameters
    ----------
    metric_df : pandas.DataFrame
        Metric rows.
    output_dir : str or Path
        Directory for map PNG files.
    variables : Sequence[str], optional
        Metric columns to plot.
    shape_folder : str or Path, optional
        Directory containing ID-matched shapefiles.
    mask_folder : str or Path, optional
        Directory containing ID-matched NetCDF mask files.
    mask_var : str, optional
        Mask variable name.
    shape_files_by_id : Mapping[object, Path], optional
        Explicit shapefiles by ID.
    mask_files_by_id : Mapping[object, Path], optional
        Explicit mask files by ID.
    id_col : str, optional
        ID column name.
    output_prefix : str, optional
        Output filename prefix.
    dpi : int, optional
        Output image resolution.
    title_context : str, optional
        Context added to map titles.

    Returns
    -------
    list[Path]
        Written PNG files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = calculate_metric_medians(
        metric_df, variables=variables, id_col=id_col
    )
    if metric_rows.empty:
        logger.warning("No metric rows available for catchment median maps.")
        return []
    metric_gdf = create_catchment_metric_geodataframe(
        metric_rows=metric_rows,
        shape_folder=shape_folder,
        mask_folder=mask_folder,
        mask_var=mask_var,
        shape_files_by_id=shape_files_by_id,
        mask_files_by_id=mask_files_by_id,
        id_col=id_col,
    )
    if metric_gdf.empty:
        logger.warning("No catchment geometries available for median maps.")
        return []
    variables = get_metric_variables(metric_gdf, variables=variables, id_col=id_col)
    return plot_catchment_metric_maps(
        metric_gdf=metric_gdf,
        variables=variables,
        output_dir=output_dir,
        output_prefix=output_prefix,
        dpi=dpi,
        title_context=title_context,
    )


def plot_catchment_metric_maps(
    metric_gdf,
    variables,
    output_dir,
    output_prefix="catchment_map",
    cmap="viridis",
    dpi=200,
    title_context=None,
):
    """Plot metric values on catchment polygons.

    Parameters
    ----------
    metric_gdf : geopandas.GeoDataFrame
        Metric rows with geometry.
    variables : Sequence[str]
        Metric columns to plot.
    output_dir : str or Path
        Directory for map PNG files.
    output_prefix : str, optional
        Output filename prefix.
    cmap : str, optional
        Matplotlib colormap.
    dpi : int, optional
        Output image resolution.
    title_context : str, optional
        Context added to map titles.

    Returns
    -------
    list[Path]
        Written PNG files.
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except Exception as exc:
        msg = "cartopy is required for catchment median maps."
        raise ImportError(msg) from exc

    output_dir = Path(output_dir)
    output_files = []
    plot_gdf = sort_geodataframe_by_area_desc(metric_gdf)
    min_lon, min_lat, max_lon, max_lat = plot_gdf.total_bounds
    lon_pad = (max_lon - min_lon) * 0.1 if max_lon > min_lon else 0.1
    lat_pad = (max_lat - min_lat) * 0.1 if max_lat > min_lat else 0.1

    for variable in variables:
        if variable not in plot_gdf.columns:
            logger.warning(f"Skipping catchment map for missing variable {variable}.")
            continue
        values = pd.to_numeric(plot_gdf[variable], errors="coerce")
        if values.notna().sum() == 0:
            logger.warning(
                f"Skipping catchment map for {variable}: all values are NaN."
            )
            continue
        vmin, vmax, extend = _get_metric_color_limits(variable, values.to_numpy())
        cmap_obj = plt.get_cmap(cmap).copy()
        if extend in ["min", "both"]:
            cmap_obj.set_under("lightgray")
        if extend in ["max", "both"]:
            cmap_obj.set_over("darkred")
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=False)

        fig = plt.figure(figsize=(7, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent(
            [
                min_lon - lon_pad,
                max_lon + lon_pad,
                min_lat - lat_pad,
                max_lat + lat_pad,
            ],
            crs=ccrs.PlateCarree(),
        )
        ax.add_feature(cfeature.BORDERS, linewidth=0.6)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor="0.95")
        ax.add_feature(cfeature.OCEAN, facecolor="0.97")
        plot_gdf.assign(**{variable: values}).plot(
            column=variable,
            ax=ax,
            cmap=cmap_obj,
            norm=norm,
            edgecolor="0.25",
            linewidth=0.4,
            transform=ccrs.PlateCarree(),
        )
        scalar_mappable = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        scalar_mappable.set_array([])
        cb = plt.colorbar(
            scalar_mappable,
            ax=ax,
            orientation="vertical",
            shrink=0.8,
            extend=extend,
        )
        cb.set_label(variable)
        title = f"{variable} by catchment"
        if title_context is not None:
            title = f"{title} ({title_context})"
        ax.set_title(title)
        output_file = output_dir / f"{output_prefix}_{variable}.png"
        fig.tight_layout()
        fig.savefig(output_file, dpi=dpi)
        plt.close(fig)
        output_files.append(output_file)
        logger.info(f"Wrote catchment median map to {output_file}.")
    return output_files


def _read_geometry_for_id(
    match_id,
    shape_folder=None,
    mask_folder=None,
    mask_var=None,
    shape_files_by_id=None,
    mask_files_by_id=None,
):
    """Read a shape or mask geometry for one ID.

    Parameters
    ----------
    match_id : object
        ID used for geometry lookup.
    shape_folder : str or Path, optional
        Directory containing shapefiles.
    mask_folder : str or Path, optional
        Directory containing mask files.
    mask_var : str, optional
        Mask variable name.
    shape_files_by_id : Mapping[object, Path], optional
        Explicit shape files by ID.
    mask_files_by_id : Mapping[object, Path], optional
        Explicit mask files by ID.

    Returns
    -------
    geopandas.GeoDataFrame or None
        Geometry for the ID.
    """
    if shape_files_by_id is not None and match_id in shape_files_by_id:
        return read_shape_geometry(shape_files_by_id[match_id], geometry_id=match_id)
    if mask_files_by_id is not None and match_id in mask_files_by_id:
        return read_mask_geometry(
            mask_files_by_id[match_id], mask_var=mask_var, geometry_id=match_id
        )
    shape_file = find_matching_geometry_file(shape_folder, match_id, ".shp")
    if shape_file is not None:
        return read_shape_geometry(shape_file, geometry_id=match_id)
    mask_file = find_matching_geometry_file(mask_folder, match_id, ".nc")
    if mask_file is not None:
        return read_mask_geometry(mask_file, mask_var=mask_var, geometry_id=match_id)
    return None


def _geometry_union(geometry):
    """Return a unary geometry union.

    Parameters
    ----------
    geometry : geopandas.GeoSeries
        Geometries to dissolve.

    Returns
    -------
    shapely.Geometry
        Dissolved geometry.
    """
    if hasattr(geometry, "union_all"):
        return geometry.union_all()
    return geometry.unary_union


def _get_metric_color_limits(variable, values):
    """Get color limits for a metric variable.

    Parameters
    ----------
    variable : str
        Metric variable name.
    values : Sequence[float]
        Numeric values.

    Returns
    -------
    tuple[float, float, str]
        Minimum, maximum, and colorbar extension.
    """
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        msg = f"No finite values available for {variable}."
        raise ValueError(msg)
    if variable == "kge":
        vmin, vmax = -0.5, 1.0
    elif variable == "nse":
        vmin, vmax = -0.1, 1.0
    else:
        vmin, vmax = float(np.nanmin(finite_values)), float(np.nanmax(finite_values))
    if np.isfinite(vmin) and np.isfinite(vmax) and vmin == vmax:
        vmin -= 1.0
        vmax += 1.0
    extend = "neither"
    if np.nanmin(finite_values) < vmin and np.nanmax(finite_values) > vmax:
        extend = "both"
    elif np.nanmin(finite_values) < vmin:
        extend = "min"
    elif np.nanmax(finite_values) > vmax:
        extend = "max"
    return vmin, vmax, extend
