"""Create comparison plots from one or more discharge-evaluation results CSVs.

Builds on the generic metric-comparison machinery in
`mhm_tools.post.metric_plots` (CDF/violin/catchment-map plots, summary CSV,
overview PDF) with discharge-specific defaults, and adds a spatial diff map
that is specific to discharge-evaluation's per-gauge `x`/`y` columns.

Authors
-------
- Simon Lüdke
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mhm_tools.common.constants import WMO_REGION_BOUNDS
from mhm_tools.common.logger import ErrorLogger, log_errors
from mhm_tools.common.plotter import (
    PLOT_DPI,
    add_map_colorbar,
    calculate_map_figure_size,
    create_axis_label,
    create_comparison_title,
    create_discrete_colour_norm,
    create_summary_text,
    get_metric_plot_style,
    style_map_axes,
)
from mhm_tools.common.utils import sanitize_name

logger = logging.getLogger(__name__)

DEFAULT_DISCHARGE_COMPARISON_VARIABLES = ("kge", "nse", "alpha", "beta", "gamma")
DISCHARGE_COMPARISON_PLOT_TYPES = ("cdf", "violin", "catchment-map", "map-diff")
DEFAULT_VALUE_MIN = -100
DEFAULT_VALUE_MAX = 100


def _sanitize_name(value):
    """Create a filesystem-safe name part for output file names.

    Parameters
    ----------
    value : object
        Value used in an output filename.

    Returns
    -------
    str
        Safe filename part.
    """
    return sanitize_name(value)


def _read_dfs_by_name(input_paths, input_names, file_names):
    """Read each input's results.csv file(s) into one DataFrame per label.

    Parameters
    ----------
    input_paths : Sequence[str]
        `results.csv` files or directories containing them.
    input_names : Sequence[str]
        Plot labels matching input_paths.
    file_names : str
        Glob pattern used when an input path is a directory.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Concatenated results table by input label.
    """
    from mhm_tools.post.metric_plots import get_metric_csv_files

    dfs_by_name = {}
    for input_path, input_name in zip(input_paths, input_names):
        csv_files = get_metric_csv_files(input_path, file_names=file_names)
        dfs_by_name[input_name] = pd.concat(
            [pd.read_csv(csv_file) for csv_file in csv_files], ignore_index=True
        )
    return dfs_by_name


def _write_sanitized_results_csvs(
    input_paths,
    input_names,
    file_names,
    variables,
    output_dir,
    value_min=DEFAULT_VALUE_MIN,
    value_max=DEFAULT_VALUE_MAX,
):
    """Write one sanitized `results.csv` copy per input, for plotting.

    Out-of-range values (e.g. from a diverged/blown-up simulation) can
    dominate a KDE-based violin plot's shape, or an auto-scaled axis, far
    out of proportion to how many gauges they represent. Values outside
    `[value_min, value_max]` are replaced with NaN in the listed `variables`
    columns only (gauge id/coordinates are left untouched) before any plot
    is drawn from them. For metrics in
    `mhm_tools.common.plotter.HIGHER_IS_BETTER_METRICS` (kge, nse, ...),
    values above 1 are additionally discarded regardless of `value_max`,
    since they're bounded above by 1 by construction and a value beyond
    that always signals a calculation bug rather than a real score.

    Parameters
    ----------
    input_paths : Sequence[str]
        `results.csv` files or directories containing them.
    input_names : Sequence[str]
        Plot labels matching input_paths.
    file_names : str
        Glob pattern used when an input path is a directory.
    variables : Sequence[str]
        Metric columns to sanitize.
    output_dir : str or Path
        Directory to write the sanitized CSV copies into.
    value_min : float, optional
        Lower bound of the kept value range.
    value_max : float, optional
        Upper bound of the kept value range.

    Returns
    -------
    list[Path]
        One sanitized CSV file per input, in `input_names` order.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dfs_by_name = _read_dfs_by_name(input_paths, input_names, file_names)

    from mhm_tools.common.plotter import HIGHER_IS_BETTER_METRICS

    sanitized_paths = []
    for index, input_name in enumerate(input_names):
        df = dfs_by_name[input_name]
        for variable in variables:
            if variable not in df.columns:
                continue
            values = pd.to_numeric(df[variable], errors="coerce")
            invalid = (values < value_min) | (values > value_max)
            if invalid.any():
                logger.warning(
                    f"Discarding {int(invalid.sum())} out-of-range "
                    f"{variable!r} value(s) in {input_name!r} "
                    f"(outside [{value_min}, {value_max}])."
                )
            # kge/nse/etc. are bounded above by 1 by construction; a value
            # above that is never a real (if poor) score - it signals a
            # calculation bug upstream, so it's flagged distinctly and
            # excluded rather than merely treated as a wide-range outlier.
            if variable in HIGHER_IS_BETTER_METRICS:
                impossible = values > 1.0
                if impossible.any():
                    logger.warning(
                        f"{variable!r} cannot exceed 1 by construction; "
                        f"discarding {int(impossible.sum())} value(s) > 1 in "
                        f"{input_name!r} as physically impossible - check "
                        "the upstream metric calculation."
                    )
                invalid = invalid | impossible
            df[variable] = values.where(~invalid)
        sanitized_path = output_dir / f"{index:03d}_{_sanitize_name(input_name)}.csv"
        df.to_csv(sanitized_path, index=False)
        sanitized_paths.append(sanitized_path)
    return sanitized_paths


def _sort_region_files_by_region(files):
    """Sort per-region output files by region, then by plot type.

    `write_discharge_eval_comparison_region_plots` (cdf/violin) and
    `write_discharge_metric_diff_region_maps` (map-diff) each loop over
    every region independently, so their combined output is grouped by
    function/plot-type first. This regroups it by region instead - all of
    one region's plots together - matching `WMO_REGION_BOUNDS`' order, so
    e.g. "Africa: cdf, violin, map-diff" appears as one block in the
    overview PDF instead of being split across separate cdf/violin/map-diff
    sections.

    Parameters
    ----------
    files : Sequence[str or Path]
        Per-region output file paths, in any order.

    Returns
    -------
    list[Path]
        The same files, grouped by region (WMO_REGION_BOUNDS order) and then
        by plot type (cdf, violin, map-diff).
    """
    from mhm_tools.common.constants import WMO_REGION_BOUNDS

    region_order = {region: index for index, region in enumerate(WMO_REGION_BOUNDS)}
    sanitized_to_region = {
        _sanitize_name(region): region for region in WMO_REGION_BOUNDS
    }
    plot_type_order = {"cdf": 0, "violin": 1, "map_diff": 2, "catchment_map": 3}

    def sort_key(file_path):
        name = Path(file_path).name
        region_index = len(region_order)
        for sanitized, region in sanitized_to_region.items():
            if f"_region_{sanitized}" in name:
                region_index = region_order[region]
                break
        plot_type_index = next(
            (
                index
                for prefix, index in plot_type_order.items()
                if name.startswith(prefix)
            ),
            len(plot_type_order),
        )
        return (region_index, plot_type_index, name)

    return sorted((Path(f) for f in files), key=sort_key)


def _resolve_reference_name(reference_name, input_names, dfs_by_name):
    """Resolve and validate the baseline run's label for diff maps.

    Parameters
    ----------
    reference_name : str or None
        Explicitly requested baseline label, or None to default to the
        first input.
    input_names : Sequence[str]
        Plot labels in input order.
    dfs_by_name : Mapping[str, pandas.DataFrame]
        Results tables by input label.

    Returns
    -------
    str
        Validated baseline label.
    """
    reference_name = reference_name or input_names[0]
    if reference_name not in dfs_by_name:
        msg = (
            f"--reference-name {reference_name!r} not found among input names "
            f"{input_names}."
        )
        with ErrorLogger(logger):
            raise ValueError(msg)
    return reference_name


def _get_region_extent(region):
    """Get a cartopy map extent from a discharge-evaluation region's bounds.

    Parameters
    ----------
    region : str
        Region name, a key of `constants.WMO_REGION_BOUNDS`.

    Returns
    -------
    tuple[float, float, float, float]
        Lon min, lon max, lat min, lat max.
    """
    from mhm_tools.common.constants import WMO_REGION_BOUNDS

    bounds = WMO_REGION_BOUNDS[region]
    lon_slice, lat_slice = bounds["lon_slice"], bounds["lat_slice"]
    return lon_slice.start, lon_slice.stop, lat_slice.start, lat_slice.stop


def _prepare_discharge_metric_diff_data(
    reference_df, other_df, variable, lon_col="x", lat_col="y"
):
    """Get per-gauge coordinates and metric differences between two runs.

    Parameters
    ----------
    reference_df : pandas.DataFrame
        Baseline run's results table; must contain `id`, `lon_col`, `lat_col`.
    other_df : pandas.DataFrame
        Other run's results table; must contain `id`.
    variable : str
        Metric column name to difference.
    lon_col : str, optional
        Gauge longitude/x column name.
    lat_col : str, optional
        Gauge latitude/y column name.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Longitudes, latitudes, and `other - reference` metric differences for
        gauges present with finite values in both inputs.
    """
    if "id" not in reference_df.columns or "id" not in other_df.columns:
        msg = "Both inputs need an 'id' column to match gauges for a diff map."
        with ErrorLogger(logger):
            raise ValueError(msg)
    empty = (np.array([]), np.array([]), np.array([]))
    if variable not in reference_df.columns or variable not in other_df.columns:
        logger.warning(f"Column {variable!r} missing in reference or other input.")
        return empty

    ref_subset = reference_df[["id", lon_col, lat_col, variable]].rename(
        columns={variable: "__ref_value__"}
    )
    other_subset = other_df[["id", variable]].rename(
        columns={variable: "__other_value__"}
    )
    merged = ref_subset.merge(other_subset, on="id", how="inner")
    if merged.empty:
        logger.warning("No overlapping gauge ids between reference and other input.")
        return empty

    lons = pd.to_numeric(merged[lon_col], errors="coerce").to_numpy(dtype=float)
    lats = pd.to_numeric(merged[lat_col], errors="coerce").to_numpy(dtype=float)
    ref_values = pd.to_numeric(merged["__ref_value__"], errors="coerce")
    other_values = pd.to_numeric(merged["__other_value__"], errors="coerce")
    diff = (other_values - ref_values).to_numpy(dtype=float)

    valid = np.isfinite(lons) & np.isfinite(lats) & np.isfinite(diff)
    return lons[valid], lats[valid], diff[valid]


@log_errors(raise_exceptions=True)
def plot_discharge_metric_diff_map(
    reference_df,
    other_df,
    reference_name,
    other_name,
    variable,
    output_dir,
    lon_col="x",
    lat_col="y",
    cmap="coolwarm_r",
    point_size=6,
    dpi=PLOT_DPI,
    extent=None,
    region=None,
):
    """Plot a gauge map of one metric's difference between two runs.

    Parameters
    ----------
    reference_df : pandas.DataFrame
        Baseline run's results table.
    other_df : pandas.DataFrame
        Other run's results table, compared against the baseline.
    reference_name : str
        Baseline run's plot label.
    other_name : str
        Other run's plot label.
    variable : str
        Metric column name to difference.
    output_dir : str or Path
        Directory for the output PNG file.
    lon_col : str, optional
        Gauge longitude/x column name.
    lat_col : str, optional
        Gauge latitude/y column name.
    cmap : str, optional
        Diverging colormap name, centered on zero.
    point_size : int, optional
        Base marker size before density scaling.
    dpi : int, optional
        Output image resolution.
    extent : tuple[float, float, float, float], optional
        Explicit (lon_min, lon_max, lat_min, lat_max) map extent, e.g. from a
        discharge-evaluation region's bounds. Defaults to the gauge data's
        own extent, padded by 10%.
    region : str, optional
        Region name, added to the plot title and output file name.

    Returns
    -------
    Path or None
        Written PNG file, or None if there was no overlapping data to plot.
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except Exception as exc:
        logger.error("cartopy is required for diff maps but could not be imported.")
        raise exc

    lons, lats, diff = _prepare_discharge_metric_diff_data(
        reference_df, other_df, variable, lon_col=lon_col, lat_col=lat_col
    )
    if lons.size == 0:
        logger.warning(
            f"No overlapping finite {variable!r} values between "
            f"{other_name!r} and {reference_name!r}"
            f"{f' in region {region!r}' if region else ''}; skipping diff map."
        )
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    style = get_metric_plot_style("diff")
    style["cmap"] = cmap
    cmap_obj, norm, bounds, extend, ticks = create_discrete_colour_norm(
        diff, bounds_type="data", **style
    )

    size_scale = 1.0
    if len(diff) > 200:
        size_scale = (200 / len(diff)) ** 0.5
    point_size = max(4, point_size * size_scale)

    if extent is None:
        min_lon, max_lon = np.nanmin(lons), np.nanmax(lons)
        min_lat, max_lat = np.nanmin(lats), np.nanmax(lats)
        lon_pad = (max_lon - min_lon) * 0.1 or 0.1
        lat_pad = (max_lat - min_lat) * 0.1 or 0.1
        extent = (
            min_lon - lon_pad,
            max_lon + lon_pad,
            min_lat - lat_pad,
            max_lat + lat_pad,
        )

    fig, ax = plt.subplots(
        figsize=calculate_map_figure_size(extent),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="0.97")
    ax.add_feature(cfeature.OCEAN, facecolor="0.85")

    sc = ax.scatter(
        lons,
        lats,
        c=diff,
        cmap=cmap_obj,
        norm=norm,
        s=point_size,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.25,
        transform=ccrs.PlateCarree(),
    )
    add_map_colorbar(
        fig, ax, sc, bounds, extend, ticks, create_axis_label(f"Δ{variable}")
    )
    style_map_axes(ax)
    region_suffix = f"_region_{_sanitize_name(region)}" if region else ""
    region_title = f" - {region}" if region else ""
    fig.suptitle(
        f"{create_comparison_title(other_name, reference_name)}{region_title}",
        fontweight="normal",
        fontsize="x-large",
    )
    ax.set_title(
        f"Δ{variable} = {other_name} - {reference_name} "
        f"({create_summary_text(diff, bounds=bounds)})"
    )
    fig.tight_layout()
    output_file = (
        output_dir / f"map_diff_{variable}_{_sanitize_name(other_name)}_vs_"
        f"{_sanitize_name(reference_name)}{region_suffix}.png"
    )
    fig.savefig(output_file, dpi=dpi)
    plt.close(fig)
    logger.info(f"Wrote difference map to {output_file}")
    return output_file


def write_discharge_metric_diff_maps(
    input_paths,
    output_dir,
    variables=None,
    input_names=None,
    file_names="results.csv",
    reference_name=None,
    cmap="coolwarm_r",
    dpi=PLOT_DPI,
):
    """Write per-gauge metric diff maps comparing runs against a baseline.

    Parameters
    ----------
    input_paths : Sequence[str]
        `results.csv` files or directories containing them.
    output_dir : str or Path
        Directory for output PNG files.
    variables : Sequence[str], optional
        Metric columns to difference. Defaults to
        `DEFAULT_DISCHARGE_COMPARISON_VARIABLES`.
    input_names : Sequence[str], optional
        Plot labels matching input_paths. Defaults to path names.
    file_names : str, optional
        Glob pattern used when an input path is a directory.
    reference_name : str, optional
        Baseline run's label to difference all other runs against. Defaults
        to the first input.
    cmap : str, optional
        Diverging colormap name, centered on zero.
    dpi : int, optional
        Output image resolution.

    Returns
    -------
    list[Path]
        Written PNG files.
    """
    from mhm_tools.post.metric_plots import get_metric_input_names

    variables = (
        list(variables) if variables else list(DEFAULT_DISCHARGE_COMPARISON_VARIABLES)
    )
    input_names = get_metric_input_names(input_paths, input_names)
    if len(input_names) < 2:
        logger.warning("map-diff needs at least two input paths; skipping.")
        return []

    dfs_by_name = _read_dfs_by_name(input_paths, input_names, file_names)
    reference_name = _resolve_reference_name(reference_name, input_names, dfs_by_name)
    reference_df = dfs_by_name[reference_name]

    output_files = []
    for other_name, other_df in dfs_by_name.items():
        if other_name == reference_name:
            continue
        for variable in variables:
            output_file = plot_discharge_metric_diff_map(
                reference_df=reference_df,
                other_df=other_df,
                reference_name=reference_name,
                other_name=other_name,
                variable=variable,
                output_dir=output_dir,
                cmap=cmap,
                dpi=dpi,
            )
            if output_file is not None:
                output_files.append(output_file)
                logger.info(f"Wrote diff map to {output_file}")
    return output_files


def write_discharge_metric_diff_region_maps(
    input_paths,
    output_dir,
    variables=None,
    input_names=None,
    file_names="results.csv",
    reference_name=None,
    cmap="coolwarm_r",
    dpi=PLOT_DPI,
):
    """Write one metric diff map per discharge-evaluation region with data.

    Each gauge's region is derived from its id (see
    `discharge_evaluation.get_region_from_id`); each region's map is zoomed
    to that region's `constants.WMO_REGION_BOUNDS` extent instead of
    auto-fitting to the (possibly sparse) gauge locations within it.

    Parameters
    ----------
    input_paths : Sequence[str]
        `results.csv` files or directories containing them.
    output_dir : str or Path
        Directory for output PNG files.
    variables : Sequence[str], optional
        Metric columns to difference. Defaults to
        `DEFAULT_DISCHARGE_COMPARISON_VARIABLES`.
    input_names : Sequence[str], optional
        Plot labels matching input_paths. Defaults to path names.
    file_names : str, optional
        Glob pattern used when an input path is a directory.
    reference_name : str, optional
        Baseline run's label to difference all other runs against. Defaults
        to the first input.
    cmap : str, optional
        Diverging colormap name, centered on zero.
    dpi : int, optional
        Output image resolution.

    Returns
    -------
    list[Path]
        Written PNG files, one per region/other-run/variable combination
        that has overlapping data.
    """
    from mhm_tools.post.discharge_evaluation import get_region_from_id
    from mhm_tools.post.metric_plots import get_metric_input_names

    variables = (
        list(variables) if variables else list(DEFAULT_DISCHARGE_COMPARISON_VARIABLES)
    )
    input_names = get_metric_input_names(input_paths, input_names)
    if len(input_names) < 2:
        logger.warning("map-diff needs at least two input paths; skipping.")
        return []

    dfs_by_name = _read_dfs_by_name(input_paths, input_names, file_names)
    for df in dfs_by_name.values():
        df["region"] = df["id"].apply(get_region_from_id)
    reference_name = _resolve_reference_name(reference_name, input_names, dfs_by_name)
    reference_df = dfs_by_name[reference_name]

    output_files = []
    for region in WMO_REGION_BOUNDS:
        extent = _get_region_extent(region)
        region_reference_df = reference_df[reference_df["region"] == region]
        for other_name, other_df in dfs_by_name.items():
            if other_name == reference_name:
                continue
            region_other_df = other_df[other_df["region"] == region]
            for variable in variables:
                output_file = plot_discharge_metric_diff_map(
                    reference_df=region_reference_df,
                    other_df=region_other_df,
                    reference_name=reference_name,
                    other_name=other_name,
                    variable=variable,
                    output_dir=output_dir,
                    cmap=cmap,
                    dpi=dpi,
                    extent=extent,
                    region=region,
                )
                if output_file is not None:
                    output_files.append(output_file)
                    logger.info(f"Wrote regional diff map to {output_file}")
    return output_files


def write_discharge_eval_comparison_region_plots(
    input_paths,
    output_dir,
    variables=None,
    input_names=None,
    file_names="results.csv",
    plot_types=("cdf",),
    dpi=PLOT_DPI,
    axis_limits_by_variable=None,
):
    """Write one CDF/violin comparison plot per discharge-evaluation region.

    Each gauge's region is derived from its id (see
    `discharge_evaluation.get_region_from_id`). Mirrors the styling of the
    global comparison plots (per-run color and, for CDF, line style, via
    `metric_plots`'s "label" grouping).

    Parameters
    ----------
    input_paths : Sequence[str]
        `results.csv` files or directories containing them.
    output_dir : str or Path
        Directory for output PNG files.
    variables : Sequence[str], optional
        Metric columns to plot. Defaults to
        `DEFAULT_DISCHARGE_COMPARISON_VARIABLES`.
    input_names : Sequence[str], optional
        Plot labels matching input_paths. Defaults to path names.
    file_names : str, optional
        Glob pattern used when an input path is a directory.
    plot_types : Sequence[str], optional
        Which of "cdf"/"violin" to create per-region plots for.
    dpi : int, optional
        Output image resolution.
    axis_limits_by_variable : Mapping[str, Sequence[float]], optional
        Explicit value-axis limits by variable, matching the global plots.

    Returns
    -------
    list[Path]
        Written PNG files, one per plot-type/region/variable combination
        that has data in at least one input.
    """
    from mhm_tools.common.plotter import (
        plot_metric_cdf_comparison,
        plot_metric_violin_comparison,
    )
    from mhm_tools.post.discharge_evaluation import get_region_from_id
    from mhm_tools.post.metric_plots import (
        _create_metric_label_metadata,
        _get_metric_plot_colors_by_label,
        get_metric_input_names,
    )

    variables = (
        list(variables) if variables else list(DEFAULT_DISCHARGE_COMPARISON_VARIABLES)
    )
    input_names = get_metric_input_names(input_paths, input_names)
    dfs_by_name = _read_dfs_by_name(input_paths, input_names, file_names)
    for df in dfs_by_name.values():
        df["region"] = df["id"].apply(get_region_from_id)

    label_metadata = _create_metric_label_metadata(input_names=input_names)
    colors_by_label = _get_metric_plot_colors_by_label(
        input_names=input_names, label_metadata=label_metadata, color_by="label"
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = []
    for region in WMO_REGION_BOUNDS:
        for variable in variables:
            values_by_label = {}
            for name, df in dfs_by_name.items():
                if variable not in df.columns:
                    continue
                region_values = pd.to_numeric(
                    df.loc[df["region"] == region, variable], errors="coerce"
                )
                region_values = region_values[np.isfinite(region_values)]
                if region_values.empty:
                    continue
                values_by_label[name] = region_values.to_numpy(dtype=float)
            if not values_by_label:
                continue
            axis_limits = (axis_limits_by_variable or {}).get(variable)
            if "cdf" in plot_types:
                output_file = (
                    output_dir / f"cdf_{variable}_region_{_sanitize_name(region)}.png"
                )
                plot_metric_cdf_comparison(
                    values_by_label=values_by_label,
                    variable_name=variable,
                    output_file=output_file,
                    title=f"CDF of {variable}: {region}",
                    x_limits=axis_limits,
                    dpi=dpi,
                    colors=colors_by_label,
                    show_median_line=True,
                )
                output_files.append(output_file)
                logger.info(f"Wrote regional CDF plot to {output_file}")
            if "violin" in plot_types:
                output_file = (
                    output_dir
                    / f"violin_{variable}_region_{_sanitize_name(region)}.png"
                )
                plot_metric_violin_comparison(
                    values_by_label=values_by_label,
                    variable_name=variable,
                    output_file=output_file,
                    title=f"Distribution of {variable}: {region}",
                    y_limits=axis_limits,
                    dpi=dpi,
                    colors=colors_by_label,
                )
                output_files.append(output_file)
                logger.info(f"Wrote regional violin plot to {output_file}")
    return output_files


def write_discharge_eval_comparison_region_catchment_maps(
    input_paths,
    output_dir,
    variables=None,
    input_names=None,
    file_names="results.csv",
    shape_folder=None,
    mask_folder=None,
    mask_var=None,
    dpi=PLOT_DPI,
):
    """Write one catchment-map set per discharge-evaluation region.

    Each gauge's region is derived from its id (see
    `discharge_evaluation.get_region_from_id`). Each run's per-region rows
    are aggregated (median by id, via `write_catchment_median_maps`) and
    plotted independently, matching how the global catchment-map is one map
    per run rather than a run-to-run comparison. Each map is framed by that
    region's `constants.WMO_REGION_BOUNDS` extent rather than
    auto-fitting to the matched catchment geometries' own bounds, which can
    balloon out to (near) the whole globe if a matched geometry is
    malformed or not in the expected lon/lat CRS.

    Parameters
    ----------
    input_paths : Sequence[str]
        `results.csv` files or directories containing them.
    output_dir : str or Path
        Directory for output PNG files.
    variables : Sequence[str], optional
        Metric columns to plot. Defaults to
        `DEFAULT_DISCHARGE_COMPARISON_VARIABLES`.
    input_names : Sequence[str], optional
        Plot labels matching input_paths. Defaults to path names.
    file_names : str, optional
        Glob pattern used when an input path is a directory.
    shape_folder : str or Path, optional
        Folder with shapefiles matched by gauge id.
    mask_folder : str or Path, optional
        Folder with NetCDF masks matched by gauge id.
    mask_var : str, optional
        Mask variable name.
    dpi : int, optional
        Output image resolution.

    Returns
    -------
    list[Path]
        Written PNG files, one per region/run/variable combination that
        has data and a matching shape/mask.
    """
    from mhm_tools.common.catchment_maps import write_catchment_median_maps
    from mhm_tools.post.discharge_evaluation import get_region_from_id
    from mhm_tools.post.metric_plots import get_metric_input_names

    variables = (
        list(variables) if variables else list(DEFAULT_DISCHARGE_COMPARISON_VARIABLES)
    )
    input_names = get_metric_input_names(input_paths, input_names)
    dfs_by_name = _read_dfs_by_name(input_paths, input_names, file_names)
    for df in dfs_by_name.values():
        df["region"] = df["id"].apply(get_region_from_id)
    multiple_inputs = len(input_names) > 1

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = []
    for region in WMO_REGION_BOUNDS:
        region_extent = _get_region_extent(region)
        for name in input_names:
            region_df = dfs_by_name[name]
            region_df = region_df[region_df["region"] == region]
            if region_df.empty:
                continue
            output_prefix = f"catchment_map_region_{_sanitize_name(region)}"
            title_context = region
            if multiple_inputs:
                output_prefix = f"{output_prefix}_{_sanitize_name(name)}"
                title_context = f"{region}, {name}"
            output_files.extend(
                write_catchment_median_maps(
                    metric_df=region_df,
                    output_dir=output_dir,
                    variables=variables,
                    shape_folder=shape_folder,
                    mask_folder=mask_folder,
                    mask_var=mask_var,
                    output_prefix=output_prefix,
                    dpi=dpi,
                    title_context=title_context,
                    extent=region_extent,
                )
            )
            logger.info(f"Wrote regional catchment maps for {title_context}")
    return output_files


def write_discharge_eval_comparison_plots(
    input_paths,
    output_dir,
    variables=None,
    input_names=None,
    file_names="results.csv",
    plot_types=None,
    dpi=PLOT_DPI,
    shape_folder=None,
    mask_folder=None,
    mask_var=None,
    reference_name=None,
    map_diff_cmap="coolwarm_r",
    map_diff_dpi=PLOT_DPI,
):
    """Write discharge-evaluation comparison plots from `results.csv` files.

    Parameters
    ----------
    input_paths : Sequence[str]
        `results.csv` files or directories containing them.
    output_dir : str or Path
        Directory for output files.
    variables : Sequence[str], optional
        Metric columns to plot. Defaults to
        `DEFAULT_DISCHARGE_COMPARISON_VARIABLES`.
    input_names : Sequence[str], optional
        Plot labels matching input_paths. Defaults to path names.
    file_names : str, optional
        Glob pattern used when an input path is a directory.
    plot_types : Sequence[str], optional
        Plot types to create; see `DISCHARGE_COMPARISON_PLOT_TYPES`. Defaults
        to `cdf`, `map-diff`, plus `catchment-map` whenever
        `shape_folder` or `mask_folder` is given.
    dpi : int, optional
        Output image resolution for cdf/violin plots.
    shape_folder : str or Path, optional
        Folder with shapefiles matched by gauge id, for catchment maps.
    mask_folder : str or Path, optional
        Folder with NetCDF masks matched by gauge id, for catchment maps.
    mask_var : str, optional
        Mask variable name for catchment maps.
    reference_name : str, optional
        Baseline run's label for diff maps. Defaults to the first input.
    map_diff_cmap : str, optional
        Diverging colormap name for diff maps.
    map_diff_dpi : int, optional
        Output image resolution for diff maps.

    Returns
    -------
    list[Path]
        Written files.
    """
    import tempfile

    from mhm_tools.common.plotter import (
        create_metric_summary_rows,
        write_metric_plot_overview_pdf,
    )
    from mhm_tools.post.discharge_evaluation import get_discharge_cdf_x_limits
    from mhm_tools.post.metric_plots import (
        get_metric_input_names,
        get_metric_values_by_input,
        write_metric_plots,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variables = (
        list(variables) if variables else list(DEFAULT_DISCHARGE_COMPARISON_VARIABLES)
    )
    if plot_types is None:
        plot_types = ["cdf", "map-diff"]
        if shape_folder is not None or mask_folder is not None:
            plot_types.append("catchment-map")
    plot_types = list(plot_types)
    input_names = get_metric_input_names(input_paths, input_names)

    shared_plot_types = [
        pt for pt in plot_types if pt in ("cdf", "violin", "catchment-map")
    ]
    distribution_plot_types = [
        pt for pt in shared_plot_types if pt in ("cdf", "violin")
    ]
    # Kept in two groups - not one flat list - so the final overview PDF can
    # be ordered "global plots first, then per-region" regardless of which
    # plot types were requested or in what order they were generated.
    global_files = []
    region_files = []
    values_by_variable = {}
    with tempfile.TemporaryDirectory(prefix="discharge_eval_comparison_") as tmp_dir:
        # Sanitize once, up front: a diverged/blown-up run's extreme outlier
        # values would otherwise dominate a KDE-based violin's shape or an
        # auto-scaled axis, far out of proportion to how many gauges they
        # represent. All plot types below read from these sanitized copies.
        sanitized_paths = _write_sanitized_results_csvs(
            input_paths=input_paths,
            input_names=input_names,
            file_names=file_names,
            variables=variables,
            output_dir=tmp_dir,
        )

        if shared_plot_types:
            axis_limits_by_variable = None
            if distribution_plot_types:
                values_by_variable = get_metric_values_by_input(
                    input_paths=sanitized_paths,
                    variables=variables,
                    input_names=input_names,
                )
                axis_limits_by_variable = {
                    variable: get_discharge_cdf_x_limits(
                        variable, np.concatenate(list(values_by_input.values()))
                    )
                    for variable, values_by_input in values_by_variable.items()
                }
            global_files.extend(
                write_metric_plots(
                    input_paths=sanitized_paths,
                    variables=variables,
                    output_dir=output_dir,
                    input_names=input_names,
                    plot_types=shared_plot_types,
                    dpi=dpi,
                    shape_folder=shape_folder,
                    mask_folder=mask_folder,
                    mask_var=mask_var,
                    style_by=None,
                    axis_limits_by_variable=axis_limits_by_variable,
                )
            )
            if distribution_plot_types:
                region_files.extend(
                    write_discharge_eval_comparison_region_plots(
                        input_paths=sanitized_paths,
                        output_dir=output_dir,
                        variables=variables,
                        input_names=input_names,
                        plot_types=distribution_plot_types,
                        dpi=dpi,
                        axis_limits_by_variable=axis_limits_by_variable,
                    )
                )
            if "catchment-map" in shared_plot_types:
                region_files.extend(
                    write_discharge_eval_comparison_region_catchment_maps(
                        input_paths=sanitized_paths,
                        output_dir=output_dir,
                        variables=variables,
                        input_names=input_names,
                        shape_folder=shape_folder,
                        mask_folder=mask_folder,
                        mask_var=mask_var,
                        dpi=dpi,
                    )
                )
        if "map-diff" in plot_types:
            global_files.extend(
                write_discharge_metric_diff_maps(
                    input_paths=sanitized_paths,
                    output_dir=output_dir,
                    variables=variables,
                    input_names=input_names,
                    reference_name=reference_name,
                    cmap=map_diff_cmap,
                    dpi=map_diff_dpi,
                )
            )
            region_files.extend(
                write_discharge_metric_diff_region_maps(
                    input_paths=sanitized_paths,
                    output_dir=output_dir,
                    variables=variables,
                    input_names=input_names,
                    reference_name=reference_name,
                    cmap=map_diff_cmap,
                    dpi=map_diff_dpi,
                )
            )

    output_files = global_files + _sort_region_files_by_region(region_files)
    # write_metric_plots already wrote its own overview PDF/summary CSV from
    # only the plots it knows about (global cdf/violin/catchment-map); with
    # region and map-diff plots generated by separate calls afterward, that
    # overview is incomplete. Rebuild it here from everything, in the
    # "global first, then per-region" order the two lists above preserve -
    # this overwrites the same file rather than leaving a stale duplicate.
    png_files = [f for f in output_files if Path(f).suffix.lower() == ".png"]
    if png_files:
        summary_rows = create_metric_summary_rows(values_by_variable)
        overview_file = write_metric_plot_overview_pdf(
            output_file=output_dir / "metric_plots_overview.pdf",
            plot_files=png_files,
            summary_rows=summary_rows,
            title="Discharge evaluation comparison overview",
        )
        if overview_file not in output_files:
            output_files.append(overview_file)
    return output_files
