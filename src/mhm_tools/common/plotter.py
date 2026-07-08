"""
Plotting utilities for creating geospatial maps with Cartopy and Matplotlib.

Includes functions for plotting:
- Constant data maps with a legend patch.
- Discrete data maps with colorbars and extensions.
- General wrapper `plot_map` that auto-selects plotting strategy.

Authors
-------
- Jeisson Leal
- Simon Lüdke
"""

import logging
from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm, ListedColormap

logger = logging.getLogger(__name__)

ONE_CENTERED_METRICS = {
    "alpha",
    "beta",
    "gamma",
    "rho",
    "rs",
    "sigma",
    "general-beta",
    "spatial-alpha",
    "spatial-gamma",
    "temporal-alpha",
    "temporal-gamma",
}
ZERO_CENTERED_METRICS = {
    "mean-bias",
    "nrmse",
    "sigma-error",
    "wd",
}
HIGHER_IS_BETTER_METRICS = {
    "comb",
    "esp",
    "kge",
    "mspaef",
    "nse",
    "spaef",
}
LOWER_IS_BETTER_METRICS = {
    "waspaef",
}
METRIC_SUMMARY_VALUE_COLUMNS = {"value", "min", "max", "mean", "median"}

try:  # cartopy is optional
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:  # pragma: no cover - cartopy may be absent in some installs
    ccrs = None
    cfeature = None


def _require_cartopy() -> None:
    """Raise an informative error if cartopy is not installed."""
    if ccrs is None or cfeature is None:
        msg = (
            "cartopy is required for geospatial plotting but is not installed. "
            "Install with `pip install cartopy` to enable plotting functions."
        )
        raise ImportError(msg)


def calculate_cdf_values(values: Sequence[float]):
    """Calculate sorted values and CDF coordinates.

    Parameters
    ----------
    values : Sequence[float]
        Numeric values to sort.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Sorted values and matching CDF coordinates.
    """
    sorted_values = np.sort(np.asarray(values, dtype=float))
    cdf_values = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    return sorted_values, cdf_values


def plot_cdf_values(
    ax,
    values: Sequence[float],
    label: str,
    color: Optional[str] = None,
    linestyle="-",
    cdf_values: Optional[Sequence[float]] = None,
    marker_size: int = 16,
    draw_line: bool = True,
    draw_points: bool = True,
):
    """Plot one CDF series on an existing axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    values : Sequence[float]
        Numeric values for the CDF line.
    label : str
        Legend label for the series.
    color : str, optional
        Matplotlib color for line and points.
    linestyle : object, optional
        Matplotlib linestyle for the CDF line.
    cdf_values : Sequence[float], optional
        Precomputed CDF coordinates.
    marker_size : int, optional
        Scatter marker size.
    draw_line : bool, optional
        Draw the CDF line when true.
    draw_points : bool, optional
        Draw CDF points when true.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Plotted x values and CDF y values.
    """
    sorted_values = np.asarray(values, dtype=float)
    if cdf_values is None:
        sorted_values, cdf_values = calculate_cdf_values(sorted_values)
    else:
        cdf_values = np.asarray(cdf_values, dtype=float)
    if draw_line:
        ax.plot(
            sorted_values,
            cdf_values,
            color=color,
            linestyle=linestyle,
            linewidth=1.0,
        )
    if draw_points:
        ax.scatter(
            sorted_values,
            cdf_values,
            s=marker_size,
            color=color,
            label=label,
        )
    return sorted_values, cdf_values


def plot_metric_cdf_comparison(
    values_by_label: Mapping[str, Sequence[float]],
    variable_name: str,
    output_file: Path,
    title: Optional[str] = None,
    x_limits: Optional[Sequence[float]] = None,
    dpi: int = 450,
    colors: Optional[Mapping[str, str]] = None,
    linestyles: Optional[Mapping[str, object]] = None,
    show_median_line: bool = False,
) -> None:
    """Create a comparison CDF plot for one metric variable.

    Parameters
    ----------
    values_by_label : Mapping[str, Sequence[float]]
        Numeric metric values grouped by plot label.
    variable_name : str
        Metric column name shown on the x-axis.
    output_file : Path
        PNG file to write.
    title : str, optional
        Plot title. Defaults to a CDF title for the variable.
    x_limits : Sequence[float], optional
        Lower and upper x-axis limits.
    dpi : int, optional
        Output image resolution.
    colors : Mapping[str, str], optional
        Optional colors by label.
    linestyles : Mapping[str, object], optional
        Optional line styles by label.
    show_median_line : bool, optional
        Draw vertical median lines when true.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted_any = False
    series_count = len(values_by_label)
    tab20_colors = plt.get_cmap("tab20").colors
    continuous_cmap = plt.get_cmap("nipy_spectral")
    for color_index, (label, values) in enumerate(values_by_label.items()):
        values_array = np.asarray(values, dtype=float)
        values_array = values_array[np.isfinite(values_array)]
        if values_array.size == 0:
            continue
        median_value = float(np.nanmedian(values_array))
        if show_median_line:
            label_with_count = (
                f"{label} (n={values_array.size}, median={median_value:.3f})"
            )
        else:
            label_with_count = f"{label} (n={values_array.size})"
        if colors is not None and label in colors:
            color = colors[label]
        elif series_count <= len(tab20_colors):
            color = tab20_colors[color_index % len(tab20_colors)]
        else:
            color_fraction = 0.05 + (0.9 * color_index / max(series_count - 1, 1))
            color = continuous_cmap(color_fraction)
        linestyle = "-"
        if linestyles is not None and label in linestyles:
            linestyle = linestyles[label]
        plot_cdf_values(
            ax,
            values_array,
            label=label_with_count,
            color=color,
            linestyle=linestyle,
        )
        if show_median_line:
            ax.axvline(
                median_value,
                color=color,
                linestyle="dotted",
                linewidth=1,
            )
        plotted_any = True
    if not plotted_any:
        plt.close(fig)
        msg = f"No finite values available for {variable_name}."
        raise ValueError(msg)

    ax.set_title(title or f"CDF of {variable_name}")
    ax.set_xlabel(variable_name)
    ax.set_ylabel("CDF")
    ax.set_ylim(0.0, 1.01)
    if x_limits is not None:
        ax.set_xlim(x_limits[0], x_limits[1])
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_file, dpi=dpi)
    plt.close(fig)


def plot_metric_violin_comparison(
    values_by_label: Mapping[str, Sequence[float]],
    variable_name: str,
    output_file: Path,
    title: Optional[str] = None,
    dpi: int = 450,
    colors: Optional[Mapping[str, str]] = None,
) -> None:
    """Create a violin plot for one metric variable.

    Parameters
    ----------
    values_by_label : Mapping[str, Sequence[float]]
        Numeric metric values grouped by plot label.
    variable_name : str
        Metric column name shown on the y-axis.
    output_file : Path
        PNG file to write.
    title : str, optional
        Plot title. Defaults to a violin title for the variable.
    dpi : int, optional
        Output image resolution.
    colors : Mapping[str, str], optional
        Optional colors by label.

    Returns
    -------
    None
    """
    labels = []
    finite_values = []
    violin_colors = []
    for label, values in values_by_label.items():
        values_array = np.asarray(values, dtype=float)
        values_array = values_array[np.isfinite(values_array)]
        if values_array.size == 0:
            continue
        median_value = float(np.nanmedian(values_array))
        labels.append(f"{label}\n(n={values_array.size}, median={median_value:.3f})")
        finite_values.append(values_array)
        violin_colors.append(colors.get(label) if colors is not None else None)
    if not finite_values:
        msg = f"No finite values available for {variable_name}."
        raise ValueError(msg)

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))
    parts = ax.violinplot(
        finite_values,
        showmeans=False,
        showmedians=True,
        showextrema=True,
    )
    for body, color in zip(parts["bodies"], violin_colors):
        body.set_alpha(0.75)
        if color is not None:
            body.set_facecolor(color)
            body.set_edgecolor(color)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(variable_name)
    ax.set_title(title or f"Distribution of {variable_name}")
    ax.grid(axis="y", linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_file, dpi=dpi)
    plt.close(fig)


def create_metric_summary_rows(
    values_by_variable: Mapping[str, Mapping[str, Sequence[float]]],
):
    """Create summary rows for metric values.

    Parameters
    ----------
    values_by_variable : Mapping[str, Mapping[str, Sequence[float]]]
        Metric values grouped by variable and realisation label.

    Returns
    -------
    list[dict[str, object]]
        Summary rows with value or distribution statistics.
    """
    summary_rows = []
    for variable_name, values_by_label in values_by_variable.items():
        for label, values in values_by_label.items():
            values_array = np.asarray(values, dtype=float)
            values_array = values_array[np.isfinite(values_array)]
            if values_array.size == 0:
                continue
            if values_array.size == 1:
                summary_rows.append(
                    {
                        "realisation": label,
                        "variable": variable_name,
                        "value": float(values_array[0]),
                    }
                )
                continue
            summary_rows.append(
                {
                    "realisation": label,
                    "variable": variable_name,
                    "n": int(values_array.size),
                    "min": float(np.min(values_array)),
                    "max": float(np.max(values_array)),
                    "mean": float(np.mean(values_array)),
                    "median": float(np.median(values_array)),
                }
            )
    return summary_rows


def write_metric_plot_overview_pdf(
    output_file: Path,
    plot_files: Sequence[Path],
    summary_rows: Sequence[Mapping[str, object]],
    append_pdf_files: Optional[Sequence[Path]] = None,
    title: Optional[str] = None,
    dpi: int = 150,
) -> Path:
    """Write a PDF overview with summary table and plot pages.

    Parameters
    ----------
    output_file : str or Path
        PDF file to write.
    plot_files : Sequence[str or Path]
        Plot image files to include.
    summary_rows : Sequence[Mapping[str, object]]
        Metric summary rows for the table page.
    append_pdf_files : Sequence[str or Path], optional
        Existing PDF files appended after the generated overview pages.
    title : str, optional
        Overview title.
    dpi : int, optional
        PDF image resolution.

    Returns
    -------
    Path
        Written PDF file.
    """
    output_file = Path(output_file)
    existing_plot_files = [Path(plot_file) for plot_file in plot_files]
    existing_plot_files = [
        plot_file for plot_file in existing_plot_files if plot_file.is_file()
    ]
    if not existing_plot_files and not summary_rows:
        msg = "No plots or summary rows available for overview PDF."
        raise ValueError(msg)
    append_pdf_files = append_pdf_files or []

    with PdfPages(output_file) as pdf:
        if summary_rows:
            _write_metric_summary_table_pages(
                pdf=pdf,
                summary_rows=summary_rows,
                title=title or "Metric plot overview",
            )
        for plot_file in existing_plot_files:
            image = plt.imread(plot_file)
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(plot_file.name)
            fig.tight_layout()
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)
    _append_pdf_files_to_pdf(output_file, append_pdf_files)
    return output_file


def _append_pdf_files_to_pdf(output_file: Path, append_pdf_files: Sequence[Path]):
    """Append existing PDF files to an overview PDF.

    Parameters
    ----------
    output_file : Path
        PDF file that receives appended pages.
    append_pdf_files : Sequence[str or Path]
        Existing PDF files to append.

    Returns
    -------
    None
    """
    append_pdf_files = [
        Path(pdf_file)
        for pdf_file in append_pdf_files
        if Path(pdf_file).is_file() and Path(pdf_file) != output_file
    ]
    if not append_pdf_files:
        return
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        logger.warning("pypdf is required to append hydrograph PDFs to the overview.")
        return

    temporary_file = output_file.with_name(
        f"{output_file.stem}_tmp{output_file.suffix}"
    )
    try:
        writer = PdfWriter()
        for pdf_file in [output_file, *append_pdf_files]:
            reader = PdfReader(str(pdf_file))
            for page in reader.pages:
                writer.add_page(page)
        with temporary_file.open("wb") as output_stream:
            writer.write(output_stream)
        temporary_file.replace(output_file)
    except Exception as exc:
        logger.warning(f"Could not append PDF files to {output_file}: {exc}")
        if temporary_file.is_file():
            temporary_file.unlink()


def _write_metric_summary_table_pages(
    pdf,
    summary_rows: Sequence[Mapping[str, object]],
    title: str,
    rows_per_page: int = 24,
) -> None:
    """Write paginated metric summary table pages to a PDF.

    Parameters
    ----------
    pdf : matplotlib.backends.backend_pdf.PdfPages
        Open PDF writer.
    summary_rows : Sequence[Mapping[str, object]]
        Metric summary rows.
    title : str
        Table page title.
    rows_per_page : int, optional
        Number of table rows per page.

    Returns
    -------
    None
    """
    columns = ["realisation", "variable"]
    if any("value" in row for row in summary_rows):
        columns.append("value")
    if any("n" in row for row in summary_rows):
        columns.extend(["n", "min", "max", "mean", "median"])
    best_cells = _get_metric_summary_best_cells(summary_rows, columns)
    _log_metric_summary_table(summary_rows, columns)
    row_starts = range(0, len(summary_rows), rows_per_page)
    for page_index, start_row in enumerate(row_starts):
        page_rows = summary_rows[start_row : start_row + rows_per_page]
        table_data = [
            [_format_metric_summary_value(row.get(column)) for column in columns]
            for row in page_rows
        ]
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        page_title = title
        if len(summary_rows) > rows_per_page:
            page_title = f"{title} ({page_index + 1})"
        ax.set_title(page_title, fontsize=14, pad=16)
        table = ax.table(
            cellText=table_data,
            colLabels=columns,
            loc="center",
            cellLoc="left",
            colLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.3)
        _style_metric_summary_table(
            table=table,
            summary_rows=summary_rows,
            columns=columns,
            start_row=start_row,
            row_count=len(page_rows),
        )
        _highlight_metric_summary_table_cells(
            table=table,
            best_cells=best_cells,
            columns=columns,
            start_row=start_row,
            row_count=len(page_rows),
        )
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def _log_metric_summary_table(summary_rows, columns):
    """Log the metric summary table as formatted text.

    Parameters
    ----------
    summary_rows : Sequence[Mapping[str, object]]
        Metric summary rows.
    columns : Sequence[str]
        Rendered table columns.

    Returns
    -------
    None
    """
    table_text = _create_metric_summary_table_text(summary_rows, columns)
    if table_text:
        logger.info(f"Metric summary table:\n{table_text}")


def _create_metric_summary_table_text(summary_rows, columns):
    """Create a formatted text table for metric summaries.

    Parameters
    ----------
    summary_rows : Sequence[Mapping[str, object]]
        Metric summary rows.
    columns : Sequence[str]
        Rendered table columns.

    Returns
    -------
    str
        Formatted table text.
    """
    table_rows = [
        [_format_metric_summary_value(row.get(column)) for column in columns]
        for row in summary_rows
    ]
    if not table_rows:
        return ""
    widths = [
        max(len(str(column)), *(len(row[column_index]) for row in table_rows))
        for column_index, column in enumerate(columns)
    ]
    header = "  ".join(
        str(column).ljust(width) for column, width in zip(columns, widths)
    )
    separator = "  ".join("-" * width for width in widths)
    lines = [header, separator]
    previous_variable = None
    for row_index, row_values in enumerate(table_rows):
        variable = summary_rows[row_index].get("variable")
        if previous_variable is not None and variable != previous_variable:
            lines.append("")
        lines.append(
            "  ".join(value.ljust(width) for value, width in zip(row_values, widths))
        )
        previous_variable = variable
    return "\n".join(lines)


def _style_metric_summary_table(table, summary_rows, columns, start_row, row_count):
    """Style summary table header and variable separators.

    Parameters
    ----------
    table : matplotlib.table.Table
        Rendered table.
    summary_rows : Sequence[Mapping[str, object]]
        All metric summary rows.
    columns : Sequence[str]
        Rendered table columns.
    start_row : int
        Global row index of the first page row.
    row_count : int
        Number of data rows on the page.

    Returns
    -------
    None
    """
    for column_index, _ in enumerate(columns):
        header_cell = table[(0, column_index)]
        header_cell.set_text_props(weight="bold")
        header_cell.set_linewidth(1.2)
        header_cell.set_facecolor("#eeeeee")

    for page_row in range(row_count):
        global_row = start_row + page_row
        if global_row == 0:
            continue
        variable = summary_rows[global_row].get("variable")
        previous_variable = summary_rows[global_row - 1].get("variable")
        if variable == previous_variable:
            continue
        for column_index, _ in enumerate(columns):
            cell = table[(page_row + 1, column_index)]
            cell.set_linewidth(1.4)
            cell.set_facecolor("#f7f7f7")


def _get_metric_summary_best_cells(summary_rows, columns):
    """Get summary table cells containing the best metric values.

    Parameters
    ----------
    summary_rows : Sequence[Mapping[str, object]]
        Metric summary rows.
    columns : Sequence[str]
        Rendered table columns.

    Returns
    -------
    set[tuple[int, str]]
        Row index and column name pairs to highlight.
    """
    best_cells = set()
    variables = sorted({row.get("variable") for row in summary_rows}, key=str)
    for variable in variables:
        preference = _get_metric_summary_preference(variable)
        if preference is None:
            continue
        variable_rows = [
            (row_index, row)
            for row_index, row in enumerate(summary_rows)
            if row.get("variable") == variable
        ]
        for column in columns:
            if column not in METRIC_SUMMARY_VALUE_COLUMNS:
                continue
            values = []
            for row_index, row in variable_rows:
                value = row.get(column)
                if isinstance(
                    value, (int, float, np.integer, np.floating)
                ) and np.isfinite(value):
                    values.append((row_index, float(value)))
            if not values:
                continue
            for row_index in _get_best_metric_value_row_indices(values, preference):
                best_cells.add((row_index, column))
    return best_cells


def _get_metric_summary_preference(metric_name):
    """Get how a metric should be optimized in summary tables.

    Parameters
    ----------
    metric_name : str
        Metric name from the summary table.

    Returns
    -------
    tuple[str, float or None] or None
        Preference mode and optional target value.
    """
    normalized_name = _normalize_metric_summary_name(metric_name)
    if normalized_name in ONE_CENTERED_METRICS:
        return "center", 1.0
    if normalized_name in ZERO_CENTERED_METRICS:
        return "center", 0.0
    if normalized_name in HIGHER_IS_BETTER_METRICS:
        return "max", None
    if normalized_name in LOWER_IS_BETTER_METRICS:
        return "min", None
    return None


def _normalize_metric_summary_name(metric_name):
    """Normalize metric names for summary table preference lookup.

    Parameters
    ----------
    metric_name : str
        Raw metric name.

    Returns
    -------
    str
        Normalized metric name.
    """
    normalized_name = str(metric_name).strip().lower().replace("_", "-")
    if normalized_name.startswith("avg-"):
        normalized_name = normalized_name[4:]
    return normalized_name


def _get_best_metric_value_row_indices(values, preference):
    """Get row indices for the best metric value entries.

    Parameters
    ----------
    values : Sequence[tuple[int, float]]
        Row indices and numeric metric values.
    preference : tuple[str, float or None]
        Preference mode and optional target value.

    Returns
    -------
    list[int]
        Row indices with the best value.
    """
    mode, target = preference
    value_array = np.asarray([value for _, value in values], dtype=float)
    if mode == "center":
        scores = np.abs(value_array - target)
        best_score = float(np.min(scores))
        return [
            row_index
            for (row_index, _), score in zip(values, scores)
            if np.isclose(score, best_score)
        ]
    if mode == "max":
        best_value = float(np.max(value_array))
        return [
            row_index for row_index, value in values if np.isclose(value, best_value)
        ]
    if mode == "min":
        best_value = float(np.min(value_array))
        return [
            row_index for row_index, value in values if np.isclose(value, best_value)
        ]
    return []


def _highlight_metric_summary_table_cells(
    table,
    best_cells,
    columns,
    start_row,
    row_count,
):
    """Highlight best metric values in a rendered summary table page.

    Parameters
    ----------
    table : matplotlib.table.Table
        Rendered table.
    best_cells : set[tuple[int, str]]
        Global row and column pairs to highlight.
    columns : Sequence[str]
        Rendered table columns.
    start_row : int
        Global row index of the first page row.
    row_count : int
        Number of data rows on the page.

    Returns
    -------
    None
    """
    for page_row in range(row_count):
        global_row = start_row + page_row
        for column_index, column in enumerate(columns):
            if (global_row, column) not in best_cells:
                continue
            cell = table[(page_row + 1, column_index)]
            cell.set_facecolor("#d9ead3")
            cell.set_text_props(weight="bold")


def _format_metric_summary_value(value):
    """Format one metric summary table value.

    Parameters
    ----------
    value : object
        Value to format.

    Returns
    -------
    str
        Formatted table value.
    """
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def plot_constant_data_map(
    lon,
    lat,
    arr,
    vmin,
    vmax,
    cb_label,
    title,
    out_path,
    cmap="RdBu",
):
    """Plot a map for constant-valued data.

    Creates a uniform-colored map with a legend patch instead of a colorbar.
    """
    _require_cartopy()

    base_cmap = plt.get_cmap(cmap)
    single_color = base_cmap(0.5)  # middle color

    cmap = ListedColormap([single_color])
    norm = BoundaryNorm([vmin - 1, vmax + 1], ncolors=1)

    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())

    lon2d, lat2d = np.meshgrid(lon, lat)
    ax.pcolormesh(
        lon2d,
        lat2d,
        np.full_like(arr, fill_value=vmin),
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        shading="auto",
    )

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.gridlines(draw_labels=True, linewidth=0.2, linestyle="--")

    # Remove colorbar, add legend box with patch
    patch = mpatches.Patch(color=single_color, label=f"{vmin:.2f} {cb_label}")
    ax.legend(handles=[patch], loc="lower right", framealpha=0.8, fontsize=12)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_discrete_data_map(
    lon,
    lat,
    arr,
    vmin,
    vmax,
    cb_label,
    title,
    out_path,
    cmap="RdBu",
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
):
    """Plot a map with discrete bins using a colorbar.

    Uses Cartopy for geographic projection and Matplotlib for color mapping.
    """
    _require_cartopy()

    # Determine how the colorbar should handle data outside [vmin, vmax]
    extend = "neither"
    if np.nanmin(arr) < vmin and np.nanmax(arr) > vmax:
        extend = "both"
    elif np.nanmin(arr) < vmin:
        extend = "min"
    elif np.nanmax(arr) > vmax:
        extend = "max"

    # Create a discrete colormap with ~9 bins and possible extensions
    levels = np.linspace(vmin, vmax, 10)
    n_bins = len(levels) - 1

    # Account for extra colors needed if data exceeds bounds
    extra = {"neither": 0, "min": 1, "max": 1, "both": 2}[extend]

    base_cmap = plt.get_cmap(cmap, n_bins + extra)
    cmap = ListedColormap(base_cmap(np.arange(n_bins + extra)))
    norm = BoundaryNorm(levels, ncolors=n_bins + extra, extend=extend)

    # Set up the plot using Cartopy's PlateCarree projection
    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Automatically set extent to data bounds unless overridden
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())

    # Create 2D grids of lon/lat for plotting
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Plot the data field using pcolormesh with georeferenced lon/lat
    mesh = ax.pcolormesh(
        lon2d,
        lat2d,
        arr,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        shading="auto",
    )

    # Optionally override axis limits
    if x_min is not None or x_max is not None:
        ax.set_xlim(left=x_min, right=x_max)
    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)

    # Add cartographic features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.gridlines(draw_labels=True, linewidth=0.2, linestyle="--")

    # Create colorbar with optional extensions to show clipping
    cb = plt.colorbar(
        mesh, ax=ax, orientation="vertical", shrink=0.8, pad=0.05, extend=extend
    )
    cb.set_label(cb_label)

    # Final plot layout
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_map(
    data: xr.DataArray,
    cb_label: str,
    title: str,
    out_path: Path,
    cmap: str = "RdBu",
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    Plot and save a 2D DataArray over longitude and latitude using Cartopy.

    This function creates a geographically-aware color plot using a discrete colormap,
    with automatic colorbar extensions when values fall outside the given [vmin, vmax].

    Parameters
    ----------
    data : xr.DataArray
        2D input data with 'lon' and 'lat' coordinates.
    cb_label : str
        Label for the colorbar.
    title : str
        Title of the plot.
    out_path : Path
        Path where the figure will be saved.
    cmap : str, optional
        Name of the Matplotlib colormap to use.
    x_min, x_max, y_min, y_max : float, optional
        Manual spatial limits for zooming.
    vmin, vmax : float, optional
        Color scale limits. If not provided, data min/max are used.
    """
    _require_cartopy()

    # Extract longitude, latitude, and data values
    lon = data["lon"].values
    lat = data["lat"].values
    arr = np.squeeze(data.values)  # remove singleton dimension (e.g., time)

    # Create 2D grids of lon/lat for plotting
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Determine color limits if not provided
    if vmin is None:
        vmin = float(np.nanmin(arr))
    if vmax is None:
        vmax = float(np.nanmax(arr))

    # Fix for constant data (e.g., all zeros) so vmin != vmax
    if vmin == vmax:
        # Constant data plotting
        plot_constant_data_map(
            lon=lon,
            lat=lat,
            arr=arr,
            vmin=vmin,
            vmax=vmax,
            cb_label=cb_label,
            title=title,
            out_path=out_path,
            cmap=cmap,
        )
    else:
        # plot regular discrete map
        plot_discrete_data_map(
            lon=lon,
            lat=lat,
            arr=arr,
            vmin=vmin,
            vmax=vmax,
            cb_label=cb_label,
            title=title,
            out_path=out_path,
            cmap=cmap,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )
