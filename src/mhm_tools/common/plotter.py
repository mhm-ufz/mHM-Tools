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
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgba
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mhm_tools.common.constants import KGE_CONSTANT_MEAN_BOUND, NSE_CONSTANT_MEAN_BOUND
from mhm_tools.common.logger import log_errors

logger = logging.getLogger(__name__)

# Dataset roles, kept the same in every figure
INPUT_COLOR = "#79A3E6"
REF_COLOR = "#008176"
RATIO_COLOR = "#0000A7"
# colour of skill values below the lower colour limit, e.g. KGE < -0.41, NSE < 0
KGE_UNDER_COLOR = "lightgray"
FIGURE_WIDTH = 10.5
PLOT_DPI = 400
# name and unit shown for metrics whose column name alone is unclear; the
# discharge diff is the volume of sim - obs over every compared time step
METRIC_LABELS = {
    "diff": ("volume difference sim - obs", "m³"),
    "rel_diff": ("relative volume difference", "-"),
}
# colour bin edges of percent difference maps, by the largest absolute value
# they cover, each with a neutral bin around 0; never wider than +-100 %
PERCENT_DIFF_BOUNDS = {
    25: (-25, -20, -15, -10, -5, -2.5, 2.5, 5, 10, 15, 20, 25),
    50: (-50, -40, -30, -20, -10, -5, 5, 10, 20, 30, 40, 50),
    75: (-75, -60, -45, -30, -15, -5, 5, 15, 30, 45, 60, 75),
    100: (-100, -75, -50, -25, -10, 10, 25, 50, 75, 100),
}
# colour of the bin around the center of every diverging map, the grey in the
# middle of coolwarm
NEUTRAL_COLOR = "#dcdddd"
# metrics stored as fractions but mapped in percent
PERCENT_METRICS = {"rel_diff"}
# recessive ink for axes, labels and captions
AXIS_COLOR = "#9aa5ab"
CAPTION_COLOR = "#5b6770"
GRID_COLOR = "#e4e8ea"

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
CORRELATION_METRICS = {
    "gamma",
    "pearson",
    "r",
    "rho",
    "rs",
    "spearman",
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


def style_axes(ax, show_grid=False):
    """Apply the recessive axis styling shared by every figure.

    Args:
        ax: Matplotlib axes to style.
        show_grid: Draw a light horizontal grid behind the data.

    Returns
    -------
        The styled axes.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS_COLOR)
        ax.spines[spine].set_linewidth(0.5)
    ax.tick_params(colors=CAPTION_COLOR, labelcolor=CAPTION_COLOR, length=3, width=0.5)
    if show_grid:
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
    return ax


def style_map_axes(ax):
    """Remove the ticks of a map axes and draw thin spines.

    Args:
        ax: Matplotlib or cartopy axes showing a map.

    Returns
    -------
        The styled axes.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.25)
    return ax


def add_map_colorbar(fig, ax, mappable, bounds, extend, ticks, label):
    """Attach a discrete colorbar to the right of a map axes.

    Args:
        fig: Figure holding the map.
        ax: Map axes the colorbar is attached to.
        mappable: Image or scatter returned by the plot call.
        bounds: Bin edges of the colour norm.
        extend: Colorbar extension, one of "neither", "min", "max" or "both".
        ticks: Tick positions on the colorbar.
        label: Colorbar label including units.

    Returns
    -------
        The created colorbar.
    """
    # axes_class keeps the colorbar a plain axes next to cartopy GeoAxes
    colorbar_ax = make_axes_locatable(ax).append_axes(
        "right", size="5%", pad=0.1, axes_class=plt.Axes
    )
    colorbar = fig.colorbar(
        mappable,
        cax=colorbar_ax,
        boundaries=bounds,
        extend=extend,
        ticks=ticks,
        label=label,
    )
    # shortest form of every tick, so -0.41 and -0.2 instead of -0.41 and -0.20
    colorbar.set_ticks(ticks, labels=[f"{tick + 0.0:g}" for tick in ticks])
    return colorbar


def create_comparison_title(input_name, ref_name, years=None):
    """Create the figure title of an input/reference comparison.

    Args:
        input_name: Label of the input dataset.
        ref_name: Label of the reference dataset.
        years: Sequence of the covered years, None to leave the period out.

    Returns
    -------
        Title string.
    """
    title = f"Comparison {input_name} with {ref_name}"
    if years is not None and len(years) > 0:
        first_year, last_year = years[0], years[-1]
        if first_year == last_year:
            title += f" for year {first_year}"
        else:
            title += f" for years {first_year}-{last_year}"
    return title


def create_axis_label(name, units=None):
    """Create an axis or colorbar label with units in square brackets.

    A metric in `METRIC_LABELS` gets its name and unit from there when no unit
    is given.

    Args:
        name: Quantity shown on the axis.
        units: Units of the quantity, None or "1" for dimensionless values.

    Returns
    -------
        Label of the form ``"name [units]"``.
    """
    if units is None and name in METRIC_LABELS:
        name, units = METRIC_LABELS[name]
    if units in (None, "", "1", "-"):
        units = "-"
    return f"{name} [{units}]"


def calculate_map_figure_size(
    extent, panel_count=1, extra_width=1.0, extra_height=1.2, max_panel_height=7.0
):
    """Size a map figure of standard width from the aspect of the mapped domain.

    Args:
        extent: Map extent as (lon_min, lon_max, lat_min, lat_max).
        panel_count: Number of map panels side by side.
        extra_width: Inches reserved for colorbars and padding.
        extra_height: Inches reserved for titles, captions and legends.
        max_panel_height: Largest height of one panel in inches.

    Returns
    -------
        The (width, height) figure size in inches.
    """
    lon_min, lon_max, lat_min, lat_max = extent
    lat_span = lat_max - lat_min
    aspect = (lon_max - lon_min) / lat_span if lat_span > 0 else 1.0
    panel_width = (FIGURE_WIDTH - extra_width) / panel_count
    # keep a very thin domain from collapsing into a line
    panel_height = min(max_panel_height, max(panel_width / aspect, 1.5))
    return FIGURE_WIDTH, panel_height + extra_height


def get_metric_plot_style(metric_name, kge_vmin=KGE_CONSTANT_MEAN_BOUND):
    """Get the colour settings of a metric map.

    Skill and correlation metrics get a sequential map, ratios and differences
    a diverging map centred on their no-difference value.

    Args:
        metric_name: Metric name, e.g. ``"kge"``, ``"alpha"`` or ``"diff"``.
        kge_vmin: Lower colour limit of KGE maps.

    Returns
    -------
        Dict with ``cmap``, ``center``, ``vmin``, ``vmax``, ``under_color`` and
        ``max_extended_vmin`` for ``create_discrete_colour_norm`` with
        ``bounds_type="data"``. None limits are derived from the data.
    """
    metric = metric_name.lower()
    style = {
        "cmap": "viridis_r",
        "center": None,
        "vmin": None,
        "vmax": None,
        "under_color": None,
        "max_extended_vmin": None,
    }
    if metric == "kge":
        style.update(vmin=kge_vmin, vmax=1.0, under_color=KGE_UNDER_COLOR)
    elif metric == "nse":
        style.update(
            vmin=NSE_CONSTANT_MEAN_BOUND, vmax=1.0, under_color=KGE_UNDER_COLOR
        )
    elif metric in CORRELATION_METRICS:
        # positive correlations stay on the scale, only negative ones extend
        style.update(vmax=1.0, max_extended_vmin=0.0)
    elif metric in ONE_CENTERED_METRICS:
        style.update(cmap="coolwarm_r", center=1.0)
    elif (
        metric in ZERO_CENTERED_METRICS
        or metric in PERCENT_METRICS
        or metric.startswith("diff")
    ):
        style.update(cmap="coolwarm_r", center=0.0)
    elif metric == "rmse":
        style.update(cmap="magma_r", vmin=0.0)
    return style


def round_sensibly(value):
    """Round map half-range to sensible steps and return decimals for labels.

    Returns
    -------
    tuple[float, int]
        (rounded_value, round_dec)
    """
    value = float(abs(value))
    if not np.isfinite(value) or value == 0:
        return 1e-6, 6

    thresholds = [
        (1.4, 2, 2),  # step 0.5
        (0.15, 5, 2),  # step 0.2
        (0.015, 50, 3),  # step 0.02
        (0.0015, 500, 4),  # step 0.002
        (0.00015, 5000, 5),  # step 0.0002
        (0.0, 50000, 6),  # step 0.00002
    ]
    for threshold, scale, round_dec in thresholds:
        if value > threshold:
            rounded = round(value * scale) / scale
            rounded = max(rounded, 1 / scale)
            return rounded, round_dec
    return value, 6


def calculate_nice_step(raw_step, factors=(1.0, 2.0, 2.5, 5.0, 10.0)):
    """Calculate the nice step closest to a raw colour bin width.

    Args:
        raw_step: Raw bin width, e.g. the colour range divided by 9.
        factors: Allowed multiples of a power of ten, ascending up to 10.

    Returns
    -------
        One of `factors` times a power of ten, 1.0 for an unusable width.
    """
    if not np.isfinite(raw_step) or raw_step <= 0:
        return 1.0
    magnitude = 10 ** np.floor(np.log10(raw_step))
    candidates = magnitude * np.array(factors)
    return float(candidates[np.argmin(np.abs(np.log(candidates / raw_step)))])


def calculate_colour_limits(
    values, center=None, vmin=None, vmax=None, max_extended_vmin=None
):
    """Calculate colour limits that are not stretched by outliers.

    Each end uses the data min/max, or the 1st/99th percentile when the
    outliers beyond it span more than the bulk of the data on that side. With
    a center the limits are symmetric around it and left unrounded, since
    ``create_discrete_colour_norm`` widens them to the edge of a bin centred on
    the center. Sequential limits are rounded outwards to multiples of the bin
    step, so rounding never pushes values off the scale.

    Args:
        values: Array of the plotted values.
        center: No-difference value of a diverging map, None for sequential maps.
        vmin: Lower limit to keep instead of deriving it.
        vmax: Upper limit to keep instead of deriving it.
        max_extended_vmin: Highest derived lower limit once values fall below
            it, e.g. 0 for correlations; None for no such cap.

    Returns
    -------
        Tuple of (vmin, vmax).
    """
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return (0.0 if vmin is None else vmin, 1.0 if vmax is None else vmax)
    data_min, data_max = finite_values.min(), finite_values.max()
    low, high = np.quantile(finite_values, [0.01, 0.99])
    mean = finite_values.mean()
    lower = low if abs(low - data_min) > abs(mean - low) else data_min
    upper = high if abs(data_max - high) > abs(high - mean) else data_max
    if center is not None:
        half_range = max(abs(upper - center), abs(center - lower))
        lower, upper = center - half_range, center + half_range
    else:
        # the step follows the final range, so a given limit counts as well
        lower = lower if vmin is None else vmin
        upper = upper if vmax is None else vmax
        if upper > lower:
            # the small shift keeps floating point noise from adding a step
            step = calculate_nice_step((upper - lower) / 9)
            if vmin is None:
                lower = round(float(np.floor(lower / step + 1e-9) * step), 10)
            if vmax is None:
                upper = round(float(np.ceil(upper / step - 1e-9) * step), 10)
    if vmin is None and max_extended_vmin is not None and data_min < lower:
        lower = min(lower, max_extended_vmin)
    lower = lower if vmin is None else vmin
    upper = upper if vmax is None else vmax
    return float(lower), float(upper)


def get_percent_diff_bounds(values, largest=None):
    """Return the narrowest percent difference bin edges that hold the values.

    Outliers are ignored the way `calculate_colour_limits` ignores them on
    every other map, so a few outlying cells get a colorbar arrow instead of
    widening the bins.

    Parameters
    ----------
    values : array_like
        Differences in percent.
    largest : float, optional
        Largest absolute difference to cover instead of the one derived from
        `values`, so several maps sharing one colorbar get the same bins.

    Returns
    -------
    ndarray
        Bin edges from `PERCENT_DIFF_BOUNDS`: +-25 %, +-50 %, +-75 % or +-100 %
        when the differences are larger, never wider.
    """
    widest = max(PERCENT_DIFF_BOUNDS)
    if largest is None:
        largest = calculate_colour_limits(values, center=0.0)[1]
    limit = next(
        (limit for limit in sorted(PERCENT_DIFF_BOUNDS) if abs(largest) <= limit),
        widest,
    )
    return np.array(PERCENT_DIFF_BOUNDS[limit], dtype=float)


def create_bin_colormap(cmap, bin_count, extent, under_color=None, centred=False):
    """Create the colormap of a discrete colour norm with one colour per bin.

    An extension gets the colormap colour past the last bin, so values beyond
    a limit never share the colour of the bin next to it.

    Args:
        cmap: Colormap or colormap name the bin colours are sampled from.
        bin_count: Number of colour bins.
        extent: Colorbar extension, one of "neither", "min", "max" or "both".
        under_color: Colour of every value below the lowest bin, None for the
            colormap colour past it.
        centred: Whether the bins are centred on a no-difference value, which
            keeps the middle bin neutral and on the colormap midpoint.

    Returns
    -------
        ListedColormap of the bin colours with its under and over colours.
    """
    reserve_low = extent in {"min", "both"} and under_color is None
    reserve_high = extent in {"max", "both"}
    sample_low, sample_high = reserve_low, reserve_high
    if centred and reserve_low != reserve_high:
        # an extension on one side only would shift the bins along the colormap,
        # so both ends are sampled to keep the centre bin on its midpoint
        sample_low = sample_high = True
    colours = plt.get_cmap(cmap)(
        np.linspace(0, 1, bin_count + int(sample_low) + int(sample_high))
    )
    bin_colours = colours[int(sample_low) : len(colours) - int(sample_high)].copy()
    if centred and bin_count % 2 == 1:
        # the bin around the center is always neutral, whatever the colormap
        bin_colours[bin_count // 2] = to_rgba(NEUTRAL_COLOR)
    if under_color is None:
        under_color = colours[0] if reserve_low else bin_colours[0]
    return ListedColormap(bin_colours).with_extremes(
        under=under_color, over=colours[-1] if reserve_high else bin_colours[-1]
    )


def create_discrete_colour_norm(
    values,
    diff_to_mean=None,
    center=1,
    vmin=0,
    vmax=1,
    cmap=plt.cm.coolwarm_r,
    bounds_type="fixed",
    under_color=None,
    max_extended_vmin=None,
):
    """Create a discrete colour norm with about 9 bins for one map.

    Sequential maps: the outer bin edges are the colour limits, the inner edges
    sit on multiples of a nice step (1, 2, 2.5 or 5 times a power of ten), e.g.
    -0.41, -0.2, 0, 0.2, ..., 1 for a KGE.
    Diverging maps ("fixed", "max", or "data" with a center): an odd number of
    bins with one centred on the center, edges at center +- 0.5, 1.5, ... steps
    of 1, 2 or 5 times a power of ten, and the limits widened to the outer edges.

    Behaviour by `bounds_type`:
    - "max": set vmin=center - diff_to_mean and vmax=center + diff_to_mean
    - "quantiles": set vmin/vmax to the 5th/95th percentiles of `values`
    - "fixed": set vmin=center - 0.5 and vmax=center + 0.5
    - "data": derive vmin/vmax with `calculate_colour_limits`; a vmin or vmax
      that is not None is kept, a center that is not None makes them symmetric
      and `max_extended_vmin` caps a lower limit that values fall below
    - "given": use vmin/vmax as they are
    - "percent": use the narrowest `PERCENT_DIFF_BOUNDS` (+-25 %, +-50 %,
      +-75 % or +-100 %) that holds the values without their outliers, see
      `get_percent_diff_bounds`, or `diff_to_mean` when given, ignoring center,
      vmin and vmax; values beyond the bins get an arrow

    Returns
    -------
    cmap : Colormap
        One colour per bin. Values beyond a limit with a colorbar extension
        get a colour of their own: `under_color` below the lowest bin if
        given, otherwise the colormap colour past the last bin.
    norm : BoundaryNorm
        Norm mapping values to the bins.
    bounds : ndarray
        Bin edges used by BoundaryNorm.
    extent : {"neither", "min", "max", "both"}
        Whether data extend beyond bounds.
    ticks : ndarray
        Tick positions at the bin edges (every second edge when there are more
        than 10 sequential or 11 diverging bins).
    """

    def _step_decimals(step):
        step_abs = abs(float(step))
        if step_abs == 0:
            return 0
        decimals = int(max(0, -np.floor(np.log10(step_abs))))
        if not np.isclose(step_abs * (10**decimals), round(step_abs * (10**decimals))):
            decimals += 1
        return min(decimals, 6)

    if bounds_type == "max" and diff_to_mean is not None:
        vmin = center - diff_to_mean
        vmax = center + diff_to_mean
    if bounds_type == "quantiles":
        vmin, vmax = (
            np.nanquantile(values, 0.05),
            np.nanquantile(values, 0.95),
        )
        if abs(vmax - vmin) < abs(vmax / 3) or vmin == vmax:
            vmin, vmax = (float(np.nanmin(values)), float(np.nanmax(values)))
        if abs(vmax - vmin) < abs(vmax / 3) or vmin == vmax:
            vmin, vmax = (
                vmin - abs(vmin / 3),
                vmax + abs(vmax / 3),
            )
    if bounds_type == "fixed":
        vmin, vmax = center - 0.5, center + 0.5
    if bounds_type == "data":
        vmin, vmax = calculate_colour_limits(
            values, center, vmin, vmax, max_extended_vmin
        )

    values_np = np.asarray(values)
    if vmin is None or vmax is None or not np.isfinite(vmin) or not np.isfinite(vmax):
        finite_values = values_np[np.isfinite(values_np)]
        if finite_values.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = (
                float(np.nanmin(finite_values)),
                float(np.nanmax(finite_values)),
            )
    if np.isclose(vmax, vmin):
        delta = max(abs(vmax), 1.0) * 0.5
        vmin, vmax = vmin - delta, vmax + delta

    diverging = bounds_type in {"fixed", "max"} or (
        bounds_type == "data" and center is not None
    )
    if bounds_type == "percent":
        bounds = get_percent_diff_bounds(values_np, largest=diff_to_mean)
        ticks = bounds
    elif diverging:
        # one bin is centred on the center value, so values close to it stand
        # out; steps of 1, 2 or 5 keep the edges at center +- half a step readable
        half_range = max(vmax - center, center - vmin)
        step = calculate_nice_step(2 * half_range / 9, factors=(1.0, 2.0, 5.0, 10.0))
        side_bins = max(1, int(np.ceil(half_range / step - 0.5 - 1e-9)))
        edge_offsets = np.arange(-side_bins, side_bins + 2) - 0.5
        bounds = np.round(center + step * edge_offsets, _step_decimals(step / 2))
        ticks = bounds
        if bounds.size > 12:
            # every second edge outwards from the centre bin, plus both limits
            keep_tick = np.round(np.abs(edge_offsets) - 0.5) % 2 == 0
            keep_tick[[0, -1]] = True
            ticks = bounds[keep_tick]
    else:
        step = calculate_nice_step((vmax - vmin) / 9)
        inner_edges = step * np.arange(np.ceil(vmin / step), np.floor(vmax / step) + 1)
        # an inner edge within half a step of a limit would leave a sliver bin
        min_distance = step / 2 * (1 - 1e-9)
        inner_edges = inner_edges[
            (inner_edges - vmin >= min_distance) & (vmax - inner_edges >= min_distance)
        ]
        inner_edges = np.round(inner_edges, _step_decimals(step))
        bounds = np.round(np.concatenate(([vmin], inner_edges, [vmax])), 6)
        ticks = bounds
        if bounds.size > 11:
            # every second inner edge, plus both limits
            even_edges = inner_edges[np.round(inner_edges / step) % 2 == 0]
            ticks = np.round(np.concatenate(([vmin], even_edges, [vmax])), 6)

    extent = "neither"
    if bounds_type in {"data", "given", "percent"}:
        # every value outside the bounds gets an arrow on the colorbar
        finite_values = values_np[np.isfinite(values_np)]
        below = finite_values.size > 0 and finite_values.min() < bounds[0]
        above = finite_values.size > 0 and finite_values.max() > bounds[-1]
        extent = {
            (False, False): "neither",
            (True, False): "min",
            (False, True): "max",
            (True, True): "both",
        }[(bool(below), bool(above))]
    else:
        if np.nanquantile(values_np, 0.96) > bounds[-1]:
            extent = "max"
        if np.nanquantile(values_np, 0.049) < bounds[0]:
            extent = "min" if extent == "neither" else "both"

    cmap = create_bin_colormap(
        cmap,
        bounds.size - 1,
        extent,
        under_color=under_color,
        centred=diverging or bounds_type == "percent",
    )
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm, bounds, extent, ticks


def plot_single_map(
    ax,
    values,
    diff_to_mean=None,
    center=1,
    vmin=0,
    vmax=1,
    cmap=plt.cm.coolwarm_r,
    # cmap = plt.cm.RdBu,
    bounds_type="fixed",
    under_color=None,
    **imshow_kwargs,
):
    """Plot a single map on a Matplotlib Axes.

    Bounds and colormap come from `create_discrete_colour_norm`, see there for
    the behaviour of `bounds_type`. `imshow_kwargs` (e.g. extent, origin,
    transform) are passed on to `ax.imshow`.

    Returns
    -------
    im : AxesImage
        The image artist.
    bounds : ndarray
        Bin edges used by BoundaryNorm.
    extent : {"neither", "min", "max", "both"}
        Whether data extend beyond bounds.
    ticks : ndarray
        Tick centers for colorbar labels (every second bin center).
    """
    cmap, norm, bounds, extent, ticks = create_discrete_colour_norm(
        values,
        diff_to_mean=diff_to_mean,
        center=center,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        bounds_type=bounds_type,
        under_color=under_color,
    )
    im = ax.imshow(np.asarray(values), cmap=cmap, norm=norm, **imshow_kwargs)
    return im, bounds, extent, ticks


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
    draw_line: bool = True,
    draw_points: bool = True,
    point_marker: str = "o",
    point_marker_size: int = 4,
    point_opacity: float = 0.7,
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
    draw_line : bool, optional
        Draw the CDF line when true.
    draw_points : bool, optional
        Draw CDF points when true.
    point_marker : str, optional
        Matplotlib marker style for CDF points.
    point_marker_size : int, optional
        Scatter marker size.
    point_opacity : float, optional
        Opacity for CDF points.

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
            s=point_marker_size,
            color=color,
            label=label,
            alpha=point_opacity,
            marker=point_marker,
        )
    return sorted_values, cdf_values


def _get_lower_axis_limit(values, floor=-1.0):
    """Get a lower axis limit that cuts off extreme low outliers.

    Parameters
    ----------
    values : Sequence[float]
        Numeric values the axis is drawn from.
    floor : float, optional
        Lower limit used once the data reaches at or below it.

    Returns
    -------
    float
        `floor` if the data's minimum is at or below it; otherwise a small
        padding below the data's own minimum, so metrics that never approach
        `floor` (e.g. ratios near 1) keep a tightly-fit axis instead of being
        stretched down to `floor`.
    """
    min_value = float(np.nanmin(values))
    if not np.isfinite(min_value) or min_value <= floor:
        return floor
    padding = max(abs(min_value) * 0.05, 0.05)
    return min_value - padding


def _get_metric_floor(variable_name):
    """Get the lowest axis limit drawn for a metric.

    Args:
        variable_name: Metric name.

    Returns
    -------
        The constant-mean bound for KGE and NSE, otherwise -1.
    """
    metric = str(variable_name).lower()
    if metric == "kge":
        return KGE_CONSTANT_MEAN_BOUND
    if metric == "nse":
        return NSE_CONSTANT_MEAN_BOUND
    return -1.0


@log_errors(raise_exceptions=True)
def plot_metric_cdf_comparison(
    values_by_label: Mapping[str, Sequence[float]],
    variable_name: str,
    output_file: Path,
    title: Optional[str] = None,
    x_limits: Optional[Sequence[float]] = None,
    dpi: int = PLOT_DPI,
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
        Figure title. Defaults to a CDF title for the variable.
    x_limits : Sequence[float], optional
        Lower and upper x-axis limits. Defaults to the data range, cut off at
        -1 (at the constant-mean bound for KGE and NSE).
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
    # landscape aspect, so the plot fills a page of the overview PDF
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 6.5))
    plotted_any = False
    series_count = len(values_by_label)
    tab20_colors = plt.get_cmap("tab20").colors
    continuous_cmap = plt.get_cmap("nipy_spectral")
    series_to_draw = []
    plotted_labels = []
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
        elif series_count == 1:
            color = INPUT_COLOR
        elif series_count <= len(tab20_colors):
            color = tab20_colors[color_index % len(tab20_colors)]
        else:
            color_fraction = 0.05 + (0.9 * color_index / max(series_count - 1, 1))
            color = continuous_cmap(color_fraction)
        linestyle = "-"
        if linestyles is not None and label in linestyles:
            linestyle = linestyles[label]
        # the points stay out of the legend, the solid line represents the series
        sorted_values, cdf_values = plot_cdf_values(
            ax,
            values_array,
            label=None,
            color=color,
            linestyle=linestyle,
            draw_line=False,
            draw_points=True,
            point_marker="+",
            point_marker_size=4,
            point_opacity=0.3,
        )
        series_to_draw.append(
            (
                sorted_values,
                cdf_values,
                color,
                linestyle,
                median_value,
                label_with_count,
            )
        )
        plotted_labels.append(label)
        plotted_any = True
    if not plotted_any:
        plt.close(fig)
        msg = f"No finite values available for {variable_name}."
        raise ValueError(msg)

    # Draw every line after every point so no series' line is obscured by
    # another series' points when many CDFs overlap (e.g. per-continent plots).
    for (
        sorted_values,
        cdf_values,
        color,
        linestyle,
        median_value,
        label_with_count,
    ) in series_to_draw:
        ax.plot(
            sorted_values,
            cdf_values,
            color=color,
            linestyle=linestyle,
            linewidth=1.0,
            label=label_with_count,
        )
        if show_median_line:
            ax.axvline(
                median_value,
                color=color,
                linestyle="dashed",
                linewidth=1,
            )

    all_values = np.concatenate([sorted_values for sorted_values, *_ in series_to_draw])
    fig.suptitle(
        title or f"CDF of {variable_name}", fontweight="normal", fontsize="x-large"
    )
    # one series gets its summary in the panel title, several keep it in the legend
    if len(plotted_labels) == 1:
        summary = create_summary_text(
            all_values, units=METRIC_LABELS.get(variable_name, (None, None))[1]
        )
        ax.set_title(f"{plotted_labels[0]} ({summary})")
    ax.set_xlabel(create_axis_label(variable_name))
    ax.set_ylabel(create_axis_label("CDF"))
    ax.set_ylim(0.0, 1.01)
    # every CDF crosses this line at its median
    ax.axhline(0.5, color=CAPTION_COLOR, linestyle="--", linewidth=0.8, zorder=1)
    if x_limits is not None:
        ax.set_xlim(x_limits[0], x_limits[1])
    else:
        ax.set_xlim(
            left=_get_lower_axis_limit(
                all_values, floor=_get_metric_floor(variable_name)
            )
        )
    style_axes(ax)
    ax.grid(True, color="black", linestyle=":", linewidth=0.4, alpha=0.3)
    legend = ax.legend()
    # thicker legend lines keep the series colours easy to tell apart
    for handle in legend.legend_handles:
        handle.set_linewidth(2.5)
    fig.tight_layout()
    fig.savefig(output_file, dpi=dpi)
    plt.close(fig)
    logger.info(f"Wrote CDF plot to {output_file}")


@log_errors(raise_exceptions=True)
def plot_metric_violin_comparison(
    values_by_label: Mapping[str, Sequence[float]],
    variable_name: str,
    output_file: Path,
    title: Optional[str] = None,
    y_limits: Optional[Sequence[float]] = None,
    dpi: int = PLOT_DPI,
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
        Figure title. Defaults to a violin title for the variable.
    y_limits : Sequence[float], optional
        Lower and upper y-axis limits. Defaults to the data range, cut off at
        -1 (at the constant-mean bound for KGE and NSE).
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
        # The KDE's bandwidth/shape is computed from every value handed to
        # it, so a handful of values far outside the displayed y_limits can
        # squash the in-range shape flat even though those points are never
        # actually visible - restrict the density estimate to the displayed
        # range (falling back to the full data if that would empty it) so
        # the rendered shape matches what a reader can actually see, the
        # same way an explicit x_limits keeps a CDF plot's visible curve
        # from being distorted by off-screen outliers.
        display_values = values_array
        if y_limits is not None:
            within_range = (values_array >= y_limits[0]) & (values_array <= y_limits[1])
            if within_range.any():
                display_values = values_array[within_range]
        finite_values.append(display_values)
        violin_colors.append(colors.get(label) if colors is not None else None)
    if not finite_values:
        msg = f"No finite values available for {variable_name}."
        raise ValueError(msg)
    if len(finite_values) == 1 and violin_colors[0] is None:
        violin_colors[0] = INPUT_COLOR

    fig, ax = plt.subplots(figsize=(max(FIGURE_WIDTH, len(labels) * 1.2), 5))
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
    ax.set_ylabel(create_axis_label(variable_name))
    fig.suptitle(
        title or f"Distribution of {variable_name}",
        fontweight="normal",
        fontsize="x-large",
    )
    if y_limits is not None:
        ax.set_ylim(y_limits[0], y_limits[1])
    else:
        ax.set_ylim(
            bottom=_get_lower_axis_limit(
                np.concatenate(finite_values), floor=_get_metric_floor(variable_name)
            )
        )
    style_axes(ax)
    ax.grid(axis="y", color="black", linestyle=":", linewidth=0.5, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_file, dpi=dpi)
    plt.close(fig)
    logger.info(f"Wrote violin plot to {output_file}")


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
    table_text = create_table_text(summary_rows, columns)
    if table_text:
        logger.info(f"Metric summary table:\n{table_text}")


def create_table_text(summary_rows, columns):
    """Create an aligned text table, e.g. for log output.

    Floats are shown with 4 significant digits. A blank line separates rows
    whose ``variable`` entry differs.

    Parameters
    ----------
    summary_rows : Sequence[Mapping[str, object]]
        Table rows as mappings from column name to value.
    columns : Sequence[str]
        Rendered table columns.

    Returns
    -------
    str
        Formatted table text, empty when there are no rows.
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


def get_map_extent_and_origin(lon, lat):
    """Return the imshow extent and origin for cell-centre lon/lat coordinates.

    Args:
        lon: 1D array of longitude cell centres.
        lat: 1D array of latitude cell centres.

    Returns
    -------
        Tuple of (extent tuple of the cell edges, origin string).
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon_half = abs(lon[-1] - lon[0]) / (lon.size - 1) / 2 if lon.size > 1 else 0.5
    lat_half = abs(lat[-1] - lat[0]) / (lat.size - 1) / 2 if lat.size > 1 else 0.5
    extent = (
        float(lon.min() - lon_half),
        float(lon.max() + lon_half),
        float(lat.min() - lat_half),
        float(lat.max() + lat_half),
    )
    origin = "lower" if lat[0] <= lat[-1] else "upper"
    return extent, origin


def create_summary_text(values, units=None, bounds=None):
    """Create the median and mean summary shown in panel titles.

    Args:
        values: Array of the plotted values.
        units: Units appended to both numbers, None for dimensionless values.
        bounds: Colour bin edges; their range sets the number of decimals.

    Returns
    -------
        Text like ``"median=0.12mm, mean=0.10mm"``.
    """
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return "no valid values"
    round_dec = 2
    if bounds is not None:
        _, round_dec = round_sensibly((bounds[-1] - bounds[0]) / 2)
    unit_text = "" if units in (None, "", "1", "-") else units
    return (
        f"median={np.median(finite_values):.{round_dec}f}{unit_text}, "
        f"mean={np.mean(finite_values):.{round_dec}f}{unit_text}"
    )


def _create_geo_map_axes(extent):
    """Create a standard width figure with one PlateCarree map axes.

    Args:
        extent: Map extent as (lon_min, lon_max, lat_min, lat_max).

    Returns
    -------
        Tuple of (figure, axes).
    """
    fig, ax = plt.subplots(
        figsize=calculate_map_figure_size(extent),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    return fig, ax


def _write_geo_map(fig, ax, title, panel_title, out_path, x_min, x_max, y_min, y_max):
    """Add features, limits and titles to a map figure, then save and close it.

    Args:
        fig: Figure holding the map.
        ax: Cartopy map axes.
        title: Figure title.
        panel_title: Panel title with the summary statistics.
        out_path: Path the PNG is written to.
        x_min, x_max, y_min, y_max: Optional spatial limits for zooming.
    """
    if x_min is not None or x_max is not None:
        ax.set_xlim(left=x_min, right=x_max)
    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    style_map_axes(ax)
    ax.set_title(panel_title)
    fig.suptitle(title, fontweight="normal", fontsize="x-large")
    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI)
    plt.close(fig)
    logger.info(f"Wrote map to {out_path}")


@log_errors(raise_exceptions=True)
def plot_constant_data_map(
    lon,
    lat,
    arr,
    vmin,
    vmax,  # noqa: ARG001 - kept so both map plotters share one signature
    cb_label,
    title,
    out_path,
    cmap="RdBu",
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    panel_title=None,
):
    """Plot a map for constant-valued data.

    Creates a uniform-colored map with a legend patch instead of a colorbar.
    Cells without data stay unpainted.
    """
    _require_cartopy()

    single_color = plt.get_cmap(cmap)(0.5)  # middle color
    extent, origin = get_map_extent_and_origin(lon, lat)
    fig, ax = _create_geo_map_axes(extent)
    ax.imshow(
        np.where(np.isfinite(arr), vmin, np.nan),
        cmap=ListedColormap([single_color]),
        extent=extent,
        origin=origin,
        transform=ccrs.PlateCarree(),
        interpolation="nearest",
    )
    # Remove colorbar, add legend box with patch
    patch = mpatches.Patch(color=single_color, label=f"{vmin:.2f} {cb_label}")
    ax.legend(handles=[patch], loc="lower right", framealpha=0.8)
    summary = create_summary_text(arr)
    _write_geo_map(
        fig,
        ax,
        title,
        f"{panel_title} ({summary})" if panel_title else summary,
        out_path,
        x_min,
        x_max,
        y_min,
        y_max,
    )


@log_errors(raise_exceptions=True)
def plot_discrete_data_map(  # noqa: PLR0913
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
    under_color=None,
    panel_title=None,
    center=None,
    percent_difference=False,
):
    """Plot a map with about 9 discrete colour bins and a colorbar.

    Uses Cartopy for geographic projection and Matplotlib for color mapping.
    `under_color` paints every value below vmin in one colour of its own. A
    `center` makes it a diverging map with one bin centred on that value.
    `percent_difference` uses the bins of `PERCENT_DIFF_BOUNDS` instead, as
    narrow as the data allows (+-25 %, +-50 %, +-75 % or +-100 %).
    """
    _require_cartopy()

    arr = np.where(np.isinf(arr), np.nan, arr)
    extent, origin = get_map_extent_and_origin(lon, lat)
    fig, ax = _create_geo_map_axes(extent)
    bounds_type = "given" if center is None else "max"
    if percent_difference:
        bounds_type = "percent"
    image, bounds, extend, ticks = plot_single_map(
        ax,
        arr,
        diff_to_mean=None if center is None else max(vmax - center, center - vmin),
        center=center,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        bounds_type=bounds_type,
        under_color=under_color,
        extent=extent,
        origin=origin,
        transform=ccrs.PlateCarree(),
        interpolation="nearest",
    )
    add_map_colorbar(
        fig, ax, image, bounds, extend, ticks, cb_label or create_axis_label("value")
    )
    summary = create_summary_text(arr, bounds=bounds)
    _write_geo_map(
        fig,
        ax,
        title,
        f"{panel_title} ({summary})" if panel_title else summary,
        out_path,
        x_min,
        x_max,
        y_min,
        y_max,
    )


@log_errors(raise_exceptions=True)
def plot_map(  # noqa: PLR0913
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
    under_color: Optional[str] = None,
    center: Optional[float] = None,
    panel_title: Optional[str] = None,
    max_extended_vmin: Optional[float] = None,
    percent_difference: bool = False,
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
        Label for the colorbar, with units in square brackets.
    title : str
        Figure title.
    out_path : Path
        Path where the figure will be saved.
    cmap : str, optional
        Name of the Matplotlib colormap to use.
    x_min, x_max, y_min, y_max : float, optional
        Manual spatial limits for zooming.
    vmin, vmax : float, optional
        Color scale limits. If not provided, they follow `calculate_colour_limits`.
    under_color : str, optional
        Colour of every value below vmin, so it stays distinct from the scale.
    center : float, optional
        No-difference value of a diverging colormap, making the limits symmetric.
    panel_title : str, optional
        Name shown in front of the median and mean above the map.
    max_extended_vmin : float, optional
        Highest derived lower limit once values fall below it.
    percent_difference : bool, optional
        Colour the data, given in percent, with the bins of
        `PERCENT_DIFF_BOUNDS`: +-25 %, +-50 %, +-75 % or +-100 %, the
        narrowest that holds the data without its outliers.
    """
    _require_cartopy()

    # Extract longitude, latitude, and data values
    lon = data["lon"].values
    lat = data["lat"].values
    arr = np.squeeze(data.values)  # remove singleton dimension (e.g., time)
    arr = np.where(np.isinf(arr), np.nan, arr)

    # Crop to the shown region so colour limits and extend follow its values
    lon_mask = (lon >= (-np.inf if x_min is None else x_min)) & (
        lon <= (np.inf if x_max is None else x_max)
    )
    lat_mask = (lat >= (-np.inf if y_min is None else y_min)) & (
        lat <= (np.inf if y_max is None else y_max)
    )
    lon, lat = lon[lon_mask], lat[lat_mask]
    arr = arr[np.ix_(lat_mask, lon_mask)]

    map_args = {
        "lon": lon,
        "lat": lat,
        "arr": arr,
        "cb_label": cb_label,
        "title": title,
        "out_path": out_path,
        "cmap": cmap,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "panel_title": panel_title,
    }
    if np.nanmin(arr) == np.nanmax(arr):
        # Constant data plotting
        constant_value = float(np.nanmin(arr))
        plot_constant_data_map(vmin=constant_value, vmax=constant_value, **map_args)
    else:
        # plot regular discrete map
        vmin, vmax = calculate_colour_limits(
            arr,
            center=center,
            vmin=vmin,
            vmax=vmax,
            max_extended_vmin=max_extended_vmin,
        )
        plot_discrete_data_map(
            vmin=vmin,
            vmax=vmax,
            under_color=under_color,
            center=center,
            percent_difference=percent_difference,
            **map_args,
        )


def _add_map_panel_features(ax, extent):
    """Add coastlines, borders and gridlines to one cartopy panel axis."""
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.6, zorder=3)
    ax.add_feature(
        cfeature.BORDERS.with_scale("110m"), linewidth=0.4, alpha=0.7, zorder=3
    )
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.3,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    style_map_axes(ax)


@log_errors(raise_exceptions=True)
def plot_categorical_map_panels(
    grids: Sequence[Optional[xr.DataArray]],
    titles: Sequence[str],
    category_colors: Mapping[int, str],
    category_names: Mapping[int, str],
    out_path: Path,
    ncols: int = 2,
    panel_figsize: tuple = (7.0, 5.0),
) -> None:
    """Plot several categorical lat/lon maps as panels sharing one legend.

    Parameters
    ----------
    grids : Sequence[xarray.DataArray or None]
        2D `(lat, lon)` categorical data arrays, one per panel. A `None` entry
        leaves its panel blank with a "not available" title instead of failing.
    titles : Sequence[str]
        Panel titles, same length and order as `grids`. They get an "a) ",
        "b) ", ... prefix.
    category_colors : Mapping[int, str]
        Color per category id.
    category_names : Mapping[int, str]
        Legend label per category id.
    out_path : Path
        PNG file to write.
    ncols : int, optional
        Number of columns in the panel grid, by default 2.
    panel_figsize : tuple[float, float], optional
        Width and height of one panel; only their ratio is used, since the
        figure has the standard width.

    Returns
    -------
    None
    """
    _require_cartopy()

    category_ids = sorted(category_colors)
    cmap = ListedColormap([category_colors[cid] for cid in category_ids])
    boundaries = np.arange(category_ids[0] - 0.5, category_ids[-1] + 1.5, 1.0)
    norm = BoundaryNorm(boundaries, cmap.N)

    reference_grid = next((grid for grid in grids if grid is not None), None)
    if reference_grid is None:
        msg = "At least one grid is required to plot categorical map panels."
        raise ValueError(msg)
    extent, origin = get_map_extent_and_origin(
        reference_grid["lon"].values, reference_grid["lat"].values
    )

    n_panels = len(grids)
    nrows = int(np.ceil(n_panels / ncols))
    panel_width = FIGURE_WIDTH / ncols
    panel_height = panel_width * panel_figsize[1] / panel_figsize[0]
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(FIGURE_WIDTH, panel_height * nrows + 1.0),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    axes = np.atleast_1d(axes).ravel()

    for panel_index, (ax, grid, title) in enumerate(zip(axes, grids, titles)):
        panel_title = f"{chr(ord('a') + panel_index)}) {title}"
        if grid is None:
            logger.warning(f"Skipping panel {title!r}: no data available.")
            ax.set_title(f"{panel_title} (not available)")
            ax.axis("off")
            continue
        values = np.ma.masked_where(
            ~np.isfinite(grid.values) | (grid.values < category_ids[0]), grid.values
        )
        ax.imshow(
            values,
            cmap=cmap,
            norm=norm,
            origin=origin,
            extent=extent,
            transform=ccrs.PlateCarree(),
            interpolation="nearest",
        )
        _add_map_panel_features(ax, extent)
        ax.set_title(panel_title)

    for ax in axes[n_panels:]:
        ax.axis("off")

    handles = [
        mpatches.Patch(
            facecolor=category_colors[cid],
            edgecolor="none",
            label=category_names.get(cid, str(cid)),
        )
        for cid in category_ids
    ]
    fig.legend(
        handles=handles,
        ncol=min(len(category_ids), 6),
        frameon=True,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.subplots_adjust(bottom=0.1)
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    logger.info(f"Wrote {out_path}")
