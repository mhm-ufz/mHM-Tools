"""
Plotting utilities for creating geospatial maps with Cartopy and Matplotlib.

Includes functions for plotting:
- Constant data maps with a legend patch.
- Discrete data maps with colorbars and extensions.
- General wrapper `plot_map` that auto-selects plotting strategy.

Authors
-------
- Jeisson Leal
"""

from pathlib import Path
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap

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
