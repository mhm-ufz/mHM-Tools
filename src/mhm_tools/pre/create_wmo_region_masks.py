"""Create WMO-region masks for global mHM/mRM setups.

The module delineates (or reuses) global basins, rasterizes World
Meteorological Organization (WMO) regions from a shapefile set and a GeoJSON,
maps every basin to exactly one region while prioritising basins covered by
the original shapefiles and enforcing region contiguity, excludes Greenland
(or another polygon), and writes one mask per region plus a 4-panel overview
plot.

Because every region is built as a union of whole basins (never split
across a region boundary), every region boundary is composed exclusively of
drainage divides: no region boundary ever cuts a flow path, so each region
mask is a self-contained mRM routing domain with no cross-border inflow.

Authors
-------
- Simon Lüdke
"""

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from mhm_tools.common.constants import GREENLAND_COORDS
from mhm_tools.common.file_handler import get_xarray_ds_from_file, write_mask_to_file
from mhm_tools.common.logger import ErrorLogger, log_arguments
from mhm_tools.common.plotter import plot_categorical_map_panels
from mhm_tools.common.xarray_utils import (
    combine_region_grids,
    create_mask_from_polygon,
    create_valid_data_mask,
    get_single_data_var,
)
from mhm_tools.pre.fill_nearest import fill_dataarray_with_nearest

logger = logging.getLogger(__name__)

try:
    from shapely import contains_xy
except ImportError:  # pragma: no cover - fallback for older shapely
    from shapely.vectorized import contains as contains_xy

# WMO regions I-VI. Confirmed with the user: exactly these 6 regions are
# supported; Greenland, Antarctica, and any shapefile with an out-of-range
# region digit are all excluded rather than treated as a 7th region.
WMO_REGIONS = {
    1: "Africa",
    2: "Asia",
    3: "South America",
    4: "North/Central America",
    5: "SW Pacific",
    6: "Europe",
}
WMO_REGION_COLORS = {
    1: "#e41a1c",
    2: "#377eb8",
    3: "#4daf4a",
    4: "#ff7f00",
    5: "#984ea3",
    6: "#a65628",
}
N_REGIONS = len(WMO_REGIONS)
SHAPE_REGION_REGEX = r"(\d{7})$"
REGION_VAR = "wmo_region"


def _import_rasterio_features():
    """Import rasterio.features, raising an informative error if unavailable."""
    try:
        from rasterio import features
    except ImportError as exc:
        msg = "rasterio is required for shapefile/GeoJSON rasterization."
        with ErrorLogger(logger):
            raise ImportError(msg) from exc
    return features


# ---------------------------------------------------------------------------
# Step 2: region rasters from shapefiles and GeoJSON
# ---------------------------------------------------------------------------


def get_region_id_from_shape_file_name(
    shape_file, *, region_regex=SHAPE_REGION_REGEX, n_regions=N_REGIONS
):
    """Return the WMO region id encoded in a shapefile's name, or None if unusable.

    Parameters
    ----------
    shape_file : str or pathlib.Path
        Shapefile path; only the filename stem is inspected.
    region_regex : str, optional
        Regex whose first captured group's leading digit is the region id.
    n_regions : int, optional
        Number of supported regions; ids outside `1..n_regions` are rejected.

    Returns
    -------
    int or None
        The region id, or None if the name doesn't match or is out of range.
    """
    match = re.search(region_regex, Path(shape_file).stem)
    if match is None:
        return None
    region_id = int(match.group(1)[0])
    if region_id < 1 or region_id > n_regions:
        return None
    return region_id


def group_shape_files_by_region(
    shape_dir,
    *,
    region_regex=SHAPE_REGION_REGEX,
    n_regions=N_REGIONS,
    shape_file_pattern="*.shp",
):
    """Group shapefiles below `shape_dir` by their filename-encoded region id.

    Parameters
    ----------
    shape_dir : str or pathlib.Path
        Directory to search for shapefiles.
    region_regex : str, optional
        See `get_region_id_from_shape_file_name`.
    n_regions : int, optional
        Number of supported regions.
    shape_file_pattern : str, optional
        Glob pattern selecting shapefiles.

    Returns
    -------
    dict[int, list[pathlib.Path]]
        Shapefiles grouped by region id.
    """
    shape_dir = Path(shape_dir)
    groups = {}
    unusable = []
    for shape_file in sorted(shape_dir.glob(shape_file_pattern)):
        region_id = get_region_id_from_shape_file_name(
            shape_file, region_regex=region_regex, n_regions=n_regions
        )
        if region_id is None:
            unusable.append(shape_file)
            continue
        groups.setdefault(region_id, []).append(shape_file)
    if unusable:
        preview = [str(f) for f in unusable[:10]]
        logger.warning(
            f"Skipped {len(unusable)} shapefile(s) with no usable region id "
            f"(pattern {region_regex!r}, valid range 1-{n_regions}): {preview}"
            + (" ..." if len(unusable) > 10 else "")
        )
    total = sum(len(files) for files in groups.values())
    logger.info(
        f"Parsed {total} usable shapefile(s) across {len(groups)} region(s) below {shape_dir}."
    )
    return groups


def read_shape_file_regions(
    shape_dir,
    *,
    region_regex=SHAPE_REGION_REGEX,
    n_regions=N_REGIONS,
    shape_file_pattern="*.shp",
    target_crs="EPSG:4326",
):
    """Return `(shape_file, region_id, geometry)` for every usable shapefile below `shape_dir`.

    Each shapefile is read once; both the region rasterization and the anchor
    determination steps use this same list, so shapefiles are never re-read.

    Parameters
    ----------
    shape_dir : str or pathlib.Path
        Directory to search for shapefiles.
    region_regex, n_regions, shape_file_pattern : see `group_shape_files_by_region`.
    target_crs : str, optional
        CRS every geometry is reprojected to.

    Returns
    -------
    list[tuple[pathlib.Path, int, shapely geometry]]
        One row per usable shapefile.
    """
    groups = group_shape_files_by_region(
        shape_dir,
        region_regex=region_regex,
        n_regions=n_regions,
        shape_file_pattern=shape_file_pattern,
    )
    rows = []
    for region_id, shape_files in groups.items():
        for shape_file in shape_files:
            shape_gdf = gpd.read_file(shape_file)
            if shape_gdf.empty:
                continue
            if shape_gdf.crs is not None and str(shape_gdf.crs) != target_crs:
                shape_gdf = shape_gdf.to_crs(target_crs)
            geometries = [
                geom
                for geom in shape_gdf.geometry
                if geom is not None and not geom.is_empty
            ]
            if not geometries:
                continue
            geometry = (
                geometries[0]
                if len(geometries) == 1
                else _geometry_union(gpd.GeoSeries(geometries))
            )
            rows.append((shape_file, region_id, geometry))
    return rows


def _geometry_union(geometry):
    """Dissolve a GeoSeries into one geometry, handling shapely 1.x/2.x API differences."""
    if hasattr(geometry, "union_all"):
        return geometry.union_all()
    return geometry.unary_union


def create_dissolved_region_geometry(geometries, *, batch_size=200):
    """Incrementally dissolve many geometries into one, batched to bound memory.

    Parameters
    ----------
    geometries : list[shapely geometry]
        Geometries to dissolve.
    batch_size : int, optional
        Geometries unioned per batch before partial results are collapsed.

    Returns
    -------
    shapely geometry or None
        The dissolved geometry, or None if `geometries` is empty.
    """
    partials = []
    for batch_start in range(0, len(geometries), batch_size):
        batch = geometries[batch_start : batch_start + batch_size]
        partial = _geometry_union(gpd.GeoSeries(batch))
        if partial is not None and not partial.is_empty:
            partials.append(partial)
        if len(partials) >= 8:
            partials = [_geometry_union(gpd.GeoSeries(partials))]
    if not partials:
        return None
    return _geometry_union(gpd.GeoSeries(partials))


def rasterize_region_geometries(
    shape_regions, lat, lon, land_mask, transform, *, dissolve_batch_size=200
):
    """Rasterize per-region dissolved shapefile geometries onto a lat/lon grid.

    Each region's shapefiles are dissolved into one polygon and rasterized
    once; a cell already claimed by an earlier region is never overwritten
    and cells outside `land_mask` are never assigned, however large the
    region's polygon is.

    Parameters
    ----------
    shape_regions : list[tuple[pathlib.Path, int, shapely geometry]]
        Rows from `read_shape_file_regions`.
    lat, lon : numpy.ndarray
        1D target grid coordinates.
    land_mask : numpy.ndarray
        Boolean 2D `(lat, lon)` valid-land mask.
    transform : affine.Affine
        Affine transform of the target grid.
    dissolve_batch_size : int, optional
        See `create_dissolved_region_geometry`.

    Returns
    -------
    xarray.DataArray
        Region grid with NaN at every cell not covered by a shapefile -- the
        anchor/provenance input for the basin-to-region mapping step.
    """
    features = _import_rasterio_features()

    by_region = {}
    for _, region_id, geometry in shape_regions:
        by_region.setdefault(region_id, []).append(geometry)

    out = np.full((len(lat), len(lon)), np.nan, dtype=np.float32)
    for region_id in sorted(by_region):
        geometries = by_region[region_id]
        logger.info(f"Region {region_id}: dissolving {len(geometries)} shape(s).")
        geometry = create_dissolved_region_geometry(
            geometries, batch_size=dissolve_batch_size
        )
        if geometry is None or geometry.is_empty:
            logger.warning(f"Region {region_id}: no usable geometry after dissolve.")
            continue
        raster = features.rasterize(
            [(geometry, region_id)],
            out_shape=out.shape,
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=False,
        )
        assigned_elsewhere = np.isfinite(out) & (raster > 0)
        if np.any(assigned_elsewhere):
            logger.warning(
                f"Region {region_id}: {int(assigned_elsewhere.sum())} cell(s) already "
                "assigned to another region; keeping the earlier assignment."
            )
        assign = (raster > 0) & ~np.isfinite(out) & land_mask
        out[assign] = region_id
    return xr.DataArray(
        out, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"], name=REGION_VAR
    )


def create_region_grid_from_geojson_file(
    geojson_file, lat, lon, land_mask, *, region_property="WMO_RA", n_regions=N_REGIONS
):
    """Rasterize a WMO-region GeoJSON onto a lat/lon grid via point-in-polygon.

    Parameters
    ----------
    geojson_file : str or pathlib.Path
        WMO region GeoJSON (e.g. https://github.com/OGCMetOceanDWG/wmo-ra).
    lat, lon : numpy.ndarray
        1D target grid coordinates.
    land_mask : numpy.ndarray
        Boolean 2D `(lat, lon)` valid-land mask.
    region_property : str, optional
        Integer region-id property on each GeoJSON feature.
    n_regions : int, optional
        Number of supported regions; out-of-range feature ids are skipped.

    Returns
    -------
    xarray.DataArray
        Region grid, NaN outside every feature (or outside `land_mask`).
    """
    region_gdf = gpd.read_file(geojson_file)
    grid = np.full((len(lat), len(lon)), np.nan, dtype=np.float32)
    lon2d, lat2d = np.meshgrid(lon, lat)
    for _, row in region_gdf.iterrows():
        region_id = int(row[region_property])
        if region_id < 1 or region_id > n_regions:
            logger.warning(
                f"Skipping GeoJSON feature with out-of-range region id {region_id}."
            )
            continue
        inside = contains_xy(row.geometry, lon2d, lat2d)
        grid[inside & land_mask] = region_id
    return xr.DataArray(
        grid, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"], name=REGION_VAR
    )


def fill_region_grid_gaps(region_grid, land_mask):
    """Fill remaining unset valid land cells via nearest-neighbour; non-land cells become 0.

    Parameters
    ----------
    region_grid : xarray.DataArray
        Region grid, NaN at unset cells.
    land_mask : numpy.ndarray
        Boolean 2D `(lat, lon)` valid-land mask.

    Returns
    -------
    xarray.DataArray
        Filled region grid; every valid land cell has a region id.
    """
    filled = region_grid.copy(deep=True)
    filled_count = fill_dataarray_with_nearest(
        filled,
        missing_value=None,
        mask=~np.asarray(land_mask, dtype=bool),
        fill_value=0,
    )
    logger.info(f"Filled {filled_count} region cell(s) via nearest-neighbour.")
    return filled


def create_exclude_mask(data_array, exclude_polygon):
    """Return a boolean exclusion mask for the given exclude-polygon spec, or None to disable.

    Parameters
    ----------
    data_array : xarray.DataArray
        Reference 2D array with `lat`/`lon` coordinates defining the grid.
    exclude_polygon : {"greenland", "none"} or str or pathlib.Path or numpy.ndarray or None
        `"greenland"` uses the built-in Greenland outline; `"none"`/`None`
        disables exclusion; a path to a shapefile/GeoJSON excludes every
        polygon found there; an array of (lon, lat) vertices is used directly.

    Returns
    -------
    numpy.ndarray or None
        Boolean mask, True for excluded cells, or None if exclusion is disabled.
    """
    if exclude_polygon is None:
        return None
    if isinstance(exclude_polygon, str) and exclude_polygon.strip().lower() == "none":
        return None
    if (
        isinstance(exclude_polygon, str)
        and exclude_polygon.strip().lower() == "greenland"
    ):
        return create_mask_from_polygon(data_array, GREENLAND_COORDS)
    if isinstance(exclude_polygon, np.ndarray):
        return create_mask_from_polygon(data_array, exclude_polygon)

    path = Path(exclude_polygon)
    if not path.is_file():
        msg = f"Exclude-polygon file not found: {path}"
        with ErrorLogger(logger):
            raise FileNotFoundError(msg)
    exclude_gdf = gpd.read_file(path)
    if exclude_gdf.crs is not None and str(exclude_gdf.crs) != "EPSG:4326":
        exclude_gdf = exclude_gdf.to_crs("EPSG:4326")
    mask = np.zeros(data_array.shape, dtype=bool)
    for geometry in exclude_gdf.geometry:
        if geometry is None or geometry.is_empty:
            continue
        polygons = geometry.geoms if hasattr(geometry, "geoms") else [geometry]
        for polygon in polygons:
            mask |= create_mask_from_polygon(
                data_array, np.asarray(polygon.exterior.coords)
            )
    return mask


# ---------------------------------------------------------------------------
# Step 3: map every basin to a region (shape-basin priority + contiguity)
# ---------------------------------------------------------------------------


@dataclass
class BasinIndex:
    """Dense node indexing for the valid basin ids of a basin id raster."""

    unique_ids: np.ndarray
    id_to_node: np.ndarray
    cell_counts: np.ndarray

    @property
    def n_nodes(self):
        """Number of distinct valid basins."""
        return self.unique_ids.size


def create_basin_node_index(basin_ids, *, nodata_value=0):
    """Build a dense node index for the valid (non-nodata) ids in a basin raster.

    Replaces `numpy.unique` over the whole grid with a single `bincount` pass,
    and gives every downstream step an O(1) basin-id-to-node lookup instead of
    a per-basin full-grid scan.

    Parameters
    ----------
    basin_ids : numpy.ndarray
        Integer basin id raster; `nodata_value` marks invalid cells.
    nodata_value : int, optional
        Value marking invalid/no-basin cells, by default 0.

    Returns
    -------
    BasinIndex
        Dense node index over the valid basin ids.
    """
    basin_ids = np.asarray(basin_ids)
    max_id = int(basin_ids.max()) if basin_ids.size else 0
    counts = np.bincount(
        basin_ids.ravel().astype(np.intp, copy=False), minlength=max_id + 1
    )
    counts[nodata_value] = 0
    unique_ids = np.flatnonzero(counts)
    id_to_node = np.full(max_id + 1, -1, dtype=np.int64)
    id_to_node[unique_ids] = np.arange(unique_ids.size, dtype=np.int64)
    cell_counts = counts[unique_ids].astype(np.int64)
    logger.info(f"Indexed {unique_ids.size} unique basin(s).")
    return BasinIndex(
        unique_ids=unique_ids, id_to_node=id_to_node, cell_counts=cell_counts
    )


def create_basin_adjacency_edges(
    basin_ids, basin_index, *, connectivity=8, wrap_longitude=False, nodata_value=0
):
    """Return undirected basin-adjacency edges and their shared border cell counts.

    Uses shifted-array comparison (compare the grid against itself shifted by
    one cell in each direction) instead of per-basin dilation, which would be
    hopeless at basin counts in the hundreds of thousands.

    Parameters
    ----------
    basin_ids : numpy.ndarray
        Integer basin id raster, nodata is `nodata_value`.
    basin_index : BasinIndex
        Node index built from the same `basin_ids` array.
    connectivity : {4, 8}, optional
        Pixel adjacency used to detect basin borders, by default 8.
    wrap_longitude : bool, optional
        Also compare the first and last columns, for a globally wrapping grid.
    nodata_value : int, optional
        Value marking invalid/no-basin cells.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        `node_a`, `node_b` (each edge listed once, `node_a <= node_b`), and
        `border_cell_counts` (shared border length per edge, in cells).
    """
    id_to_node = basin_index.id_to_node
    n_nodes = basin_index.n_nodes

    def _pairs_from_shift(a, b):
        a, b = np.asarray(a), np.asarray(b)
        keep = (a != b) & (a != nodata_value) & (b != nodata_value)
        if not np.any(keep):
            return np.empty(0, dtype=np.int64)
        node_a = id_to_node[a[keep]]
        node_b = id_to_node[b[keep]]
        valid = (node_a >= 0) & (node_b >= 0)
        node_a, node_b = node_a[valid], node_b[valid]
        lo = np.minimum(node_a, node_b).astype(np.int64)
        hi = np.maximum(node_a, node_b).astype(np.int64)
        return lo * n_nodes + hi

    key_batches = [
        _pairs_from_shift(basin_ids[:-1, :], basin_ids[1:, :]),
        _pairs_from_shift(basin_ids[:, :-1], basin_ids[:, 1:]),
    ]
    if connectivity == 8:
        key_batches.append(_pairs_from_shift(basin_ids[:-1, :-1], basin_ids[1:, 1:]))
        key_batches.append(_pairs_from_shift(basin_ids[:-1, 1:], basin_ids[1:, :-1]))
    if wrap_longitude:
        key_batches.append(_pairs_from_shift(basin_ids[:, -1], basin_ids[:, 0]))

    all_keys = np.concatenate(key_batches)
    if all_keys.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty
    keys, border_cell_counts = np.unique(all_keys, return_counts=True)
    node_a, node_b = np.divmod(keys, n_nodes)
    logger.info(f"Built {keys.size} basin-adjacency edge(s).")
    return node_a, node_b, border_cell_counts.astype(np.int64)


def create_adjacency_matrix(node_a, node_b, n_nodes):
    """Build a symmetric CSR adjacency matrix (unit weights) from an edge list.

    Parameters
    ----------
    node_a, node_b : numpy.ndarray
        Edge endpoints.
    n_nodes : int
        Total number of basin nodes.

    Returns
    -------
    scipy.sparse.csr_matrix
        Symmetric adjacency matrix with unit edge weights.
    """
    if node_a.size == 0:
        return coo_matrix((n_nodes, n_nodes)).tocsr()
    rows = np.concatenate([node_a, node_b])
    cols = np.concatenate([node_b, node_a])
    data = np.ones(rows.size, dtype=np.float64)
    adjacency = coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    # coo->csr sums duplicate (row, col) entries; force unit weights so a
    # surviving duplicate can never silently become a weight-2 edge.
    adjacency.data[:] = 1.0
    return adjacency


def calculate_basin_region_overlap_counts(
    basin_ids, region_raster, basin_index, *, n_regions=N_REGIONS, nodata_value=0
):
    """Count, per basin node, how many cells overlap each region id.

    One vectorized O(n_cells) pass, replacing the O(n_basins x n_cells) loop
    this step supersedes.

    Parameters
    ----------
    basin_ids : numpy.ndarray
        Integer basin id raster, nodata is `nodata_value`.
    region_raster : numpy.ndarray
        Integer region id raster (`1..n_regions`), 0 = nodata, same grid.
    basin_index : BasinIndex
        Node index built from `basin_ids`.
    n_regions : int, optional
        Number of regions.
    nodata_value : int, optional
        Value marking invalid/no-basin cells in `basin_ids`.

    Returns
    -------
    numpy.ndarray
        `(n_regions + 1, n_nodes)` overlap-cell-count array; index 0 along the
        first axis is unused (regions are numbered from 1).
    """
    id_to_node = basin_index.id_to_node
    n_nodes = basin_index.n_nodes
    overlap_counts = np.zeros((n_regions + 1, n_nodes), dtype=np.int64)
    valid = (
        (basin_ids != nodata_value)
        & (region_raster >= 1)
        & (region_raster <= n_regions)
    )
    if not np.any(valid):
        return overlap_counts
    node = id_to_node[basin_ids[valid]]
    region = region_raster[valid].astype(np.int64)
    keep = node >= 0
    node, region = node[keep], region[keep]
    for region_id in range(1, n_regions + 1):
        selected = region == region_id
        if np.any(selected):
            overlap_counts[region_id] += np.bincount(node[selected], minlength=n_nodes)
    return overlap_counts


def calculate_shape_basin_overlap(
    geometry, basin_ids, transform, *, dilate_on_empty=True
):
    """Return basin ids and cell counts overlapping one shape geometry, largest first.

    Rasterizes only the geometry's bounding-box window against `basin_ids`
    (never the full grid) using an escalation ladder for small/edge-case
    shapes: interior-only, then `all_touched=True`, then one cell of
    dilation -- stopping as soon as any basin cell is matched.

    Parameters
    ----------
    geometry : shapely geometry
        Shape to rasterize against the basin grid.
    basin_ids : numpy.ndarray
        Integer basin id raster, nodata is 0.
    transform : affine.Affine
        Affine transform of `basin_ids`.
    dilate_on_empty : bool, optional
        Allow the one-cell dilation escalation step.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Basin ids and matching overlap cell counts, sorted by count
        descending. Both arrays are empty if no basin cells could be matched.
    """
    from affine import Affine
    from scipy.ndimage import binary_dilation

    features = _import_rasterio_features()
    height, width = basin_ids.shape
    minx, miny, maxx, maxy = geometry.bounds
    inv = ~transform
    col0f, row0f = inv * (minx, maxy)
    col1f, row1f = inv * (maxx, miny)
    row0 = max(0, int(np.floor(min(row0f, row1f))) - 1)
    row1 = min(height, int(np.ceil(max(row0f, row1f))) + 1)
    col0 = max(0, int(np.floor(min(col0f, col1f))) - 1)
    col1 = min(width, int(np.ceil(max(col0f, col1f))) + 1)
    if row0 >= row1 or col0 >= col1:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    window_transform = transform * Affine.translation(col0, row0)
    window_basins = basin_ids[row0:row1, col0:col1]

    shape_mask = np.zeros_like(window_basins, dtype=bool)
    for all_touched in (False, True):
        shape_mask = features.rasterize(
            [(geometry, 1)],
            out_shape=window_basins.shape,
            transform=window_transform,
            fill=0,
            dtype="uint8",
            all_touched=all_touched,
        ).astype(bool)
        if np.any(shape_mask & (window_basins != 0)):
            break
    if dilate_on_empty and not np.any(shape_mask & (window_basins != 0)):
        shape_mask = binary_dilation(shape_mask, structure=np.ones((3, 3)))

    overlap = window_basins[shape_mask & (window_basins != 0)]
    if overlap.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    basin_ids_hit, counts = np.unique(overlap, return_counts=True)
    order = np.argsort(counts)[::-1]
    return basin_ids_hit[order], counts[order]


def create_anchor_basin_regions(
    shape_regions,
    basin_ids,
    transform,
    *,
    min_shape_overlap_fraction=0.5,
    strict_anchors=False,
):
    """Determine hard-pinned basin-to-region anchors from individual shapefiles.

    A shapefile's plurality basin becomes an anchor only if the shapefile is
    itself concentrated on that basin (`min_shape_overlap_fraction`); how much
    of that basin the shapefile covers is irrelevant and is never a rejection
    criterion, since a small named tributary shapefile inside a much larger
    basin is still valid evidence for that basin's region.

    Parameters
    ----------
    shape_regions : list[tuple[pathlib.Path, int, shapely geometry]]
        `(shape_file, region_id, geometry)` rows, e.g. from `read_shape_file_regions`.
    basin_ids : numpy.ndarray
        Integer basin id raster, nodata is 0, same grid as the region rasters.
    transform : affine.Affine
        Affine transform of `basin_ids`.
    min_shape_overlap_fraction : float, optional
        Minimum fraction of a shape's overlapping cells that must fall in its
        plurality basin for that basin to become an anchor.
    strict_anchors : bool, optional
        Raise instead of warning when two shapefiles claim the same basin for
        different regions.

    Returns
    -------
    dict[int, int]
        `{basin_id: region_id}` anchor mapping.
    """
    candidates = {}
    skipped = 0
    for shape_file, region_id, geometry in shape_regions:
        basin_ids_hit, counts = calculate_shape_basin_overlap(
            geometry, basin_ids, transform
        )
        if basin_ids_hit.size == 0:
            logger.warning(
                f"Shapefile {shape_file} matched no basin cells; skipping as an anchor."
            )
            skipped += 1
            continue
        total = int(counts.sum())
        best_basin_id, best_count = int(basin_ids_hit[0]), int(counts[0])
        if best_count / total < min_shape_overlap_fraction:
            top_candidates = list(zip(basin_ids_hit[:3].tolist(), counts[:3].tolist()))
            logger.warning(
                f"Shapefile {shape_file}: plurality basin {best_basin_id} covers only "
                f"{best_count}/{total} ({best_count / total:.0%}) of its overlap "
                f"(< {min_shape_overlap_fraction:.0%}); skipping as an anchor. "
                f"Top candidates (basin_id, cells): {top_candidates}"
            )
            skipped += 1
            continue
        candidates.setdefault(best_basin_id, []).append(
            (region_id, best_count, shape_file)
        )

    anchors = {}
    conflicts = 0
    for basin_id, claims in candidates.items():
        regions_claimed = {region_id for region_id, _, _ in claims}
        if len(regions_claimed) == 1:
            anchors[basin_id] = claims[0][0]
            continue
        conflicts += 1
        claims_sorted = sorted(claims, key=lambda claim: claim[1], reverse=True)
        winner_region, winner_count, winner_file = claims_sorted[0]
        msg = (
            f"Basin {basin_id}: {len(claims)} shapefile(s) claim different regions "
            f"({claims_sorted}); using region {winner_region} from {winner_file} "
            f"(largest overlap count {winner_count})."
        )
        if strict_anchors:
            with ErrorLogger(logger):
                raise ValueError(msg)
        logger.warning(msg)
        anchors[basin_id] = winner_region

    logger.info(
        f"Determined {len(anchors)} anchor basin(s) from {len(shape_regions)} shapefile(s) "
        f"({skipped} skipped, {conflicts} region conflict(s))."
    )
    return anchors


def create_region_labels_from_anchors(
    adjacency, anchor_nodes, anchor_regions, n_nodes, *, n_regions=N_REGIONS
):
    """Multi-source-BFS region growing from hard-pinned anchor basins.

    Runs one multi-source Dijkstra search per region (never all regions'
    seeds mixed in one call, since `min_only` breaks ties by heap-pop order,
    which would make ties invisible and non-deterministic), keeping a running
    best distance per node. An anchor always has distance 0 in its own region
    and therefore always wins, with no special-casing required.

    Parameters
    ----------
    adjacency : scipy.sparse.csr_matrix
        Undirected basin adjacency matrix, unit edge weights.
    anchor_nodes : numpy.ndarray
        Node indices of every anchor basin.
    anchor_regions : numpy.ndarray
        Region id per entry in `anchor_nodes`.
    n_nodes : int
        Total number of basin nodes.
    n_regions : int, optional
        Number of regions.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        `region_by_node` (0 = unassigned), `hop_distance_by_node` (-1 =
        unreached), and `is_tied`, a boolean `(n_regions + 1, n_nodes)` array.
    """
    best_dist = np.full(n_nodes, np.inf)
    region_by_node = np.zeros(n_nodes, dtype=np.int16)
    is_tied = np.zeros((n_regions + 1, n_nodes), dtype=bool)

    for region_id in range(1, n_regions + 1):
        seeds = anchor_nodes[anchor_regions == region_id]
        if seeds.size == 0:
            logger.warning(f"Region {region_id} has no anchor basins.")
            continue
        dist = dijkstra(adjacency, directed=False, indices=seeds, min_only=True)
        closer = dist < best_dist
        equal = (dist == best_dist) & np.isfinite(dist)
        is_tied[:, closer] = False
        is_tied[region_id, closer | equal] = True
        region_by_node[closer] = region_id
        best_dist = np.minimum(best_dist, dist)

    hop_distance_by_node = np.where(np.isfinite(best_dist), best_dist, -1).astype(
        np.int32
    )
    return region_by_node, hop_distance_by_node, is_tied


def calculate_region_component_labels(node_a, node_b, region_by_node, n_nodes):
    """Return connected components of the same-region subgraph.

    Two basins are in the same component here only if connected entirely
    through basins sharing their assigned region -- the graph-space
    equivalent of `scipy.ndimage.label` on the final per-pixel region raster,
    except it operates on whole basins, so it can never split one basin's
    cells across two components.

    Parameters
    ----------
    node_a, node_b : numpy.ndarray
        Basin adjacency edge endpoints.
    region_by_node : numpy.ndarray
        Region id per node.
    n_nodes : int
        Total number of basin nodes.

    Returns
    -------
    tuple[int, numpy.ndarray]
        Number of components and the component id per node.
    """
    if node_a.size == 0:
        return n_nodes, np.arange(n_nodes, dtype=np.int64)
    same_region = region_by_node[node_a] == region_by_node[node_b]
    if not np.any(same_region):
        return n_nodes, np.arange(n_nodes, dtype=np.int64)
    rows = np.concatenate([node_a[same_region], node_b[same_region]])
    cols = np.concatenate([node_b[same_region], node_a[same_region]])
    same_region_graph = coo_matrix(
        (np.ones(rows.size), (rows, cols)), shape=(n_nodes, n_nodes)
    ).tocsr()
    return connected_components(same_region_graph, directed=False)


def calculate_component_border_composition(
    node_a,
    node_b,
    border_cell_counts,
    component_labels,
    region_by_node,
    n_components,
    n_regions=N_REGIONS,
):
    """Return, per same-region component, its shared-border composition with neighbouring regions.

    Parameters
    ----------
    node_a, node_b, border_cell_counts : numpy.ndarray
        Basin adjacency edges and their shared border cell counts.
    component_labels : numpy.ndarray
        Component id per node, from `calculate_region_component_labels`.
    region_by_node : numpy.ndarray
        Region id per node.
    n_components : int
        Number of components.
    n_regions : int, optional
        Number of regions.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        `tally` `(n_components, n_regions + 1)` shared border cell counts per
        neighbouring region (a component's own region is zeroed out),
        `dominant_region` per component, and `surround_fraction` (dominant
        neighbour's share of the component's total external border).
    """
    tally = np.zeros((n_components, n_regions + 1), dtype=np.int64)
    if node_a.size:
        comp_a = component_labels[node_a]
        comp_b = component_labels[node_b]
        cross = comp_a != comp_b
        if np.any(cross):
            np.add.at(
                tally,
                (comp_a[cross], region_by_node[node_b[cross]]),
                border_cell_counts[cross],
            )
            np.add.at(
                tally,
                (comp_b[cross], region_by_node[node_a[cross]]),
                border_cell_counts[cross],
            )

    # Any member node's region is the component's own region (all members of
    # a same-region component share it by construction); scatter one owner
    # node id per component to look it up.
    component_owner_node = np.zeros(n_components, dtype=np.int64)
    component_owner_node[component_labels] = np.arange(region_by_node.size)
    own_region = region_by_node[component_owner_node]
    tally[np.arange(n_components), own_region] = 0

    totals = tally.sum(axis=1)
    dominant_region = tally.argmax(axis=1)
    surround_fraction = np.divide(
        tally[np.arange(n_components), dominant_region],
        np.maximum(totals, 1),
        out=np.zeros(n_components, dtype=np.float64),
        where=totals > 0,
    )
    return tally, dominant_region, surround_fraction


def set_region_labels_for_ties_and_unreached(
    region_by_node,
    hop_distance_by_node,
    is_tied,
    overlap_counts,
    component_labels,
    n_components,
    *,
    max_component_cells_fraction=0.005,
    cell_counts=None,
):
    """Resolve ties by raster-overlap vote and assign anchor-free components a region.

    Parameters
    ----------
    region_by_node : numpy.ndarray
        Region id per node from `create_region_labels_from_anchors`.
    hop_distance_by_node : numpy.ndarray
        Hop distance per node, -1 = unreached (no anchor in the same component).
    is_tied : numpy.ndarray
        Boolean `(n_regions + 1, n_nodes)` tie bitmap.
    overlap_counts : numpy.ndarray
        `(n_regions + 1, n_nodes)` raster-overlap vote counts per node.
    component_labels : numpy.ndarray
        Connected-component id per node over the full adjacency graph
        (independent of region assignment), used to detect anchor-free islands.
    n_components : int
        Number of components in `component_labels`.
    max_component_cells_fraction : float, optional
        Above this fraction of total basin cells, log a warning instead of
        silently blanket-assigning the whole component one region.
    cell_counts : numpy.ndarray, optional
        Per-node cell counts.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Updated `region_by_node` and an `assignment_method_by_node` code array
        (0=anchor, 1=bfs, 2=tie_vote, 3=component_vote, 4=unassigned).
    """
    region_by_node = region_by_node.copy()
    assignment_method = np.where(hop_distance_by_node == 0, 0, 1).astype(np.int8)

    tie_mask = is_tied.sum(axis=0) > 1
    if np.any(tie_mask):
        masked = np.where(is_tied, overlap_counts, -1)
        tie_winner = masked[:, tie_mask].argmax(axis=0)
        region_by_node[tie_mask] = tie_winner
        assignment_method[tie_mask] = 2
        logger.info(
            f"Resolved {int(tie_mask.sum())} tied basin(s) via raster-overlap vote."
        )

    unreached = hop_distance_by_node < 0
    if np.any(unreached):
        total_cells = (
            cell_counts.sum() if cell_counts is not None else region_by_node.size
        )
        for component_id in range(n_components):
            component_mask = (component_labels == component_id) & unreached
            if not np.any(component_mask):
                continue
            component_cells = (
                int(cell_counts[component_mask].sum())
                if cell_counts is not None
                else int(component_mask.sum())
            )
            component_votes = overlap_counts[:, component_mask].sum(axis=1)
            if component_votes.sum() == 0:
                logger.warning(
                    f"Anchor-free component {component_id} ({int(component_mask.sum())} "
                    "basin(s)) has no raster-overlap votes at all; leaving unassigned."
                )
                continue
            winner = int(component_votes.argmax())
            if component_cells > max_component_cells_fraction * total_cells:
                logger.warning(
                    f"Anchor-free component {component_id} is large "
                    f"({component_cells}/{int(total_cells)} cells, "
                    f">{max_component_cells_fraction:.1%}); assigning region {winner} by "
                    "component-wide vote, but this usually means anchor shapefiles are "
                    "missing for a whole landmass -- verify this result."
                )
            region_by_node[component_mask] = winner
            assignment_method[component_mask] = 3
            logger.info(
                f"Anchor-free component {component_id} ({int(component_mask.sum())} basin(s), "
                f"{component_cells} cell(s)): assigned region {winner} by component-wide vote."
            )

    still_unassigned = region_by_node == 0
    if np.any(still_unassigned):
        assignment_method[still_unassigned] = 4
        logger.warning(
            f"{int(still_unassigned.sum())} basin(s) remain unassigned to any region."
        )
    return region_by_node, assignment_method


def set_region_labels_for_small_islands(
    region_by_node,
    node_a,
    node_b,
    border_cell_counts,
    cell_counts,
    *,
    anchor_nodes=None,
    n_regions=N_REGIONS,
    max_island_cells_fraction=0.001,
    min_surround_fraction=0.8,
    max_iterations=3,
    protect_anchor_islands=True,
):
    """Reassign small, mostly-surrounded same-region components to their dominant neighbour.

    Operates in graph space (via `calculate_region_component_labels`), so it
    can only reassign whole basins, never split one -- unlike a pixel-space
    cleanup, which could. An anchor-containing component is a deliberate,
    user-asserted exception: it is never reassigned (only logged), since
    overriding it would silently undo shape-basin prioritisation.

    Parameters
    ----------
    region_by_node : numpy.ndarray
        Region id per node.
    node_a, node_b, border_cell_counts : numpy.ndarray
        Basin adjacency edges and their shared border cell counts.
    cell_counts : numpy.ndarray
        Cell count per basin node.
    anchor_nodes : numpy.ndarray, optional
        Node indices that must never be reassigned by this cleanup pass.
    n_regions : int, optional
        Number of regions.
    max_island_cells_fraction : float, optional
        Reassign only components with at most this fraction of total land cells.
    min_surround_fraction : float, optional
        Reassign only components whose shared border is at least this
        fraction made up of one single dominant neighbouring region.
    max_iterations : int, optional
        Reassigning a component can expose a second-order island; repeat up
        to this many passes or until nothing changes.
    protect_anchor_islands : bool, optional
        Never reassign a component containing an anchor basin.

    Returns
    -------
    tuple[numpy.ndarray, list[dict]]
        Updated `region_by_node` and a log of every reassignment/protected-skip.
    """
    region_by_node = region_by_node.copy()
    n_nodes = region_by_node.size
    total_cells = cell_counts.sum()
    max_island_cells = max_island_cells_fraction * total_cells
    anchor_node_set = set(anchor_nodes.tolist()) if anchor_nodes is not None else set()
    island_log = []

    for iteration in range(max_iterations):
        n_components, component_labels = calculate_region_component_labels(
            node_a, node_b, region_by_node, n_nodes
        )
        comp_cells = np.bincount(
            component_labels, weights=cell_counts, minlength=n_components
        )
        tally, dominant_region, surround_fraction = (
            calculate_component_border_composition(
                node_a,
                node_b,
                border_cell_counts,
                component_labels,
                region_by_node,
                n_components,
                n_regions=n_regions,
            )
        )
        changed = False
        for component_id in range(n_components):
            if comp_cells[component_id] > max_island_cells:
                continue
            if surround_fraction[component_id] < min_surround_fraction:
                continue
            if tally[component_id].sum() == 0:
                continue
            component_nodes = np.flatnonzero(component_labels == component_id)
            current_region = int(region_by_node[component_nodes[0]])
            new_region = int(dominant_region[component_id])
            if new_region == current_region:
                continue
            has_anchor = protect_anchor_islands and any(
                node in anchor_node_set for node in component_nodes.tolist()
            )
            if has_anchor:
                island_log.append(
                    {
                        "iteration": iteration,
                        "action": "protected",
                        "cells": int(comp_cells[component_id]),
                        "basins": int(component_nodes.size),
                        "surround_fraction": float(surround_fraction[component_id]),
                        "current_region": current_region,
                        "dominant_neighbour_region": new_region,
                    }
                )
                continue
            region_by_node[component_nodes] = new_region
            island_log.append(
                {
                    "iteration": iteration,
                    "action": "reassigned",
                    "cells": int(comp_cells[component_id]),
                    "basins": int(component_nodes.size),
                    "surround_fraction": float(surround_fraction[component_id]),
                    "from_region": current_region,
                    "to_region": new_region,
                }
            )
            changed = True
        n_reassigned = sum(
            1
            for entry in island_log
            if entry["iteration"] == iteration and entry["action"] == "reassigned"
        )
        n_protected = sum(
            1
            for entry in island_log
            if entry["iteration"] == iteration and entry["action"] == "protected"
        )
        logger.info(
            f"Cleanup pass {iteration + 1}: {n_reassigned} reassigned, {n_protected} protected."
        )
        if not changed:
            break
    return region_by_node, island_log


def create_region_raster_from_basin_labels(
    basin_ids, basin_index, region_by_node, *, nodata_value=0
):
    """Expand per-basin-node region labels back into a per-pixel raster.

    Parameters
    ----------
    basin_ids : numpy.ndarray
        Integer basin id raster, nodata is `nodata_value`.
    basin_index : BasinIndex
        Node index built from `basin_ids`.
    region_by_node : numpy.ndarray
        Region id per node.
    nodata_value : int, optional
        Value marking invalid/no-basin cells.

    Returns
    -------
    numpy.ndarray
        Per-pixel region id raster, same shape as `basin_ids`.
    """
    id_to_region = np.zeros(basin_index.id_to_node.size, dtype=region_by_node.dtype)
    id_to_region[basin_index.unique_ids] = region_by_node
    raster = id_to_region[basin_ids]
    raster[basin_ids == nodata_value] = nodata_value
    return raster


def create_basin_region_assignment(
    basin_ids,
    region_raster,
    shape_regions,
    transform,
    *,
    n_regions=N_REGIONS,
    connectivity=8,
    wrap_longitude=False,
    min_shape_overlap_fraction=0.5,
    strict_anchors=False,
    max_island_cells_fraction=0.001,
    min_surround_fraction=0.8,
    max_cleanup_iterations=3,
    protect_anchor_islands=True,
):
    """Map every global basin to exactly one region: anchor priority, then contiguity.

    Parameters
    ----------
    basin_ids : numpy.ndarray
        Integer basin id raster, nodata is 0.
    region_raster : numpy.ndarray
        Integer region id raster (`1..n_regions`), nodata is 0, same grid as `basin_ids`.
    shape_regions : list[tuple[pathlib.Path, int, shapely geometry]]
        Individual shapefile geometries with their filename-encoded region id,
        used to determine hard-pinned anchor basins.
    transform : affine.Affine
        Affine transform shared by `basin_ids`/`region_raster`.
    n_regions : int, optional
        Number of regions.
    connectivity : {4, 8}, optional
        Pixel adjacency used to build the basin graph.
    wrap_longitude : bool, optional
        Treat the grid as wrapping at the antimeridian.
    min_shape_overlap_fraction, strict_anchors : see `create_anchor_basin_regions`.
    max_island_cells_fraction, min_surround_fraction, max_cleanup_iterations,
    protect_anchor_islands : see `set_region_labels_for_small_islands`.

    Returns
    -------
    tuple[numpy.ndarray, dict[int, int], dict]
        Final per-basin region raster, `{basin_id: region_id}` mapping, and a
        diagnostics dict.
    """
    basin_index = create_basin_node_index(basin_ids)
    node_a, node_b, border_cell_counts = create_basin_adjacency_edges(
        basin_ids, basin_index, connectivity=connectivity, wrap_longitude=wrap_longitude
    )
    adjacency = create_adjacency_matrix(node_a, node_b, basin_index.n_nodes)
    overlap_counts = calculate_basin_region_overlap_counts(
        basin_ids, region_raster, basin_index, n_regions=n_regions
    )

    anchors = create_anchor_basin_regions(
        shape_regions,
        basin_ids,
        transform,
        min_shape_overlap_fraction=min_shape_overlap_fraction,
        strict_anchors=strict_anchors,
    )
    anchor_basin_ids = np.array(sorted(anchors), dtype=np.int64)
    anchor_nodes = (
        basin_index.id_to_node[anchor_basin_ids]
        if anchor_basin_ids.size
        else np.empty(0, dtype=np.int64)
    )
    anchor_regions = np.array(
        [anchors[basin_id] for basin_id in anchor_basin_ids.tolist()], dtype=np.int16
    )
    valid_anchor = anchor_nodes >= 0
    if not np.all(valid_anchor):
        logger.warning(
            f"{int((~valid_anchor).sum())} anchor basin id(s) are not present in basin_ids; ignoring."
        )
    anchor_nodes, anchor_regions = (
        anchor_nodes[valid_anchor],
        anchor_regions[valid_anchor],
    )

    region_by_node, hop_distance_by_node, is_tied = create_region_labels_from_anchors(
        adjacency,
        anchor_nodes,
        anchor_regions,
        basin_index.n_nodes,
        n_regions=n_regions,
    )
    if anchor_nodes.size:
        mismatched = region_by_node[anchor_nodes] != anchor_regions
        if np.any(mismatched):
            logger.error(
                f"{int(mismatched.sum())} anchor basin(s) did not retain their assigned "
                "region after region growing; this indicates a bug."
            )

    n_full_components, full_component_labels = connected_components(
        adjacency, directed=False
    )
    region_by_node, _assignment_method = set_region_labels_for_ties_and_unreached(
        region_by_node,
        hop_distance_by_node,
        is_tied,
        overlap_counts,
        full_component_labels,
        n_full_components,
        cell_counts=basin_index.cell_counts,
    )

    region_by_node, island_log = set_region_labels_for_small_islands(
        region_by_node,
        node_a,
        node_b,
        border_cell_counts,
        basin_index.cell_counts,
        anchor_nodes=anchor_nodes,
        n_regions=n_regions,
        max_island_cells_fraction=max_island_cells_fraction,
        min_surround_fraction=min_surround_fraction,
        max_iterations=max_cleanup_iterations,
        protect_anchor_islands=protect_anchor_islands,
    )

    n_final_components, _ = calculate_region_component_labels(
        node_a, node_b, region_by_node, basin_index.n_nodes
    )
    logger.info(
        f"Final basin-to-region mapping: {basin_index.n_nodes} basin(s), {len(anchors)} "
        f"anchor(s), {n_final_components} same-region connected component(s) "
        f"(target: {n_regions} plus any protected anchor islands)."
    )

    final_raster = create_region_raster_from_basin_labels(
        basin_ids, basin_index, region_by_node
    )
    basin_region_map = dict(
        zip(basin_index.unique_ids.tolist(), region_by_node.tolist())
    )
    diagnostics = {
        "n_basins": int(basin_index.n_nodes),
        "n_anchors": int(anchor_nodes.size),
        "n_same_region_components": int(n_final_components),
        "n_unassigned": int(np.sum(region_by_node == 0)),
        "n_reassigned_islands": sum(
            1 for entry in island_log if entry["action"] == "reassigned"
        ),
        "n_protected_islands": sum(
            1 for entry in island_log if entry["action"] == "protected"
        ),
        "islands": island_log,
    }
    return final_raster, basin_region_map, diagnostics


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _grid_matches(cached_da, lat, lon):
    """Check whether a cached DataArray's lat/lon grid matches the target grid."""
    if cached_da.sizes.get("lat") != len(lat) or cached_da.sizes.get("lon") != len(lon):
        return False
    cached_lat = np.asarray(cached_da["lat"].values)
    cached_lon = np.asarray(cached_da["lon"].values)
    return bool(
        np.allclose([cached_lat[0], cached_lat[-1]], [lat[0], lat[-1]], atol=1e-6)
        and np.allclose([cached_lon[0], cached_lon[-1]], [lon[0], lon[-1]], atol=1e-6)
    )


class CreateWmoRegionMasks:
    """Create per-region masks for the WMO regions used in global mHM/mRM setups.

    Parameters
    ----------
    output_dir : str
        Directory for masks, region rasters, and the overview plot.
    mask_file : str
        Reference NetCDF defining the target grid and valid land cells.
    mask_var : str, optional
        Mask variable; auto-detected among `mask`/`land_mask`/`mask_l2` if omitted.
    basin_id_file : str, optional
        NetCDF with global unique basin ids. Required unless a flow direction
        file is passed to `get_global_basin_ids`/`create_region_masks`.
    basin_var : str, optional
        Basin id variable name, by default "basin".
    region_var : str, optional
        Region variable name used for cached/output rasters.
    output_file_name : str, optional
        Stem for the per-region mask files.
    exclude_polygon : str, optional
        `"greenland"` (default), a shapefile/GeoJSON path, or `"none"`.
    min_shape_overlap_fraction, max_island_cells_fraction, min_surround_fraction, strict_anchors :
        See `create_basin_region_assignment`.
    recreate_basin_ids, recreate_region_rasters, recreate_basin_mapping : bool, optional
        Force recomputation of the matching cached artifact.
    """

    def __init__(
        self,
        output_dir,
        mask_file,
        mask_var=None,
        basin_id_file=None,
        basin_var="basin",
        region_var=REGION_VAR,
        output_file_name="wmo_region_mask",
        exclude_polygon="greenland",
        min_shape_overlap_fraction=0.5,
        max_island_cells_fraction=0.001,
        min_surround_fraction=0.8,
        strict_anchors=False,
        recreate_basin_ids=False,
        recreate_region_rasters=False,
        recreate_basin_mapping=False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mask_file = mask_file
        self.mask_var = mask_var
        self.basin_id_file = Path(basin_id_file) if basin_id_file is not None else None
        self.basin_var = basin_var
        self.region_var = region_var
        self.output_file_name = output_file_name
        self.exclude_polygon = exclude_polygon
        self.min_shape_overlap_fraction = min_shape_overlap_fraction
        self.max_island_cells_fraction = max_island_cells_fraction
        self.min_surround_fraction = min_surround_fraction
        self.strict_anchors = strict_anchors
        self.recreate_basin_ids = recreate_basin_ids
        self.recreate_region_rasters = recreate_region_rasters
        self.recreate_basin_mapping = recreate_basin_mapping
        self.lat = None
        self.lon = None
        self.land_mask = None
        self.transform = None

    def read_target_grid(self):
        """Read the reference mask file and set the shared lat/lon/land_mask/transform state.

        Excluded cells (e.g. Greenland) are removed from the land mask here,
        before any region rasterization or basin voting, so they can never
        seed or bias a region; `remove_excluded_area` later applies the same
        exclusion to the final output raster and plot.
        """
        from mhm_tools.pre.catchment import _as_affine, get_transformation_matrix_nc

        mask_ds = get_xarray_ds_from_file(
            self.mask_file, normalize_latlon_coords=True, force_decending_y=True
        )
        mask_var = self.mask_var
        if mask_var is None:
            mask_var = get_single_data_var(
                mask_ds, proposed_vars=["mask", "land_mask", "mask_l2"]
            )
        if mask_var is None or mask_var not in mask_ds:
            msg = f"Could not determine mask variable in {self.mask_file}; pass mask_var explicitly."
            with ErrorLogger(logger):
                raise ValueError(msg)
        mask_da = mask_ds[mask_var]
        self.lat = np.asarray(mask_da["lat"].values)
        self.lon = np.asarray(mask_da["lon"].values)
        self.land_mask = np.asarray(
            create_valid_data_mask(mask_da, treat_zero_as_missing=True).values
        )
        self.transform = _as_affine(get_transformation_matrix_nc(mask_ds, mask_var))

        exclude_mask = create_exclude_mask(mask_da, self.exclude_polygon)
        if exclude_mask is not None:
            exclude_mask = np.asarray(exclude_mask)
            self.land_mask = self.land_mask & ~exclude_mask
            logger.info(
                f"Excluded {int(exclude_mask.sum())} cell(s) from the land mask "
                f"({self.exclude_polygon})."
            )
        logger.info(
            f"Target grid: {self.lat.size} x {self.lon.size} cells, "
            f"{int(self.land_mask.sum())} valid land cell(s)."
        )

    def get_global_basin_ids(self, fdir_file=None, **catchment_kwargs):
        """Return global basin ids remapped onto the target grid, delineating them if needed.

        Reads the basin file raw (`mask_and_scale=False`) so exact integer ids
        survive without being decoded into float NaN.
        """
        if self.basin_id_file is None:
            if fdir_file is None:
                msg = "Either basin_id_file or fdir_file must be provided to obtain global basins."
                with ErrorLogger(logger):
                    raise ValueError(msg)
            from mhm_tools.pre.catchment import create_catchment

            basin_dir = self.output_dir / "basin_ids"
            basin_file = basin_dir / "basin_ids.nc"
            if basin_file.is_file() and not self.recreate_basin_ids:
                logger.info(f"Using cached global basin ids {basin_file}.")
            else:
                create_catchment(
                    input_file=fdir_file,
                    output_path=basin_dir,
                    output_vars="basin",
                    upscale=True,
                    **catchment_kwargs,
                )
            self.basin_id_file = basin_file
        with xr.open_dataset(
            self.basin_id_file, engine="netcdf4", mask_and_scale=False
        ) as basin_ds:
            basin_da = basin_ds[self.basin_var].load()
        fill_value = basin_da.attrs.get(
            "_FillValue", basin_da.encoding.get("_FillValue", 0)
        )
        remapped = basin_da.sel(lat=self.lat, lon=self.lon, method="nearest")
        basin_ids = np.where(
            np.asarray(remapped.values) == fill_value, 0, remapped.values
        ).astype(np.int64)
        logger.info(
            f"Loaded {np.unique(basin_ids[basin_ids != 0]).size} unique basin id(s) on the target grid."
        )
        return basin_ids

    def read_cached_region_grid(self, file_path, *, recreate=False):
        """Return a cached region grid if present and matching the target grid, else None."""
        file_path = Path(file_path)
        if recreate or not file_path.is_file():
            return None
        with get_xarray_ds_from_file(file_path) as cached_ds:
            if self.region_var not in cached_ds:
                logger.warning(
                    f"Cached file {file_path} has no variable {self.region_var!r}; recreating."
                )
                return None
            cached = cached_ds[self.region_var].astype(np.float32).load()
        if not _grid_matches(cached, self.lat, self.lon):
            logger.warning(
                f"Cached file {file_path} grid does not match the target grid; recreating."
            )
            return None
        logger.info(f"Using cached region grid {file_path}.")
        return cached

    def write_region_grid(self, region_grid, file_path, long_name):
        """Write a region grid to NetCDF, converting NaN=unset to the on-disk 0=nodata convention."""
        from mhm_tools.common.constants import NC_ENCODE_MASK
        from mhm_tools.common.file_handler import write_xarray_to_file

        file_path = Path(file_path)
        on_disk = region_grid.fillna(0).astype("uint8")
        on_disk.name = self.region_var
        on_disk.attrs["long_name"] = long_name
        encoding = {
            self.region_var: {
                "dtype": "uint8",
                "zlib": True,
                "complevel": 4,
                **NC_ENCODE_MASK,
            }
        }
        write_xarray_to_file(on_disk.to_dataset(), file_path, encoding=encoding)
        return file_path

    def create_region_grid_from_shapes(self, shape_regions, dissolve_batch_size=200):
        """Rasterize per-region dissolved shapefiles onto the target grid (cached)."""
        cache_file = self.output_dir / "wmo_regions_shapes_unfilled.nc"
        cached = self.read_cached_region_grid(
            cache_file, recreate=self.recreate_region_rasters
        )
        if cached is not None:
            return cached
        unfilled = rasterize_region_geometries(
            shape_regions,
            self.lat,
            self.lon,
            self.land_mask,
            self.transform,
            dissolve_batch_size=dissolve_batch_size,
        )
        self.write_region_grid(
            unfilled, cache_file, "WMO region from shapefiles (unfilled)"
        )
        return unfilled

    def create_region_grid_from_geojson(self, geojson_file, region_property="WMO_RA"):
        """Rasterize a WMO-region GeoJSON onto the target grid (cached)."""
        cache_file = self.output_dir / "wmo_regions_geojson.nc"
        cached = self.read_cached_region_grid(
            cache_file, recreate=self.recreate_region_rasters
        )
        if cached is not None:
            return cached
        geojson_grid = create_region_grid_from_geojson_file(
            geojson_file,
            self.lat,
            self.lon,
            self.land_mask,
            region_property=region_property,
        )
        self.write_region_grid(geojson_grid, cache_file, "WMO region from GeoJSON")
        return geojson_grid

    def create_combined_region_grid(self, shapes_unfilled_grid, geojson_grid):
        """Combine shapes (priority) and geojson region grids, fill remaining gaps (cached)."""
        cache_file = self.output_dir / "wmo_regions_combined.nc"
        cached = self.read_cached_region_grid(
            cache_file, recreate=self.recreate_region_rasters
        )
        if cached is not None:
            return cached
        combined = combine_region_grids(
            priority_grid=shapes_unfilled_grid, fallback_grid=geojson_grid
        )
        combined = fill_region_grid_gaps(combined, self.land_mask)
        self.write_region_grid(
            combined,
            cache_file,
            "Combined WMO region (shapes priority, geojson fallback, nearest-neighbour filled)",
        )
        return combined

    def map_basins_to_regions(self, basin_ids, combined_region_grid, shape_regions):
        """Map every global basin to exactly one region, prioritising anchor shapefiles (cached)."""
        map_cache_file = self.output_dir / "basin_region_map.csv"
        region_cache_file = self.output_dir / "wmo_regions_basins.nc"
        if not self.recreate_basin_mapping and map_cache_file.is_file():
            cached_region_grid = self.read_cached_region_grid(region_cache_file)
            if cached_region_grid is not None:
                logger.info(f"Using cached basin-to-region mapping {map_cache_file}.")
                return cached_region_grid

        region_raster = np.nan_to_num(combined_region_grid.values, nan=0).astype(
            np.int64
        )
        final_raster, basin_region_map, diagnostics = create_basin_region_assignment(
            basin_ids,
            region_raster,
            shape_regions,
            self.transform,
            min_shape_overlap_fraction=self.min_shape_overlap_fraction,
            strict_anchors=self.strict_anchors,
            max_island_cells_fraction=self.max_island_cells_fraction,
            min_surround_fraction=self.min_surround_fraction,
        )
        logger.info(f"Basin-to-region mapping diagnostics: {diagnostics}")
        self._write_basin_region_map_csv(basin_region_map, map_cache_file)
        final_grid = xr.DataArray(
            np.where(final_raster == 0, np.nan, final_raster).astype(np.float32),
            coords={"lat": self.lat, "lon": self.lon},
            dims=["lat", "lon"],
            name=self.region_var,
        )
        self.write_region_grid(
            final_grid, region_cache_file, "Final WMO region per basin"
        )
        return final_grid

    @staticmethod
    def _write_basin_region_map_csv(basin_region_map, file_path):
        """Write the basin_id -> region_id mapping to a CSV file."""
        with Path(file_path).open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["basin_id", "region_id"])
            for basin_id, region_id in sorted(basin_region_map.items()):
                writer.writerow([basin_id, region_id])

    def remove_excluded_area(self, region_grid):
        """Set excluded-polygon cells (e.g. Greenland) to nodata in the final region grid.

        `read_target_grid` already removed these cells from the land mask
        before any region rasterization or basin voting; this re-applies the
        same exclusion to the grid being written out, as its own visible step.
        """
        exclude_mask = create_exclude_mask(region_grid, self.exclude_polygon)
        if exclude_mask is None:
            return region_grid
        return region_grid.where(~exclude_mask)

    def write_region_masks(self, region_grid):
        """Write one boolean mask NetCDF per WMO region."""
        written = []
        region_values = region_grid.values
        for region_id, region_name in WMO_REGIONS.items():
            sub_mask = (region_values == region_id) & self.land_mask
            if not np.any(sub_mask):
                logger.warning(
                    f"Region {region_id} ({region_name}) has no cells; no mask written."
                )
                continue
            file_path = self.output_dir / f"{self.output_file_name}_{region_id:02d}.nc"
            write_mask_to_file(
                sub_mask,
                self.lat,
                self.lon,
                file_path,
                long_name=f"WMO region {region_id} ({region_name}) mask",
            )
            written.append(file_path)
        logger.info(f"Wrote {len(written)} region mask(s) to {self.output_dir}.")
        return written

    def plot_region_overview(
        self, shapes_grid, geojson_grid, combined_grid, final_grid, plot_file=None
    ):
        """Plot the 4-panel WMO region overview: shapes, geojson, combined, final."""
        plot_file = (
            Path(plot_file)
            if plot_file is not None
            else self.output_dir / "wmo_regions_overview.png"
        )
        plot_categorical_map_panels(
            grids=[shapes_grid, geojson_grid, combined_grid, final_grid],
            titles=[
                "WMO regions from shapefiles",
                "WMO regions from GeoJSON",
                "Combined regions (shapes priority)",
                "Final regions per basin",
            ],
            category_colors=WMO_REGION_COLORS,
            category_names=WMO_REGIONS,
            out_path=plot_file,
            ncols=2,
        )
        return plot_file

    def create_region_masks(
        self,
        fdir_file=None,
        shape_dir=None,
        geojson_file=None,
        region_raster_file=None,
        region_property="WMO_RA",
        shape_region_regex=SHAPE_REGION_REGEX,
        dissolve_batch_size=200,
        map_basins=True,
        create_plot=True,
        plot_file=None,
        **catchment_kwargs,
    ):
        """Run the full WMO-region-mask pipeline: basins, regions, mapping, masks, plot.

        Returns
        -------
        dict
            `masks` (written file paths), `plot` (overview plot path or None),
            `final_region_grid`, and `basin_ids`.
        """
        self.read_target_grid()
        basin_ids = self.get_global_basin_ids(fdir_file=fdir_file, **catchment_kwargs)

        shapes_grid = geojson_grid = combined_grid = None
        shape_regions = []
        if region_raster_file is not None:
            logger.info(
                f"Using pre-computed region raster {region_raster_file}; skipping "
                "shapefile/GeoJSON rasterization."
            )
            with get_xarray_ds_from_file(region_raster_file) as region_ds:
                region_var = (
                    self.region_var
                    if self.region_var in region_ds
                    else get_single_data_var(region_ds)
                )
                combined_grid = region_ds[region_var].astype(np.float32).load()
        else:
            if shape_dir is None or geojson_file is None:
                msg = "Either region_raster_file, or both shape_dir and geojson_file, must be provided."
                with ErrorLogger(logger):
                    raise ValueError(msg)
            shape_regions = read_shape_file_regions(
                shape_dir, region_regex=shape_region_regex
            )
            shapes_grid = self.create_region_grid_from_shapes(
                shape_regions, dissolve_batch_size=dissolve_batch_size
            )
            geojson_grid = self.create_region_grid_from_geojson(
                geojson_file, region_property=region_property
            )
            combined_grid = self.create_combined_region_grid(shapes_grid, geojson_grid)

        final_grid = (
            self.map_basins_to_regions(basin_ids, combined_grid, shape_regions)
            if map_basins
            else combined_grid
        )
        final_grid = self.remove_excluded_area(final_grid)
        written_masks = self.write_region_masks(final_grid)

        plot_path = None
        if create_plot:
            plot_path = self.plot_region_overview(
                shapes_grid,
                geojson_grid,
                combined_grid,
                final_grid,
                plot_file=plot_file,
            )

        return {
            "masks": written_masks,
            "plot": plot_path,
            "final_region_grid": final_grid,
            "basin_ids": basin_ids,
        }


@log_arguments()
def create_wmo_region_masks(  # noqa: PLR0913
    output_dir,
    mask_file,
    mask_var=None,
    basin_id_file=None,
    basin_var="basin",
    fdir_file=None,
    shape_dir=None,
    shape_region_regex=SHAPE_REGION_REGEX,
    region_geojson=None,
    region_property="WMO_RA",
    region_raster_file=None,
    region_var=REGION_VAR,
    output_file_name="wmo_region_mask",
    exclude_polygon="greenland",
    min_shape_overlap_fraction=0.5,
    max_island_cells_fraction=0.001,
    min_surround_fraction=0.8,
    strict_anchors=False,
    dissolve_batch_size=200,
    map_basins=True,
    create_plot=True,
    plot_file=None,
    recreate_basin_ids=False,
    recreate_region_rasters=False,
    recreate_basin_mapping=False,
    **catchment_kwargs,
):
    """Create per-region WMO masks from global basins, shapefiles, and a GeoJSON.

    Delineates (or reuses) global basins, rasterizes WMO regions from a
    shapefile set and a GeoJSON, maps every basin to exactly one region while
    prioritising basins covered by the original shapefiles and enforcing
    region contiguity, excludes Greenland (or another polygon), and writes
    one mask per region plus a 4-panel overview plot.

    Args:
        output_dir (str): Directory for masks, region rasters, and the overview plot.
        mask_file (str): Reference NetCDF defining the target grid and valid land cells.
        mask_var (str, optional): Mask variable; auto-detected if omitted.
        basin_id_file (str, optional): NetCDF with global unique basin ids. Required
            unless fdir_file is given.
        basin_var (str): Basin id variable name.
        fdir_file (str, optional): Flow direction file; delineates global basins via
            create_catchment first when basin_id_file is not given.
        shape_dir (str, optional): Directory of per-basin shapefiles. Required unless
            region_raster_file is given.
        shape_region_regex (str): Regex whose first captured digit is the region id.
        region_geojson (str, optional): WMO region GeoJSON. Required unless
            region_raster_file is given.
        region_property (str): Integer region property in the GeoJSON.
        region_raster_file (str, optional): Pre-computed combined region raster;
            skips shapefile/GeoJSON rasterization.
        region_var (str): Region variable name used for cached/output rasters.
        output_file_name (str): Stem for the per-region mask files.
        exclude_polygon (str): "greenland" (default), a shapefile/GeoJSON path, or "none".
        min_shape_overlap_fraction (float): Anchor acceptance threshold.
        max_island_cells_fraction (float): Cleanup-pass size threshold, as a
            fraction of total land cells.
        min_surround_fraction (float): Cleanup-pass "mostly surrounded" threshold.
        strict_anchors (bool): Raise instead of warning on shapefile region conflicts.
        dissolve_batch_size (int): Shapefiles read per dissolve batch.
        map_basins (bool): Map basins to regions (step 3); if False, masks are
            written straight from the combined region raster.
        create_plot (bool): Create the 4-panel overview plot.
        plot_file (str, optional): Overview plot path.
        recreate_basin_ids (bool): Re-delineate global basins instead of reusing the cache.
        recreate_region_rasters (bool): Rebuild shape/geojson/combined rasters; cascades
            to also redo the basin mapping.
        recreate_basin_mapping (bool): Redo the basin-to-region mapping.
        **catchment_kwargs: Passed through to create_catchment when fdir_file is given.

    Returns
    -------
        dict: Written mask files, the overview plot path, the final region grid,
        and the basin id raster used.
    """
    if recreate_region_rasters:
        recreate_basin_mapping = True
    csm = CreateWmoRegionMasks(
        output_dir=output_dir,
        mask_file=mask_file,
        mask_var=mask_var,
        basin_id_file=basin_id_file,
        basin_var=basin_var,
        region_var=region_var,
        output_file_name=output_file_name,
        exclude_polygon=exclude_polygon,
        min_shape_overlap_fraction=min_shape_overlap_fraction,
        max_island_cells_fraction=max_island_cells_fraction,
        min_surround_fraction=min_surround_fraction,
        strict_anchors=strict_anchors,
        recreate_basin_ids=recreate_basin_ids,
        recreate_region_rasters=recreate_region_rasters,
        recreate_basin_mapping=recreate_basin_mapping,
    )
    return csm.create_region_masks(
        fdir_file=fdir_file,
        shape_dir=shape_dir,
        geojson_file=region_geojson,
        region_raster_file=region_raster_file,
        region_property=region_property,
        shape_region_regex=shape_region_regex,
        dissolve_batch_size=dissolve_batch_size,
        map_basins=map_basins,
        create_plot=create_plot,
        plot_file=plot_file,
        **catchment_kwargs,
    )
