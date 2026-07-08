from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box

from mhm_tools.common import catchment_maps
from mhm_tools.post import metric_plots


def test_discharge_shape_folder_matches_ids_and_aggregates_median(tmp_path):
    """Match gauge IDs to basin shapefiles and aggregate repeated rows."""
    shape_dir = tmp_path / "shapes"
    shape_dir.mkdir()
    gpd.GeoDataFrame(
        {"name": ["one"], "geometry": [box(0, 0, 1, 1)]},
        geometry="geometry",
        crs="EPSG:4326",
    ).to_file(shape_dir / "basin_1.shp")
    gpd.GeoDataFrame(
        {"name": ["two"], "geometry": [box(2, 0, 4, 1)]},
        geometry="geometry",
        crs="EPSG:4326",
    ).to_file(shape_dir / "basin_2.shp")
    metric_df = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "kge": [0.2, 0.8, 0.5],
            "nse": [0.1, 0.3, 0.7],
        }
    )
    median_rows = catchment_maps.calculate_metric_medians(
        metric_df, variables=["kge", "nse"]
    )

    metric_gdf = catchment_maps.create_catchment_metric_geodataframe(
        median_rows,
        shape_folder=shape_dir,
    )

    assert set(metric_gdf["id"].astype(int)) == {1, 2}
    assert np.isclose(metric_gdf.loc[metric_gdf["id"] == 1, "kge"].iloc[0], 0.5)


def test_standalone_csv_explicit_shape_paths_are_matched_by_input_order(
    tmp_path, monkeypatch
):
    """Map one CSV input to one explicit shape path."""
    csv_file = tmp_path / "catchment.csv"
    shape_file = tmp_path / "catchment.shp"
    pd.DataFrame({"kge": [0.2, 0.8]}).to_csv(csv_file, index=False)
    calls = []

    def fake_write_catchment_median_maps(**kwargs):
        """Capture catchment map writer arguments."""
        calls.append(kwargs)
        return [Path(kwargs["output_dir"]) / "catchment_map_kge.png"]

    monkeypatch.setattr(
        metric_plots,
        "write_catchment_median_maps",
        fake_write_catchment_median_maps,
    )

    output_files = metric_plots.write_metric_catchment_maps(
        input_paths=[str(csv_file)],
        variables=["kge"],
        output_dir=tmp_path / "plots",
        shape_paths=[str(shape_file)],
    )

    assert output_files == [tmp_path / "plots" / "catchment_map_kge.png"]
    assert calls[0]["shape_files_by_id"] == {"catchment": shape_file}
    assert np.isclose(calls[0]["metric_df"].loc[0, "kge"], 0.5)


def test_standalone_csv_id_column_uses_folder_matching(tmp_path, monkeypatch):
    """Keep per-ID rows when a standalone CSV contains an id column."""
    csv_file = tmp_path / "metrics.csv"
    shape_dir = tmp_path / "shapes"
    shape_dir.mkdir()
    pd.DataFrame({"id": [10, 10, 11], "kge": [0.1, 0.5, 0.9]}).to_csv(
        csv_file, index=False
    )
    calls = []

    def fake_write_catchment_median_maps(**kwargs):
        """Capture catchment map writer arguments."""
        calls.append(kwargs)
        return []

    monkeypatch.setattr(
        metric_plots,
        "write_catchment_median_maps",
        fake_write_catchment_median_maps,
    )

    metric_plots.write_metric_catchment_maps(
        input_paths=[str(csv_file)],
        variables=["kge"],
        output_dir=tmp_path / "plots",
        shape_folder=shape_dir,
    )

    assert calls[0]["shape_folder"] == shape_dir
    assert calls[0]["metric_df"]["id"].tolist() == [10, 11]
    assert np.isclose(calls[0]["metric_df"].loc[0, "kge"], 0.3)


def test_mask_file_vectorizes_to_one_geometry(tmp_path):
    """Vectorize one binary mask file to a dissolved geometry."""
    mask_file = tmp_path / "basin_1.nc"
    mask = xr.Dataset(
        {
            "mask": (
                ("lat", "lon"),
                np.array([[1, 1], [0, 1]], dtype=np.int32),
            )
        },
        coords={"lat": [1.0, 0.0], "lon": [0.0, 1.0]},
    )
    mask.to_netcdf(mask_file)

    mask_gdf = catchment_maps.read_mask_geometry(
        mask_file, mask_var="mask", geometry_id=1
    )

    assert len(mask_gdf) == 1
    assert not mask_gdf.geometry.iloc[0].is_empty


def test_geometries_are_sorted_largest_first():
    """Sort overlapping geometries so small catchments are plotted last."""
    gdf = gpd.GeoDataFrame(
        {
            "id": ["small", "large"],
            "kge": [0.1, 0.9],
            "geometry": [box(0, 0, 1, 1), box(0, 0, 3, 3)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    sorted_gdf = catchment_maps.sort_geodataframe_by_area_desc(gdf)

    assert sorted_gdf["id"].tolist() == ["large", "small"]
