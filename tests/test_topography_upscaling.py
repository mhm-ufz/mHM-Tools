"""Tests for the topography-upscaling command."""

import argparse
import logging

import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr

from mhm_tools._cli import _topography_upscaling
from mhm_tools.common.constants import NO_DATA
from mhm_tools.pre import topography_upscaling
from mhm_tools.pre.latlon import _create_grid
from mhm_tools.pre.topography_upscaling import upscale_topography

CELLSIZE = 100.0
WEST = 1000.0
NORTH = 2000.0
GRID_VARIABLES = {"lat", "lon", "crs"}


@pytest.fixture
def _mhm_tools_logger_propagates():
    """Let caplog see the mhm_tools warnings that other tests may silence."""
    logger = logging.getLogger("mhm_tools")
    previous_propagate, previous_level = logger.propagate, logger.level
    logger.propagate = True
    logger.setLevel(logging.NOTSET)
    yield
    logger.propagate = previous_propagate
    logger.setLevel(previous_level)


def _write_grid(path, values, *, west=WEST, units=None, nodata=None):
    """Write a north-up grid of 100 m cells in EPSG:32632 to NetCDF.

    Args:
        path (pathlib.Path): Output file.
        values (array-like): Grid values, row 0 is the northernmost.
        west (float): West edge of the grid in metres.
        units (str): Units attribute of the variable, or None.
        nodata (float): Fill value declared for the variable, or None.

    Returns
    -------
        pathlib.Path: The written file.
    """
    values = np.asarray(values)
    rows, cols = values.shape
    data = xr.DataArray(
        values,
        dims=("y", "x"),
        coords={
            "y": NORTH - CELLSIZE * (np.arange(rows) + 0.5),
            "x": west + CELLSIZE * (np.arange(cols) + 0.5),
        },
        name="data",
        attrs={} if units is None else {"units": units},
    ).rio.write_crs("EPSG:32632")
    if nodata is not None:
        # an encoding argument of to_netcdf would replace the whole encoding,
        # and with it the grid_mapping that links the variable to its CRS
        data.encoding["_FillValue"] = nodata
    data.to_dataset().to_netcdf(path)
    return path


def _create_plane(rows, cols):
    """Create a DEM that falls towards the east and rises towards the south.

    Args:
        rows (int): Number of rows.
        cols (int): Number of columns.

    Returns
    -------
        numpy.ndarray: Elevation, row 0 is the northernmost.
    """
    return 50.0 - np.arange(cols, dtype="float64") + np.arange(rows)[:, None]


def _read_output(path):
    """Read the upscaled variables of an output NetCDF.

    Args:
        path (pathlib.Path): Output file of upscale_topography.

    Returns
    -------
        dict: numpy arrays by variable name, NaN in missing cells.
    """
    with xr.open_dataset(path) as ds:
        return {
            name: ds[name].values for name in ds.data_vars if name not in GRID_VARIABLES
        }


def _write_west_draining_grids(tmp_path):
    """Write a 6 x 6 grid draining west with one outlet in the middle of a block edge.

    Every cell drains west, except cell (1, 3) in the middle of the west edge of
    the north-eastern block of 3 x 3 cells, which drains south-west. It also has
    the largest flow accumulation of its block.

    Args:
        tmp_path (pathlib.Path): Directory for the files.

    Returns
    -------
        tuple: The flow direction and the flow accumulation file.
    """
    codes = np.full((6, 6), 16, dtype="uint8")
    codes[1, 3] = 8
    facc = np.ones((6, 6))
    facc[1, 3] = 10.0
    return (
        _write_grid(tmp_path / "fdir.nc", codes),
        _write_grid(tmp_path / "facc.nc", facc, units="km2"),
    )


def test_dem_is_averaged_over_the_valid_cells_of_each_block(tmp_path):
    """Average the valid cells of each block, and leave empty blocks missing."""
    dem = np.array(
        [
            [10.0, 12.0, 20.0, 22.0],
            [14.0, 16.0, 24.0, NO_DATA],
            [NO_DATA, NO_DATA, 30.0, 32.0],
            [NO_DATA, NO_DATA, 34.0, 36.0],
        ]
    )
    dem_file = _write_grid(tmp_path / "dem.nc", dem, nodata=NO_DATA)

    (output,) = upscale_topography(
        tmp_path / "out.nc", dem_file=dem_file, upscale_factor=2
    )

    result = _read_output(output)
    np.testing.assert_allclose(result["dem"], [[13.0, 22.0], [np.nan, 33.0]])
    for name in ("slope", "aspect", "fdir"):
        assert np.isnan(result[name][1, 0]), name


def test_depression_fill_dem_upscales_the_filled_dem(tmp_path):
    """Upscale the input DEM, or with --depression-fill-dem the filled one."""
    dem = np.tile([40.0, 30.0, 20.0, 10.0], (4, 1))
    dem[1, 1] = 5.0  # a pit that fills up to its spill level of 20
    dem_file = _write_grid(tmp_path / "dem.nc", dem)

    (as_given,) = upscale_topography(
        tmp_path / "given.nc", dem_file=dem_file, upscale_factor=2
    )
    (filled,) = upscale_topography(
        tmp_path / "filled.nc",
        dem_file=dem_file,
        upscale_factor=2,
        depression_fill_dem=True,
    )

    assert _read_output(as_given)["dem"][0, 0] == pytest.approx(28.75)
    assert _read_output(filled)["dem"][0, 0] == pytest.approx(32.5)


def test_outlet_drains_into_the_coarse_cell_that_receives_its_water(tmp_path):
    """Point a coarse cell where its outlet drains to, not along its fine code."""
    fdir_file, facc_file = _write_west_draining_grids(tmp_path)

    (output,) = upscale_topography(
        tmp_path / "out.nc",
        fdir_file=fdir_file,
        facc_file=facc_file,
        upscale_factor=3,
        write_facc=True,
    )

    result = _read_output(output)
    # the south-west outlet in the middle of the west edge enters the western
    # coarse cell, so its coarse cell drains west (16); the western cells drain
    # off the grid edge and become sinks (0)
    np.testing.assert_array_equal(result["fdir"], [[0, 16], [0, 16]])
    np.testing.assert_array_equal(result["facc"], [[1, 10], [1, 1]])
    with xr.open_dataset(output) as ds:
        assert ds["facc"].attrs["units"] == "km2"


@pytest.mark.parametrize(
    ("codes", "facc", "expected"),
    [
        # a pit wins a tie against a cell that leaves the block
        ([[64, 0], [1, 64]], [[5.0, 5.0], [1.0, 2.0]], 0),
        # among cells leaving the block the northernmost wins, then the westernmost
        ([[1, 1], [4, 16]], [[1.0, 3.0], [3.0, 1.0]], 1),
    ],
)
def test_ties_go_to_the_pit_then_to_the_north_west(tmp_path, codes, facc, expected):
    """Break ties like mHM: pits first, then the first cell in row order."""
    fdir_file = _write_grid(tmp_path / "fdir.nc", np.asarray(codes, dtype="uint8"))
    facc_file = _write_grid(tmp_path / "facc.nc", facc)

    (output,) = upscale_topography(
        tmp_path / "out.nc",
        fdir_file=fdir_file,
        facc_file=facc_file,
        upscale_factor=2,
        keep_edge_outflow=True,
    )

    assert _read_output(output)["fdir"][0, 0] == expected


def test_keep_edge_outflow_keeps_directions_off_the_grid_edge(tmp_path):
    """Keep the direction of edge outlets, but not of outlets draining into nodata."""
    fdir_file, facc_file = _write_west_draining_grids(tmp_path)
    (output,) = upscale_topography(
        tmp_path / "out.nc",
        fdir_file=fdir_file,
        facc_file=facc_file,
        upscale_factor=3,
        keep_edge_outflow=True,
    )
    np.testing.assert_array_equal(_read_output(output)["fdir"], [[16, 16], [16, 16]])

    codes = np.full((2, 4), 16, dtype="uint8")
    codes[:, :2] = 247  # the western block has no data
    facc = np.ones((2, 4))
    facc[:, :2] = np.nan
    (output,) = upscale_topography(
        tmp_path / "nodata.nc",
        fdir_file=_write_grid(tmp_path / "fdir_nodata.nc", codes),
        facc_file=_write_grid(tmp_path / "facc_nodata.nc", facc),
        upscale_factor=2,
        keep_edge_outflow=True,
    )
    np.testing.assert_array_equal(_read_output(output)["fdir"], [[np.nan, 0]])


def test_tiles_with_keep_edge_outflow_mosaic_to_the_full_grid(tmp_path):
    """Mosaic two tiles to the full grid, also with a pit right across the seam."""
    codes = np.full((6, 6), 16, dtype="uint8")
    codes[1, 2] = 0  # a pit just west of the seam
    facc = np.ones((6, 6))
    facc[1, 2] = 11.0
    facc[1, 3] = 10.0  # drains west across the seam into the pit
    results = {}
    for name, columns, west in (
        ("full", slice(0, 6), WEST),
        ("west", slice(0, 3), WEST),
        ("east", slice(3, 6), WEST + 3 * CELLSIZE),
    ):
        (output,) = upscale_topography(
            tmp_path / f"{name}.nc",
            fdir_file=_write_grid(
                tmp_path / f"fdir_{name}.nc", codes[:, columns], west=west
            ),
            facc_file=_write_grid(
                tmp_path / f"facc_{name}.nc", facc[:, columns], west=west
            ),
            upscale_factor=3,
            keep_edge_outflow=True,
        )
        results[name] = _read_output(output)["fdir"]

    np.testing.assert_array_equal(results["full"], [[0, 16], [16, 16]])
    np.testing.assert_array_equal(
        np.hstack([results["west"], results["east"]]), results["full"]
    )


def test_computed_facc_is_written_in_km2_by_default_and_in_cells_on_request(
    tmp_path,
):
    """Write the facc of the outlets only with write_facc, in km2 or cells."""
    # every cell drains east, the eastern column off the grid edge
    fdir_file = _write_grid(tmp_path / "fdir.nc", np.ones((2, 4), dtype="uint8"))

    (km2_file,) = upscale_topography(
        tmp_path / "km2.nc", fdir_file=fdir_file, upscale_factor=2, write_facc=True
    )
    (cell_file,) = upscale_topography(
        tmp_path / "cell.nc",
        fdir_file=fdir_file,
        upscale_factor=2,
        write_facc=True,
        facc_unit="cell",
    )
    (plain_file,) = upscale_topography(
        tmp_path / "plain.nc", fdir_file=fdir_file, upscale_factor=2
    )

    km2 = _read_output(km2_file)
    np.testing.assert_array_equal(km2["fdir"], [[1, 0]])
    # a 100 m cell covers 0.01 km2
    np.testing.assert_allclose(km2["facc"], [[0.02, 0.04]])
    np.testing.assert_array_equal(_read_output(cell_file)["facc"], [[2, 4]])
    assert "facc" not in _read_output(plain_file)


@pytest.mark.parametrize(
    ("input_names", "expected"),
    [
        (("dem",), {"dem", "slope", "aspect", "fdir"}),
        (("fdir",), {"fdir"}),
        (("dem", "fdir"), {"dem", "slope", "aspect", "fdir"}),
    ],
)
def test_input_modes_write_their_variables(tmp_path, input_names, expected):
    """Write dem, slope and aspect only for a DEM, and fdir in every mode."""
    files = {
        "dem_file": _write_grid(tmp_path / "dem.nc", _create_plane(4, 6)),
        "fdir_file": _write_grid(tmp_path / "fdir.nc", np.ones((4, 6), dtype="uint8")),
    }
    inputs = {f"{name}_file": files[f"{name}_file"] for name in input_names}

    (output,) = upscale_topography(tmp_path / "out.nc", upscale_factor=2, **inputs)

    assert set(_read_output(output)) == expected


def test_coarse_grid_keeps_the_lower_left_corner(tmp_path):
    """Keep the lower-left corner in the ASCII header and the NetCDF coordinates."""
    dem_file = _write_grid(tmp_path / "dem.nc", _create_plane(4, 6))
    coarse_header = {
        "ncols": 3,
        "nrows": 2,
        "xllcorner": WEST,
        "yllcorner": NORTH - 4 * CELLSIZE,
        "cellsize": 2 * CELLSIZE,
    }

    written = upscale_topography(
        tmp_path / "asc", dem_file=dem_file, upscale_factor=2, output_extension="asc"
    )
    (nc_output,) = upscale_topography(
        tmp_path / "out.nc", dem_file=dem_file, upscale_factor=2
    )

    assert sorted(path.name for path in written) == [
        "aspect.asc",
        "dem.asc",
        "fdir.asc",
        "slope.asc",
    ]
    slope_file = next(path for path in written if path.stem == "slope")
    header = dict(line.split() for line in slope_file.read_text().splitlines()[:6])
    for key, value in coarse_header.items():
        assert float(header[key]) == pytest.approx(value), key
    x, y, _, _ = _create_grid(coarse_header, dtype="f8")
    with xr.open_dataset(nc_output) as ds:
        np.testing.assert_allclose(ds["x"].values, x)
        np.testing.assert_allclose(ds["y"].values, y)


def test_chunked_run_matches_the_in_memory_run(tmp_path):
    """Give the same result in chunks of one block as with the grids in memory."""
    rng = np.random.default_rng(0)
    dem = 100.0 + 10.0 * rng.random((6, 6))
    dem[0, 0] = NO_DATA
    codes = rng.choice(
        np.array([0, 1, 2, 4, 8, 16, 32, 64, 128], dtype="uint8"), size=(6, 6)
    )
    codes[5, 5] = 247
    facc = 100.0 * rng.random((6, 6))
    facc[5, 5] = np.nan
    files = {
        "dem_file": _write_grid(tmp_path / "dem.nc", dem, nodata=NO_DATA),
        "fdir_file": _write_grid(tmp_path / "fdir.nc", codes),
        "facc_file": _write_grid(tmp_path / "facc.nc", facc, units="km2"),
    }

    (in_memory,) = upscale_topography(
        tmp_path / "memory.nc", upscale_factor=2, write_facc=True, **files
    )
    # this budget makes every chunk a single block of 2 x 2 cells
    (chunked,) = upscale_topography(
        tmp_path / "chunked.nc",
        upscale_factor=2,
        write_facc=True,
        available_mem_gib=1e-6,
        **files,
    )

    in_memory_result = _read_output(in_memory)
    chunked_result = _read_output(chunked)
    for name in ("dem", "slope", "aspect", "fdir", "facc"):
        np.testing.assert_allclose(
            chunked_result[name], in_memory_result[name], err_msg=name
        )


def test_facc_file_of_the_computed_facc_gives_the_same_flow_direction(tmp_path):
    """Pick the same outlets with the facc as file as with the computed facc."""
    import pyflwdir

    rng = np.random.default_rng(1)
    _, codes = pyflwdir.dem.fill_depressions(100.0 + 10.0 * rng.random((6, 6)))
    facc = pyflwdir.from_array(codes, ftype="d8").upstream_area("cell")
    fdir_file = _write_grid(tmp_path / "fdir.nc", codes)
    facc_file = _write_grid(tmp_path / "facc.nc", facc.astype("float64"))

    (computed,) = upscale_topography(
        tmp_path / "computed.nc", fdir_file=fdir_file, upscale_factor=3
    )
    (from_file,) = upscale_topography(
        tmp_path / "from_file.nc",
        fdir_file=fdir_file,
        facc_file=facc_file,
        upscale_factor=3,
    )

    np.testing.assert_array_equal(
        _read_output(from_file)["fdir"], _read_output(computed)["fdir"]
    )


@pytest.mark.usefixtures("_mhm_tools_logger_propagates")
def test_inconsistent_facc_still_gives_a_valid_direction(tmp_path, caplog):
    """Pick the outlet among the leaving cells and report the inconsistency."""
    codes = np.ones((2, 4), dtype="uint8")
    # the largest facc lies on a cell that drains within its block
    facc = np.ones((2, 4))
    facc[0, 0] = 100.0

    with caplog.at_level(logging.WARNING):
        (output,) = upscale_topography(
            tmp_path / "out.nc",
            fdir_file=_write_grid(tmp_path / "fdir.nc", codes),
            facc_file=_write_grid(tmp_path / "facc.nc", facc),
            upscale_factor=2,
        )

    np.testing.assert_array_equal(_read_output(output)["fdir"], [[1, 0]])
    assert "1 coarse cells have their largest flow accumulation" in caplog.text


@pytest.mark.parametrize(
    ("request_name", "match"),
    [
        ("no input", "Give a DEM"),
        ("facc without fdir", "needs the flow direction grid"),
        ("neither factor nor resolution", "exactly one"),
        ("factor and resolution", "exactly one"),
        ("resolution no multiple", "Cell Size missmatch"),
        ("shape no multiple", "no integer multiple"),
        ("grids differ", "differs"),
        ("memory above budget", "more than"),
    ],
)
def test_invalid_requests_are_rejected(tmp_path, request_name, match):
    """Reject requests the upscaling cannot fulfil before computing anything."""
    dem_file = _write_grid(tmp_path / "dem.nc", _create_plane(4, 4))
    requests = {
        "no input": {"upscale_factor": 2},
        "facc without fdir": {
            "dem_file": dem_file,
            "facc_file": _write_grid(tmp_path / "facc.nc", np.ones((4, 4))),
            "upscale_factor": 2,
        },
        "neither factor nor resolution": {"dem_file": dem_file},
        "factor and resolution": {
            "dem_file": dem_file,
            "upscale_factor": 2,
            "target_resolution": 2 * CELLSIZE,
        },
        "resolution no multiple": {
            "dem_file": dem_file,
            "target_resolution": 1.5 * CELLSIZE,
        },
        "shape no multiple": {
            "dem_file": _write_grid(tmp_path / "odd.nc", _create_plane(5, 4)),
            "upscale_factor": 2,
        },
        "grids differ": {
            "dem_file": dem_file,
            "fdir_file": _write_grid(tmp_path / "wide.nc", np.ones((4, 6), "uint8")),
            "upscale_factor": 2,
        },
        "memory above budget": {
            "dem_file": dem_file,
            "upscale_factor": 2,
            "available_mem_gib": 1e-9,
        },
    }

    with pytest.raises(ValueError, match=match):
        upscale_topography(tmp_path / "out.nc", **requests[request_name])


def test_cli_passes_the_options_to_upscale_topography(monkeypatch):
    """Hand the parsed options on, with the memory budget in GiB."""
    received_arguments = {}

    def record_upscale_topography(**kwargs):
        """Record the arguments upscale_topography is called with."""
        received_arguments.update(kwargs)
        return ()

    monkeypatch.setattr(
        topography_upscaling, "upscale_topography", record_upscale_topography
    )
    parser = argparse.ArgumentParser()
    _topography_upscaling.add_args(parser)
    arguments = parser.parse_args(
        [
            "-o",
            "out",
            "--fdir-file",
            "dir.nc",
            "--facc-file",
            "upa.nc",
            "--target-resolution",
            "0.0625",
            "--write-facc",
            "--keep-edge-outflow",
            "--available-mem",
            "2gb",
            "--ncpus",
            "4",
        ]
    )
    _topography_upscaling.run(arguments)

    assert received_arguments["fdir_file"] == "dir.nc"
    assert received_arguments["facc_file"] == "upa.nc"
    assert received_arguments["dem_file"] is None
    assert received_arguments["upscale_factor"] is None
    assert received_arguments["target_resolution"] == pytest.approx(0.0625)
    assert received_arguments["write_facc"] is True
    assert received_arguments["keep_edge_outflow"] is True
    assert received_arguments["depression_fill_dem"] is False
    assert received_arguments["facc_unit"] == "km2"
    assert received_arguments["available_mem_gib"] == pytest.approx(2.0)
    assert received_arguments["ncpus"] == 4
