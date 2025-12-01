"""Tests for landcover_ascii_to_nc module."""

import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from mhm_tools.pre.landcover_ascii_to_nc import (
    add_time_bounds_cf,
    convert_lc_ascii_to_nc,
    parse_nml_for_landcover,
)


def test_parse_multiple_dir_lcover_and_filenames():
    """Parse multiple dir_LCover(), filenames, and year ranges from namelist."""
    nml = """
&directories_general
dir_LCover(1) = "input/luse/"
dir_LCover(2) = "other/luse/"
/

&LCover
LCoverfName(1) = 'lc_1981.asc'
LCoverfName(2) = 'lc_1991.asc'
LCoverYearStart(1) = 1981
LCoverYearEnd(1)   = 1990
LCoverYearStart(2) = 1991
LCoverYearEnd(2)   = 2000
/
"""
    dir_map, entries, fname_map, n_domains = parse_nml_for_landcover(nml)

    # directory mapping
    assert isinstance(dir_map, dict)
    assert dir_map[1] == "input/luse/"
    assert dir_map[2] == "other/luse/"

    # filenames
    assert fname_map[1] == "lc_1981.asc"
    assert fname_map[2] == "lc_1991.asc"

    # year ranges
    assert entries[1]["year_start"] == 1981
    assert entries[1]["year_end"] == 1990
    assert entries[2]["year_start"] == 1991
    assert entries[2]["year_end"] == 2000

    # nDomains not present in this namelist block
    assert n_domains is None


def test_year_range_gaps():
    """Gapped coverage (missing intermediate years) should raise."""
    nml = """
&LCover
LCoverfName(1) = 'lc_1981.asc'
LCoverfName(2) = 'lc_1995.asc'
LCoverYearStart(1) = 1981
LCoverYearEnd(1)   = 1990
LCoverYearStart(2) = 1995
LCoverYearEnd(2)   = 2000
/
"""
    # Expect a complaint about missing 1991..1994
    with pytest.raises(ValueError, match="missing years 1991..1994"):
        parse_nml_for_landcover(nml)


def test_year_range_overlap():
    """Overlapping coverage (two ranges touch/overlap) should raise."""
    nml = """
&LCover
LCoverfName(1) = 'lc_1981.asc'
LCoverfName(2) = 'lc_1989.asc'
LCoverYearStart(1) = 1981
LCoverYearEnd(1)   = 1990
LCoverYearStart(2) = 1989
LCoverYearEnd(2)   = 2000
/
"""
    with pytest.raises(ValueError, match="overlapping"):
        parse_nml_for_landcover(nml)


def test_single_year_range():
    """Single file with a single [start,end] range should be parsed."""
    nml = """
&LCover
LCoverfName(1) = 'lc_1981.asc'
LCoverYearStart(1) = 1981
LCoverYearEnd(1)   = 1990
/
"""
    _, entries, fname_map, _ = parse_nml_for_landcover(nml)

    assert len(entries) == 1
    assert fname_map[1] == "lc_1981.asc"
    assert entries[1]["year_start"] == 1981
    assert entries[1]["year_end"] == 1990


def test_choose_dir_by_ndomains():
    """
    If nDomains > 1 and dir_LCover(i) is present for multiple indices,
    we should still parse them correctly.
    """
    nml = """
nDomains = 2
dir_LCover(1) = "input/luse/"
dir_LCover(2) = "other/luse/"

&LCover
LCoverfName(1) = 'a.asc'
LCoverYearStart(1) = 1981
LCoverYearEnd(1)   = 1990
/
"""
    dir_map, entries, fname_map, n_domains = parse_nml_for_landcover(nml)

    assert n_domains == 2
    assert dir_map[1] == "input/luse/"
    assert dir_map[2] == "other/luse/"
    assert fname_map[1] == "a.asc"
    assert entries[1]["year_start"] == 1981
    assert entries[1]["year_end"] == 1990


def test_absolute_dir_is_kept():
    """Absolute dir_LCover() paths should be parsed without modification."""
    nml = """
dir_LCover(1) = "/abs/path/to/luse/"

&LCover
LCoverfName(1) = 'a.asc'
LCoverYearStart(1) = 1981
LCoverYearEnd(1)   = 1990
/
"""
    dir_map, entries, fname_map, n_domains = parse_nml_for_landcover(nml)

    assert dir_map[1] == "/abs/path/to/luse/"
    assert fname_map[1] == "a.asc"
    assert entries[1]["year_start"] == 1981
    assert entries[1]["year_end"] == 1990
    assert n_domains is None


def test_add_time_bounds_cf():
    """add_time_bounds_cf should construct numeric time + time_bnds."""
    # Make a tiny dataset with 2 timesteps
    ds = xr.Dataset(
        data_vars={
            "land_cover": (
                ("time", "lat", "lon"),
                np.zeros((2, 2, 2)),
            ),
        },
        coords={
            "time": [
                np.datetime64("1981-01-01"),
                np.datetime64("1991-01-01"),
            ],
            "lat": [1.0, 2.0],
            "lon": [1.0, 2.0],
        },
    )

    # Pretend these came from the namelist:
    # block1 covers 1981-1990, block2 covers 1991-2000
    input_infos = {
        1: {"year_start": 1981, "year_end": 1990},
        2: {"year_start": 1991, "year_end": 2000},
    }

    ds_with_bounds = add_time_bounds_cf(ds, input_infos)

    # Check presence and metadata
    assert "time" in ds_with_bounds.coords
    assert "time_bnds" in ds_with_bounds.variables

    assert ds_with_bounds.time.attrs.get("bounds") == "time_bnds"
    assert ds_with_bounds.time_bnds.dims == ("time", "nv")
    assert ds_with_bounds.time_bnds.shape == (2, 2)

    # time should now be numeric (float days since ref)
    assert np.issubdtype(ds_with_bounds.time.dtype, np.floating)


def test_convert_lc_ascii_to_nc(tmp_path: Path):
    """
    Integration test:
    - write a synthetic namelist pointing to two ASCII rasters
    - run convert_lc_ascii_to_nc
    - open result and sanity check structure
    """
    # setup folders
    luse_dir = tmp_path / "input" / "luse"
    luse_dir.mkdir(parents=True)

    # copy test ASCII rasters from repo test data
    test_files_dir = Path(__file__).parent / "files"
    shutil.copy(test_files_dir / "lc_1981.asc", luse_dir / "lc_1981.asc")
    shutil.copy(test_files_dir / "lc_1991.asc", luse_dir / "lc_1991.asc")

    # build namelist with continuous coverage 1981-1990, 1991-2000
    nml_content = f"""
&directories_general
dir_LCover(1) = "{luse_dir!s}/"
/

&LCover
LCoverfName(1) = 'lc_1981.asc'
LCoverfName(2) = 'lc_1991.asc'
LCoverYearStart(1) = 1981
LCoverYearEnd(1)   = 1990
LCoverYearStart(2) = 1991
LCoverYearEnd(2)   = 2000
/
"""
    nml_file = tmp_path / "test.nml"
    nml_file.write_text(nml_content)

    out_nc = tmp_path / "landcover.nc"

    # run the converter (default var name "land_cover")
    convert_lc_ascii_to_nc(
        input_nml=nml_file,
        output=out_nc,
    )

    # read NetCDF
    ds = xr.open_dataset(out_nc)

    # structural checks
    assert "time" in ds.coords
    assert "time_bnds" in ds.variables

    # We had two ASCII inputs -> 2 time steps
    assert ds.sizes["time"] == 2
    assert ds.time_bnds.shape == (2, 2)

    # CF-ish metadata survived
    assert ds.time.attrs.get("bounds") == "time_bnds"
    assert ds.time_bnds.dims == ("time", "nv")

    # data variable exists and first dimension is time
    assert "land_cover" in ds
    assert ds.land_cover.shape[0] == 2

    ds.close()
