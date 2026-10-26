"""Tests that importing mhm-tools and its CLI loads only what a call needs."""

import json
import subprocess
import sys

import pytest

DATA_LIBRARIES = {"xarray", "dask", "rasterio", "geopandas", "scipy"}


def _get_modules_loaded_by(code):
    """Get the modules a fresh Python process has loaded after running code.

    Args:
        code (str): Python statements to run.

    Returns
    -------
        set: Names of all modules in ``sys.modules`` after the statements.
    """
    script = f"{code}\nimport json, sys\nprint(json.dumps(sorted(sys.modules)))"
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    return set(json.loads(result.stdout.strip().splitlines()[-1]))


def test_importing_mhm_tools_loads_no_data_libraries():
    """Import the package and its constants without xarray, dask and co."""
    modules = _get_modules_loaded_by("import mhm_tools, mhm_tools.common.constants")
    assert not DATA_LIBRARIES & modules


def test_importing_the_cli_loads_no_command_and_no_tui():
    """Import the CLI without any command module, data library or Textual."""
    modules = _get_modules_loaded_by("import mhm_tools._cli._main")
    assert not (DATA_LIBRARIES | {"trogon", "textual"}) & modules
    command_modules = {
        name
        for name in modules
        if name.startswith("mhm_tools._cli._") and name != "mhm_tools._cli._main"
    }
    assert not command_modules


def test_command_help_loads_only_its_own_command_module():
    """Show the help of one command with its module only, still without xarray."""
    modules = _get_modules_loaded_by(
        "from mhm_tools._cli._main import main\n"
        "main(['evaluation', 'discharge-evaluation', '-h'])"
    )
    assert "mhm_tools._cli._discharge_evaluation" in modules
    assert "mhm_tools._cli._twsa_evaluation" not in modules
    assert not DATA_LIBRARIES & modules


@pytest.mark.parametrize(
    "code",
    [
        # the submodule bound on the package must not replace the function export
        "import mhm_tools.pre.format_lc_data\n"
        "import mhm_tools.pre as pre\n"
        "from mhm_tools.pre.format_lc_data import format_lc_data\n"
        "assert pre.format_lc_data is format_lc_data",
        "from mhm_tools.pre import format_lc_data, format_lc_periods\n"
        "assert callable(format_lc_data) and callable(format_lc_periods)",
        "import mhm_tools\n"
        "assert mhm_tools.common.netcdf.DEFAULT_COMPLEVEL == 1\n"
        "assert mhm_tools.post.__name__ == 'mhm_tools.post'\n"
        "assert mhm_tools.pre.__name__ == 'mhm_tools.pre'",
        "from mhm_tools.common import NO_DATA, read_header, set_netcdf_encoding\n"
        "from mhm_tools.common import constants, netcdf\n"
        "assert callable(set_netcdf_encoding) and netcdf.generate_bounds",
    ],
)
def test_lazy_package_exports_stay_reachable(code):
    """Reach every former eager export of the packages, in any import order."""
    _get_modules_loaded_by(code)


def test_create_dem_derivatives_help_loads_no_data_libraries():
    """Show the create-dem-derivatives help without xarray, rasterio and co."""
    modules = _get_modules_loaded_by(
        "from mhm_tools._cli._main import main\n"
        "main(['setup-creation', 'create-dem-derivatives', '-h'])"
    )
    assert "mhm_tools._cli._create_dem_derivatives" in modules
    assert not DATA_LIBRARIES & modules
