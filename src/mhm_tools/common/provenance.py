"""Output provenance helpers."""

from __future__ import annotations

import logging
import shlex
import sys
from datetime import datetime, timezone

import xarray as xr

from mhm_tools import __version__

logger = logging.getLogger(__name__)


VERSION_ATTR = "mhm_tools_version"
CREATED_ATTR = "date_created"
HISTORY_ATTR = "history"
LEGACY_HISTORY_ATTR = "hist"


def creation_date() -> str:
    """Return the current creation date as an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def apply_output_provenance(ds: xr.Dataset) -> xr.Dataset:
    """Add mhm-tools version, creation date, and command to NetCDF global attrs."""
    ds = ds.copy(deep=False)
    date = creation_date()
    if __version__ != "not_available":
        ds.attrs[VERSION_ATTR] = __version__
    ds.attrs[CREATED_ATTR] = date
    ds.attrs[HISTORY_ATTR] = _append_history(
        ds.attrs.get(HISTORY_ATTR, ds.attrs.get(LEGACY_HISTORY_ATTR)),
        f"{date}: mhm-tools command: {_command_string()}",
    )
    return ds


def _append_history(existing_history, new_entry: str) -> str:
    if existing_history is None or str(existing_history).strip() == "":
        return new_entry
    return f"{existing_history}\n{new_entry}"


def _command_string() -> str:
    if not sys.argv:
        return "<unknown>"
    return shlex.join(str(arg) for arg in sys.argv)
