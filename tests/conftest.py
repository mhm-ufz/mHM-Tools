"""Pytest configuration to prefer local src/ over installed packages."""

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault(
    "MPLBACKEND", "Agg"
)  # Use non-interactive backend for matplotlib in tests
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.is_dir():
    sys.path.insert(0, str(SRC))


@pytest.fixture
def _offline_map_projections(monkeypatch):
    """Record the projection of every saved map and skip the map features.

    Coastlines and borders would be downloaded by cartopy; the tests only need
    the gauges, so ``add_feature`` is a no-op.

    Returns
    -------
    list
        The cartopy projection of every figure saved by a map function.
    """
    pytest.importorskip("cartopy")
    from cartopy.mpl.geoaxes import GeoAxes
    from matplotlib.figure import Figure

    projections = []
    save_figure = Figure.savefig

    def record_and_save(figure, *args, **kwargs):
        """Record the map projection before saving the figure."""
        projections.append(figure.axes[0].projection)
        return save_figure(figure, *args, **kwargs)

    monkeypatch.setattr(GeoAxes, "add_feature", lambda *args, **kwargs: None)
    monkeypatch.setattr(Figure, "savefig", record_and_save)
    return projections
