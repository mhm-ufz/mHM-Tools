"""Pytest configuration to prefer local src/ over installed packages."""

import os
import sys
from pathlib import Path

os.environ.setdefault(
    "MPLBACKEND", "Agg"
)  # Use non-interactive backend for matplotlib in tests
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.is_dir():
    sys.path.insert(0, str(SRC))
