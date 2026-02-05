import os

# Ensure matplotlib never opens GUI windows during tests.
os.environ.setdefault("MPLBACKEND", "Agg")
