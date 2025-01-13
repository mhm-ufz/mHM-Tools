"""Provide general cli functionality."""

import argparse
import logging

from mhm_tools.common.logger import ErrorLogger

logger = logging.getLogger(__name__)


def parse_coords(coords_str):
    """Split the input string of 'lat,lon' by comma and convert each part to a float."""
    try:
        lat, lon = map(float, coords_str.split(","))
        return lat, lon
    except ValueError as verr:
        with ErrorLogger(logger):
            raise argparse.ArgumentTypeError from verr(
                "Coordinates must be two comma-separated floats."
            )
