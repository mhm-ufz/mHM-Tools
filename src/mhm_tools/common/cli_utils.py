import argparse
import logging

from mhm_tools.common.logger import ErrorLogger

logger = logging.getLogger(__name__)


def parse_coords(coords_str):
    # Split the input string by comma and convert each part to a float
    try:
        lat, lon = map(float, coords_str.split(","))
        return lat, lon
    except ValueError:
        with ErrorLogger(logger):
            raise argparse.ArgumentTypeError(
                "Coordinates must be two comma-separated floats."
            )
