import logging

from mhm_tools.common.constants import LOG_LEVELS

logging.basicConfig(format="%(asctime)s - %(levelname)-8s - %(message)s")
logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)


def set_log_level(level):
    """Set the logging level.

    Parameters
    ----------
    level : str
        logging level

    """
    if level not in LOG_LEVELS:
        raise ValueError(f"Invalid log level: {level}")
    logger.setLevel(LOG_LEVELS[level])
    logger.info(f"Set log level to {level}")
    return logger
