from functools import wraps
import inspect
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
    if level is None:
        return logger
    if type(level) is not str:
        # raise TypeError(f"Invalid log level type: {type(level)}")
        logger.error(f"Invalid log level type: {type(level)} - using default log level INFO")
        level = "INFO"
    if level not in LOG_LEVELS:
        logger.error(f"Invalid log level: {level} - using default log level INFO")
        level = "INFO"
        # raise ValueError(f"Invalid log level: {level}")
    level = level.upper()
    logger.setLevel(LOG_LEVELS[level])
    logger.info(f"Set log level to {level}")
    return logger


def log_arguments():
    """Log all non-None arguments passed to a function."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get the signature of the function
            signature = inspect.signature(func)
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Extract arguments and filter out None values
            non_none_args = {k: v for k, v in bound_args.arguments.items() if v is not None}

            # Log the arguments
            msg = f"Function '{func.__name__}' called with the following arguments: \n"
            for arg, value in non_none_args.items():
                msg += f"  {arg}: {value} \n"
            logger.info(msg)

            # Call the original function
            return func(*args, **kwargs)
        return wrapper
    return decorator

