"""Provide logger functionality."""

import inspect
import logging
from contextlib import AbstractContextManager
from functools import wraps

from mhm_tools.common.constants import LOG_LEVELS


def configure_mhm_tools_logger(
    log_level=None,
    count_verbose=0,
    count_quiet=0,
    log_file=None,
    log_file_level=None,
    no_colsole_logging=False,
):
    """Configure the parser setting formating as well as Stream and Filehandler."""
    # logging.basicConfig(format="%(asctime)s - %(levelname)-8s - %(message)s")
    logger = logging.getLogger("mhm_tools")
    general_level = get_lowest_level(
        log_level=log_level,
        log_file_level=log_file_level,
        count_verbose=count_verbose,
        count_quiet=count_quiet,
    )
    error_msg_gnrl = set_log_level(logger, general_level, count_verbose, count_quiet)
    error_msg_fh, error_msg_ch = None, None
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    if not no_colsole_logging:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        error_msg_ch = set_log_level(ch, log_level, count_verbose, count_quiet)
        logger.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        if log_file_level is not None:
            error_msg_fh = set_log_level(fh, log_file_level, count_verbose, count_quiet)
        else:
            error_msg_fh = set_log_level(fh, log_level, count_verbose, count_quiet)
        logger.addHandler(fh)
    if error_msg_gnrl is not None:
        logger.error(f"Logger: {error_msg_gnrl}")
    if error_msg_ch is not None:
        logger.error(f"StreamHandler: {error_msg_ch}")
    if error_msg_fh is not None:
        logger.error(f"FileHandler: {error_msg_fh}")


def get_lowest_level(log_level, log_file_level, count_verbose, count_quiet):
    """Return the most verbose log level of all handlers."""
    if log_level is not None:
        llevel = LOG_LEVELS[log_level.upper()]
    else:
        llevel = LOG_LEVELS["INFO"] - 10 * count_verbose + 10 * count_quiet

    lflevel = LOG_LEVELS[log_file_level.upper()] if log_file_level is not None else 60
    return min(llevel, lflevel)


def set_log_level(handler, level=None, count_verbose=0, count_quiet=0):
    """Set the logging level.

    Parameters
    ----------
    level : str
        logging level
    count_verbose : int
        verbosity
    count_quiet : int
        quietness

    """
    error_msg = None
    if level is None:
        level = LOG_LEVELS["INFO"] - 10 * count_verbose + 10 * count_quiet
    elif not isinstance(level, int):
        if not isinstance(level, str):
            error_msg = (
                f"Invalid log level type: {type(level)} - using default log level INFO"
            )
            level = "INFO"
        level = level.upper()
        if level not in LOG_LEVELS:
            error_msg = f"Invalid log level: {level} - using default log level INFO"
            level = "INFO"
        level = LOG_LEVELS[level.upper()]
    handler.setLevel(level)
    return error_msg


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
            non_none_args = {
                k: v for k, v in bound_args.arguments.items() if v is not None
            }

            # Log the arguments
            msg = f"Function '{func.__name__}' called with the following arguments: \n"
            for arg, value in non_none_args.items():
                msg += f"  {arg}: {value} \n"
            logging.getLogger(inspect.getmodule(func).__name__).debug(msg)

            # Call the original function
            return func(*args, **kwargs)

        return wrapper

    return decorator


class ErrorLogger(AbstractContextManager):
    """
    Context manager to log Exceptions.

    Parameters
    ----------
    logger : string, None or logging.Logger instance, optional
        Logger name to use. Will be the root logger by default.
    do_log : Bool, optional
        Whether to really log errors. Will be true by default.
    """

    def __init__(self, logger=None, do_log=True):
        self.logger = logger.name if isinstance(logger, logging.Logger) else logger
        self.do_log = do_log

    def __exit__(self, exc_type, exc_value, traceback):
        """Log all exception messages."""
        if exc_value is not None and self.do_log:
            logging.getLogger(self.logger).exception(exc_value)
