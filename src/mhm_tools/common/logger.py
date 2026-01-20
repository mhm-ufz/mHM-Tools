"""Provide logger functionality."""

import inspect
import logging
from contextlib import AbstractContextManager
from functools import wraps
from pathlib import Path

from mhm_tools.common.constants import LOG_LEVEL_STR, LOG_LEVELS


def configure_mhm_tools_logger(
    log_level=None,
    count_verbose=0,
    count_quiet=0,
    log_file=None,
    log_file_level=None,
    no_colsole_logging=False,
):
    """Configure the parser settings formatting as well as Stream and File handler."""
    logger = logging.getLogger("mhm_tools")
    logger.propagate = False
    general_level, error_msg_gnrl, error_msg_gnrl2 = get_lowest_level(
        log_level=log_level,
        log_file_level=log_file_level,
        count_verbose=count_verbose,
        count_quiet=count_quiet,
    )
    logger.setLevel(general_level)
    error_msg_fh, error_msg_ch = None, None
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    if not no_colsole_logging:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        ch_level, error_msg_ch = get_log_level(log_level, count_verbose, count_quiet)
        ch.setLevel(ch_level)
        logger.addHandler(ch)
        logger.info(f"Steam Handler set to log_level {LOG_LEVEL_STR[ch_level]}")
    if log_file:
        log_file = Path(log_file)
        if not log_file.suffix:
            logger.warning(f"Log file does not point to a file: {log_file}")
        else:
            if not log_file.parent.is_dir():
                logger.info(f"Creating folder for log_files {log_file.parent}")
                log_file.parent.mkdir(parents=True)
            if log_file.exists():
                logger.debug("Deleting existing log file.")
                log_file.unlink()
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            if log_file_level is not None:
                fh_level, error_msg_fh = get_log_level(
                    log_file_level, count_verbose, count_quiet
                )
            else:
                fh_level = general_level
            fh.setLevel(fh_level)
            logger.addHandler(fh)
            logger.info(f"File Handler set to log_level {LOG_LEVEL_STR[fh_level]}")
            logger.info(f"Writing logs to {log_file}")
    if error_msg_gnrl is not None:
        logger.error(f"Logger: {error_msg_gnrl}")
    if error_msg_ch is not None:
        logger.error(f"StreamHandler: {error_msg_ch}")
    if error_msg_fh is not None:
        logger.error(f"FileHandler: {error_msg_fh}")


def get_lowest_level(log_level, log_file_level, count_verbose, count_quiet):
    """Return the most verbose log level of all handlers."""
    llevel, ll_msg = get_log_level(
        log_level, count_verbose=count_verbose, count_quiet=count_quiet
    )
    if log_file_level is not None:
        lflevel, lf_msg = get_log_level(log_file_level)
    else:
        lflevel, lf_msg = llevel, None
    return min(llevel, lflevel), ll_msg, lf_msg


def get_log_level(level=None, count_verbose=0, count_quiet=0):
    """Set the logging level.

    Parameters
    ----------
    level : str
        logging level
    count_verbose : int
        verbosity
    count_quiet : int
        quietness

    returns level: int
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
    return level, error_msg


def log_arguments(log_level="Debug"):
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
            logger = logging.getLogger(inspect.getmodule(func).__name__)
            msg = f"Function '{func.__name__}' called with the following arguments: \n"
            for arg, value in non_none_args.items():
                msg += f"  {arg}: {value} \n"
            if log_level.upper() == "INFO":
                logger.info(msg)
            else:
                logger.debug(msg)
            # Call the original function
            try:
                return func(*args, **kwargs)
            except Exception as e:
                with ErrorLogger(logger):
                    raise e

        return wrapper

    return decorator


def log_errors(raise_exceptions=True):
    """Log all errors occuring in a function.

    Only use this wrapper if the results of the function are essential
    for further computations.
    """

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
            logger = logging.getLogger(inspect.getmodule(func).__name__)
            msg = f"Error in function '{func.__name__}' called with the following arguments: \n"
            for arg, value in non_none_args.items():
                msg += f"  {arg}: {value} \n"
            # Call the function
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # log function context
                logger.error(msg)
                if raise_exceptions:
                    with ErrorLogger(logger):
                        raise e
                else:
                    logger.error(e)
                    return None

        return wrapper

    return decorator


class ErrorLogger(AbstractContextManager):
    """Context manager to log Exceptions.

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

    def __enter__(self):
        """Enter function needed."""
        return self
