"""
Setup project wide loggers.
"""

import logging
import sys
from functools import partial
from typing import Callable

from clingo import MessageCode

COLORS = {
    "GREY": "\033[90m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "RED": "\033[91m",
    "NORMAL": "\033[0m",
}

log_string2level = {
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


class SingleLevelFilter(logging.Filter):
    """
    Filter levels.
    """

    def __init__(self, passlevel: int, reject: bool):
        # pylint: disable=super-init-not-called
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record: logging.LogRecord) -> bool:
        if self.reject:
            return record.levelno != self.passlevel  # nocoverage

        return record.levelno == self.passlevel


def setup_logger(name: str, level: int) -> logging.Logger:
    """
    Setup logger.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)
    log_message_str = "{}%(levelname)s:{}  - %(message)s{}"

    def set_handler(level: int, color: str) -> None:
        handler = logging.StreamHandler(sys.stderr)
        handler.addFilter(SingleLevelFilter(level, False))
        handler.setLevel(level)
        formatter = logging.Formatter(
            log_message_str.format(COLORS[color], COLORS["GREY"], COLORS["NORMAL"])
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    set_handler(logging.INFO, "GREEN")
    set_handler(logging.WARNING, "YELLOW")
    set_handler(logging.DEBUG, "BLUE")
    set_handler(logging.ERROR, "RED")

    return logger


CLINGO_FSTRING = "clingo: %s"


def log_clingo_message(
    message_code: MessageCode, message: str, logger: logging.Logger
) -> None:  # nocoverage
    "Log clingo message at the appropriate level"
    if message_code is MessageCode.AtomUndefined:
        logger.info(CLINGO_FSTRING, message)
    elif message_code is MessageCode.FileIncluded:
        logger.warn(CLINGO_FSTRING, message)
    elif message_code is MessageCode.GlobalVariable:
        logger.info(CLINGO_FSTRING, message)
    elif message_code is MessageCode.OperationUndefined:
        logger.info(CLINGO_FSTRING, message)
    # not sure what the appropriate log level for "Other" is... just do info for now
    elif message_code is MessageCode.Other:
        logger.info(CLINGO_FSTRING, message)
    elif message_code is MessageCode.RuntimeError:
        logger.error(CLINGO_FSTRING, message)
    elif message_code is MessageCode.VariableUnbounded:
        logger.info(CLINGO_FSTRING, message)


def get_clingo_logger_callback(
    logger: logging.Logger,
) -> Callable[[MessageCode, str], None]:
    """Return a callback function to be used by a clingo.Control
    object to log to input logger."""
    return partial(log_clingo_message, logger=logger)
