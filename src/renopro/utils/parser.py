"""
The command line parser for the project.
"""

import logging
import sys
from argparse import ArgumentParser
from textwrap import dedent
from typing import Any, cast

__all__ = ["get_parser"]

if sys.version_info[1] < 8:
    import importlib_metadata as metadata  # nocoverage
else:
    from importlib import metadata  # nocoverage

VERSION = metadata.version("renopro")


def get_parser() -> ArgumentParser:
    """
    Return the parser for command line options.
    """
    parser = ArgumentParser(
        prog="renopro",
        description=dedent(
            """\
            renopro:
            A tool for reification of non-ground clingo programs.
            """
        ),
    )

    levels = [
        (
            "error",
            logging.ERROR,
        ),
        (
            "warning",
            logging.WARNING,
        ),
        (
            "info",
            logging.INFO,
        ),
        (
            "debug",
            logging.DEBUG,
        ),
    ]

    def get(
        levels,
        name,
    ):
        for (
            key,
            val,
        ) in levels:
            if key == name:
                return val
        return None  # nocoverage

    parser.add_argument(
        "--log",
        default="warning",
        choices=[val for _, val in levels],
        metavar=f"{{{','.join(key for key, _ in levels)}}}",
        help="set log level [%(default)s]",
        type=cast(
            Any,
            lambda name: get(
                levels,
                name,
            ),
        ),
    )

    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"%(prog)s {VERSION}",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--reify",
        "-r",
        action="store_true",
    )
    group.add_argument(
        "--reflect",
        "-R",
        action="store_true",
    )
    group.add_argument(
        "--transform",
        "-t",
        action="store_true",
    )

    parser.add_argument(
        "string",
        type=str,
    )

    parser.add_argument(
        "--files",
        "-f",
        nargs="?",
    )

    return parser
