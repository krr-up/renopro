"""
The command line parser for the project.
"""

import logging
import sys
from argparse import REMAINDER, ArgumentParser
from pathlib import Path
from textwrap import dedent
from typing import Any, Optional, cast

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

    common_arg_parser = ArgumentParser(add_help=False)

    levels = [
        ("error", logging.ERROR),
        ("warning", logging.WARNING),
        ("info", logging.INFO),
        ("debug", logging.DEBUG),
    ]

    def get(levels: list[tuple[str, int]], name: Any) -> Optional[Any]:
        for key, val in levels:
            if key == name:
                return val
        return None  # nocoverage

    common_arg_parser.add_argument(
        "--log",
        "-l",
        default="warning",
        choices=[val for _, val in levels],
        metavar=f"{{{','.join(key for key, _ in levels)}}}",
        help="set log level [%(default)s]",
        type=cast(Any, lambda name: get(levels, name)),
    )

    common_arg_parser.add_argument(
        "--version", "-v", action="version", version=f"%(prog)s {VERSION}"
    )

    common_arg_parser.add_argument(
        "infiles", nargs="*", type=Path, help="Input files to be processed."
    )

    parser = ArgumentParser(
        prog="renopro",
        description=dedent(
            """\
            renopro:
            A tool for reification and reflection of non-ground clingo programs.
            """
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    reify_parser = subparsers.add_parser(
        "reify", help="Reify input program into ASP facts.", parents=[common_arg_parser]
    )
    reify_parser.add_argument(
        "--commented",
        "-c",
        action="store_true",
        help=(
            "When reifying, print documentation of predicates occurring in the "
            "output as ASP comments."
        ),
    )
    subparsers.add_parser(
        "reflect",
        help="Reflect input reified facts into their program string representation.",
        parents=[common_arg_parser],
    )
    transform_parser = subparsers.add_parser(
        "transform",
        help="Apply AST transformation to input reified facts via a meta-encoding.",
        parents=[common_arg_parser],
    )
    transform_parser.add_argument(
        "--meta-encoding",
        "-m",
        action="append",
        type=Path,
        help="Meta-encoding to be applied to reified facts.",
        required=True,
    )
    transform_parser.add_argument(
        "--input-format",
        "-i",
        help=(
            "Format of input to be transformed, either reified facts or a "
            "reflected program string. If input format is set to reflected, "
            "the input reflected program string is first reified into facts "
            "before transformation."
        ),
        choices=["reified", "reflected"],
        default="reflected",
    )
    transform_parser.add_argument(
        "--output-format",
        "-o",
        help=(
            "Format of output to be printed after transformation, either "
            "reified facts or a reflected program string. If output format is "
            "set to reflected, the transformed facts are reflected into a "
            "program string before printing."
        ),
        choices=["reified", "reflected"],
        default="reflected",
    )
    transform_parser.add_argument(
        "--clingo-options",
        "-C",
        nargs=REMAINDER,
        help=(
            "Additional arguments passed to clingo for grounding/solving "
            "logic program consisting of meta-encoding and input reified facts. "
            "All arguments after this option will be passed to clingo."
        ),
    )

    return parser
