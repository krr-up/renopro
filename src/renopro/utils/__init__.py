"""
Utilities.
"""

from typing import NoReturn


def assert_never(value: NoReturn) -> NoReturn:
    """Function to help mypy make exhaustiveness check when
    e.g. dispatching on enum values."""
    assert False, f"This code should never be reached, got: {value}"
