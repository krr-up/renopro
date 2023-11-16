"""Test cases for transformation of AST using a meta-encoding."""
import re
from itertools import count
from pathlib import Path
from typing import Dict, List, Literal
from unittest import TestCase

from clingo import Control

import renopro.predicates as preds
from renopro.rast import ReifiedAST, TransformationError

tests_dir = Path("tests", "asp")
test_transform_dir = tests_dir / "transform"


class TestTransform(TestCase):
    """Tests for AST transformations defined via a meta-encodings."""

    def setUp(self):
        # reset id generator between test cases so reification
        # auto-generates the expected integers
        preds.id_count = count()

    def assertTrasformEqual(
        self,
        input_files: List[Path],
        meta_files: List[Path],
        expected_output_file: Path,
        reify_location=True,
    ):
        rast = ReifiedAST(reify_location=reify_location)
        rast.reify_files(input_files)
        rast.transform(meta_files=meta_files)
        rast.reflect()
        transformed_str = rast.program_string.strip()
        with expected_output_file.open("r") as f:
            expected_str = self.base_str + f.read().strip()
        self.assertEqual(transformed_str, expected_str)

    def assertTransformLogs(
        self,
        input_files: List[Path],
        meta_files: List[Path],
        level: Literal["DEBUG", "INFO", "WARNING", "ERROR"],
        message2num_matches: Dict[str, int],
        reify_location=True,
    ):
        rast = ReifiedAST(reify_location=reify_location)
        rast.reify_files(input_files)
        with self.assertLogs("renopro.rast", level=level) as cm:
            try:
                rast.transform(meta_files=meta_files)
            except TransformationError as e:
                if level != "ERROR":
                    raise e
                for message, expected_num in message2num_matches.items():
                    assert_exc_msg = (
                        f"Expected {expected_num} "
                        "matches for exception message "
                        f"'{message}', found "
                    )
                    num_exception_matches = len(re.findall(message, str(e)))
                    self.assertEqual(
                        num_exception_matches,
                        expected_num,
                        msg=assert_exc_msg + str(num_exception_matches),
                    )
            logs = "\n".join(cm.output)
            for message, expected_num in message2num_matches.items():
                assert_msg = (
                    f"Expected {expected_num} "
                    "matches for log message "
                    f"'{message}', found "
                )
                reo = re.compile(message)
                num_log_matches = len(reo.findall(logs))
                self.assertEqual(
                    num_log_matches, expected_num, msg=assert_msg + str(num_log_matches)
                )

    base_str = "#program base.\n"


class TestTransformSimple(TestTransform):
    """Test case for testing basic transform functionality and simple
    transformations"""

    def test_transform_unsat(self):
        "Test that unsatisfiable transformation meta-encoding raises error."
        rast = ReifiedAST()
        pattern = r"Transformation encoding is unsatisfiable."
        with self.assertRaisesRegex(TransformationError, pattern):
            rast.transform(meta_str="#false.")

    def test_transform_log(self):
        "Test logging capabilities of transform."
        files_dir = test_transform_dir / "log"
        rast = ReifiedAST()
        pattern = (
            r"First argument of log term must be one of the string "
            r"symbols: 'debug', 'info', 'warning', 'error'"
        )
        with self.assertRaisesRegex(TransformationError, pattern):
            rast.transform(meta_str="#show log('hello', 'world') : #true.")
        self.assertTransformLogs(
            [files_dir / "input.lp"],
            [files_dir / "info_with_loc.lp"],
            "INFO",
            {r"7:12-13: Found function with name a.": 1},
        )
        self.assertTransformLogs(
            [files_dir / "input.lp"],
            [files_dir / "info_no_loc.lp"],
            "INFO",
            {r"Found function with name a. Sorry, no location.": 1},
        )
        self.assertTransformLogs(
            [files_dir / "input.lp"],
            [files_dir / "error_no_loc.lp"],
            "ERROR",
            {r"Found function with name a. Sorry, no location.": 1},
        )
        self.assertTransformLogs(
            [files_dir / "input.lp"],
            [files_dir / "error_with_loc.lp"],
            "ERROR",
            {r"7:12-13: Found function with name a.": 1},
        )

    def test_transform_bad_input(self):
        """Test transform behavior under bad input."""
        rast = ReifiedAST()
        # should log warning if rast has no facts before transformation
        with self.assertLogs("renopro.rast", level="WARNING"):
            rast.transform(meta_str="")
        # should raise error if no meta program is provided
        with self.assertRaises(ValueError):
            rast.transform()

    def test_transform_not_bad(self):
        """Test adding an additional literal to the body of rules.

        Desired outcome: for something to be good, we want to require
        that it's not bad.
        """
        files_dir = test_transform_dir / "not_bad"
        self.assertTrasformEqual(
            [files_dir / "input.lp"],
            [files_dir / "transform.lp"],
            files_dir / "output.lp",
        )

    def test_transform_add_time(self):
        """Test transforming a temporal logic program with the
        previous operator into a program with explicit time points."""
        files_dir = test_transform_dir / "prev_to_timepoints"
        for testname in ["sad", "very_sad", "constant"]:
            with self.subTest(testname=testname):
                self.assertTrasformEqual(
                    [files_dir / (testname + "_input.lp")],
                    [files_dir / "transform.lp"],
                    files_dir / (testname + "_output.lp"),
                )

