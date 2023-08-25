"""Test cases for transformation of AST using a meta-encoding."""
from pathlib import Path
from unittest import TestCase

from renopro.rast import ReifiedAST

tests_dir = Path("tests", "asp")
test_transform_dir = tests_dir / "transform"


class TestTransform(TestCase):
    """Tests for AST transformations defined via a meta-encodings."""

    base_str = "#program base.\n"

    def test_transform_not_bad(self):
        """Test adding an additional literal to the body of rules.

        Desired outcome: for something to be good, we want to require
        that it's not bad.
        """
        rast = ReifiedAST()
        files_dir = test_transform_dir / "not_bad"
        rast.reify_files([files_dir / "input.lp"])
        rast.transform(meta_files=[files_dir / "transform.lp"])
        rast.reflect()
        transformed_str = rast.program_string.strip()
        with (files_dir / "output.lp").open("r") as output:
            expected_str = self.base_str + output.read().strip()
        self.assertEqual(transformed_str, expected_str)

    def test_transform_add_time(self):
        """Test transforming a temporal logic program with the
        previous operator into a program with explicit time points."""
        for testname in ["sad", "very_sad", "constant"]:
            with self.subTest(testname=testname):
                rast = ReifiedAST()
                files_dir = test_transform_dir / "prev_to_timepoints"
                rast.reify_files([files_dir / (testname + "_input.lp")])
                rast.transform(meta_files=[files_dir / "transform.lp"])
                rast.reflect()
                transformed_str = rast.program_string.strip()
                with (files_dir / (testname + "_output.lp")).open("r") as output:
                    expected_str = self.base_str + output.read().strip()
                self.assertEqual(transformed_str, expected_str)

    def test_transform_bad_input(self):
        """Test transform behavior under bad input."""
        rast = ReifiedAST()
        # should log warning if rast has no facts before transformation
        with self.assertLogs("renopro.rast", level="WARNING"):
            rast.transform(meta_str="")
        # should raise error if no meta program is provided
        with self.assertRaises(ValueError):
            rast.transform()
