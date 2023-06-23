from unittest import TestCase
from pathlib import Path

from renopro.reify import ReifiedAST


tests_dir = Path("src", "renopro", "asp", "tests")
test_transform_dir = tests_dir / "transform"


class TestTransform(TestCase):
    """Tests for ast transformations defined via a meta-encodings."""

    base_str = "#program base.\n"

    def test_transform_add_literal(self):
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