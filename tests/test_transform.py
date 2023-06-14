from unittest import TestCase
from pathlib import Path

from renopro.reify import ReifiedAST


tests_dir = Path("src", "renopro", "asp", "tests")
test_transform_dir = tests_dir / "transform"


class TestTransform(TestCase):
    """Tests for ast transformations defined via a meta-encodings."""

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
        transformed_str = rast.program_string
        with (files_dir / "output.lp").open("r") as good_output:
            good_output_str = good_output.read().strip()
        self.assertEqual(transformed_str.strip(), good_output_str)

    def test_transform_add_time(self):
        """Test transforming a temporal logic program with the
        previous operator into a program with explicit time points."""
        rast = ReifiedAST()
        rast.reify_files([test_transform_files / "robot_input.lp"])
        rast.transform(meta_files=[test_transform_files / "robot_transform.lp"])
        transformed_str = rast.reflect()
        with (test_transform_files / "robot_output.lp").open("r") as robot_output:
            robot_output_str = robot_output.read().strip()
        self.assertEqual(transformed_str.strip(), robot_output_str)
