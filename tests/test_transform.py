"""Test cases for transformation of AST using a meta-encoding."""
import re
from itertools import count
from pathlib import Path
from typing import Dict, List, Literal, Optional
import unittest
from unittest import TestCase

import renopro.predicates as preds
from renopro.rast import ReifiedAST, TransformationError

tests_dir = Path("tests", "asp")
test_transform_dir = tests_dir / "transform"


class TestTransform(TestCase):
    """Tests for AST transformations defined via a meta-encodings."""
    maxDiff = None
    def setUp(self):
        # reset id generator between test cases so reification
        # auto-generates the expected integers
        preds.id_count = count()

    def assertTransformEqual(
        self,
        input_files: List[Path],
        meta: List[List[Path]],
        expected_output_file: Path,
        rast: Optional[ReifiedAST] = None,
    ):
        rast = ReifiedAST() if rast is None else rast
        rast.reify_files(input_files)
        for meta_files in meta:
            rast.transform(meta_files=meta_files)
        rast.reflect()
        transformed_str = rast.program_string.strip()
        with expected_output_file.open("r") as f:
            expected_str = self.base_str + f.read().strip()
        self.assertEqual(transformed_str, expected_str)

    def assertTransformLogs(
        self,
        input_files: List[Path],
        meta: List[List[Path]],
        level: Literal["DEBUG", "INFO", "WARNING", "ERROR"],
        message2num_matchess: List[Dict[str, int]],
    ):
        rast = ReifiedAST()
        rast.reify_files(input_files)
        with self.assertLogs("renopro.rast", level=level) as cm:
            for meta_files, message2num_matches in zip(meta, message2num_matchess):
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
                        "matches for log message pattern"
                        f"'{message}' in {logs}, found "
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
            [[files_dir / "info_with_loc.lp"]],
            "INFO",
            [{r"7:12-13: Found function with name a.": 1}],
        )
        self.assertTransformLogs(
            [files_dir / "input.lp"],
            [[files_dir / "info_no_loc.lp"]],
            "INFO",
            [{r"Found function with name a. Sorry, no location.": 1}],
        )
        self.assertTransformLogs(
            [files_dir / "input.lp"],
            [[files_dir / "error_no_loc.lp"]],
            "ERROR",
            [{r"Found function with name a. Sorry, no location.": 1}],
        )
        self.assertTransformLogs(
            [files_dir / "input.lp"],
            [[files_dir / "error_with_loc.lp"]],
            "ERROR",
            [{r"7:12-13: Found function with name a.": 1}],
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
        self.assertTransformEqual(
            [files_dir / "input.lp"],
            [[files_dir / "transform.lp"]],
            files_dir / "output.lp",
        )

    def test_transform_add_time(self):
        """Test transforming a temporal logic program with the
        previous operator into a program with explicit time points."""
        files_dir = test_transform_dir / "prev_to_timepoints"
        for testname in ["sad", "very_sad", "constant"]:
            with self.subTest(testname=testname):
                self.assertTransformEqual(
                    [files_dir / (testname + "_input.lp")],
                    [[files_dir / "transform.lp"]],
                    files_dir / (testname + "_output.lp"),
                )


class TestTransformTheoryParsing(TestTransform):
    """Test case for transformation that parses theory terms."""

    files_dir = test_transform_dir / "theory-parsing"

    def test_parse_theory_unparsed_theory_term_clingo_unknown(self):
        """Theory operators without an entry in the operator table of
        the appropriate theory term type should raise error."""
        self.assertTransformLogs(
            [self.files_dir / "inputs" / "clingo-unknown-operator.lp"],
            [[
                self.files_dir / "parse-unparsed-theory-terms.lp",
                self.files_dir / "inputs" / "clingo-operator-table.lp",
            ]],
            "ERROR",
            [{
                r"1:8-19: No definition for operator '\*' of arity '1' found "
                r"for theory term type 'clingo'\.": 1,
                r"1:8-19: No definition for operator '~' of arity '2' found "
                r"for theory term type 'clingo'\.": 1,
            }],
        )

    def test_parse_unparsed_theory_terms_clingo(self):
        """Test that unparsed theory terms with clingo operators
        (using provided operator table) are parsed correctly.

        """
        self.assertTransformEqual(
            [self.files_dir / "inputs" / "clingo-unparsed-theory-term.lp"],
            [[
                self.files_dir / "parse-unparsed-theory-terms.lp",
                self.files_dir / "inputs" / "clingo-operator-table.lp",
            ]],
            self.files_dir / "outputs" / "clingo-unparsed-theory-term.lp",
        )

    def test_parse_theory_terms_clingo(self):
        """Test that undefined operators are detected correctly.

        """
        rast3 = ReifiedAST()
        rast3.add_reified_files(
            [self.files_dir / "inputs" / "clingo-theory-term-unknown-reified.lp"]
        )
        pattern = r"No definition for operator '\+' of arity '1' found"
        with self.assertRaisesRegex(TransformationError, pattern):
            rast3.transform(
                meta_files=[
                    self.files_dir / "parse-theory-terms.lp",
                    self.files_dir / "inputs" / "clingo-operator-table.lp",
                ]
            )
        pattern = r"No definition for operator '!!' of arity '2' found"
        with self.assertRaisesRegex(TransformationError, pattern):
            rast3.transform(
                meta_files=[
                    self.files_dir / "parse-theory-terms.lp",
                    self.files_dir / "inputs" / "clingo-operator-table.lp",
                ]
            )

    def test_parse_telingo(self):
        """Test that parsing of telingo theory terms works correctly."""
        self.assertTransformEqual(
            [self.files_dir / "inputs" / "telingo-unparsed-term.lp",
             self.files_dir / "inputs" / "telingo-theory-def.lp"],
            [[
                self.files_dir / "parse-theory-terms.lp"
            ]],
            self.files_dir / "outputs" / "telingo-parsed-term.lp",
        )

    def test_bad_theory_atom_occurrence(self):
        """Error should be raised when theory atoms occur in incorrect
        context."""
        self.assertTransformLogs(
            [self.files_dir / "inputs" / "bad-occurrences.lp"],
            [[
                self.files_dir / "parse-theory-terms.lp"
            ]],
            "ERROR",
            [{
                r":13:2-3: Theory atom found in unexpected context, allowed context: 'head'.": 1,
                r":14:2-3: Theory atom found in unexpected context, allowed context: 'body'.": 1,
                r":15:2-3: Theory atom found in unexpected context, allowed context: 'directive'.": 1
            }],
        )


class TestTransformMetaTelingo(TestTransform):
    """Test case for transformations that implement meta-telingo
    'frontend'."""

    files_dir = test_transform_dir / "meta-telingo"

    def test_transform_meta_telingo_externals_body(self):
        """Test emission of external statements to protect temporal
        operators in the body."""
        self.assertTransformEqual(
            [self.files_dir / "inputs" / "input-body.lp"],
            [[self.files_dir / "transform-subprogram.lp"], 
             [self.files_dir / "transform-add-externals.lp"]],
            self.files_dir / "outputs" / "output-body.lp"
        )

    def test_transform_meta_telingo_externals_head(self):
        """Test emission of external statements to protect temporal
        operators in the head."""
        self.assertTransformEqual(
            [self.files_dir / "inputs" / "input-head.lp"],
            [[self.files_dir / "transform-subprogram.lp"],
             [self.files_dir / "transform-add-externals.lp"]],
            self.files_dir / "outputs" / "output-head.lp"
        )

    def test_transform_meta_telingo_theory_validation(self):
        """Test validation of theory terms in telingo theory atom elements."""
        self.assertTransformLogs(
            [self.files_dir / "inputs" / "telingo-input-bad-term.lp",
             test_transform_dir / "theory-parsing" / "inputs" / "telingo-theory-def.lp"],
            [[test_transform_dir / "theory-parsing" / "parse-theory-terms.lp"],
             [self.files_dir / "transform-theory-to-symbolic.lp"]],
            "ERROR",
            [{},
             {r":1:2-5: The term tuple of tel theory atom elements must contain a single \(temporal\) formula.": 1,
              r":2:2-5: The term tuple of tel theory atom elements must contain a single \(temporal\) formula.": 1,
              r":3:10-16: Theory sequences not allowed in tel theory atoms, found sequence type \(\).": 1,
              r":4:5-8: tel theory atoms occurring in the body may only have one element.": 1}])

    def test_transform_meta_telingo_theory_to_symbol(self):
        """Test transformation of temporal formulas in theory format to symbolic."""
        self.assertTransformEqual(
            [self.files_dir / "inputs" / "telingo-input.lp",
             test_transform_dir / "theory-parsing" / "inputs" / "telingo-theory-def.lp"],
            [[test_transform_dir / "theory-parsing" / "parse-theory-terms.lp"],
             [self.files_dir / "transform-theory-to-symbolic.lp"],
             [self.files_dir / "transform-subprogram.lp"],
             [self.files_dir / "transform-add-externals.lp"]],
            self.files_dir / "outputs" / "telingo-output.lp"
        )


if __name__ == "__main__":
    unittest.main()
