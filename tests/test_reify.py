"""Test cases for reification functionality."""
from unittest import TestCase
from pathlib import Path
from itertools import count

from clorm import FactBase, parse_fact_files

from renopro.reify import ReifiedAST
import renopro.clorm_predicates as preds


test_reify_files = Path("src", "renopro", "asp", "tests", "reify")


class TestReifyReflect(TestCase):
    """Base class for tests for reification and reflection of
    non-ground programs."""

    default_ast_facts = []
    base_str = ""

    def assertReifyReflectEqual(self,
                                prog_str: str,
                                ast_facts: FactBase):
        """Assert that reification of prog_str results in ast_facts,
        and that reflection of ast_facts result in prog_str."""
        ast_facts.add(self.default_ast_facts)
        # reset id generator counter so reification generates expected integers
        preds.id_count = count()
        for operation in ["reification", "reflection"]:
            with self.subTest(operation=operation):
                if operation == "reification":
                    rast = ReifiedAST()
                    rast.reify_program(prog_str)
                    self.assertSetEqual(rast._reified, ast_facts)
                elif operation == "reflection":
                    rast = ReifiedAST()
                    rast.add_reified_facts(ast_facts)
                    rast.reflect()
                    expected_string = self.base_str + prog_str
                    self.assertEqual(rast.program_string, expected_string)


class TestReifyReflectSingleNormalRule(TestReifyReflect):
    """Test cases for programs containing only a single normal rule."""

    base_str = "#program base.\n"

    def setUp(self):
        # reset id counter between test cases
        preds.id_count = count()

    def test_add_ast_facts(self):
        rast = ReifiedAST()
        ast_facts = [preds.Variable(id=0, name="X")]
        rast.add_reified_facts(ast_facts)
        self.assertEqual(rast._reified, FactBase(ast_facts))

    def test_reify_program_prop_fact(self):
        """Test reification of a propositional fact."""
        prog_str = "a."
        facts = parse_fact_files(
            [str(test_reify_files / "prop_fact.lp")],
            unifier=preds.AST_Facts)
        self.assertReifyReflectEqual(prog_str, facts)

    def test_reify_program_prop_normal_rule(self):
        """
        Test reification of a normal rule containing only propositional atoms.
        """
        prog_str = "a :- b; not c."
        facts = parse_fact_files(
            [str(test_reify_files / "prop_normal_rule.lp")],
            unifier=preds.AST_Facts)
        self.assertReifyReflectEqual(prog_str, facts)

    def test_reify_program_function(self):
        """
        Test reification of a variable-free normal rule with function symbols.
        """
        prog_str = "rel(2,1) :- rel(1,2)."
        facts = parse_fact_files(
            [str(test_reify_files / "function.lp")],
            unifier=preds.AST_Facts)
        self.assertReifyReflectEqual(prog_str, facts)

    def test_reify_program_nested_function(self):
        prog_str = "next(move(a))."
        facts = parse_fact_files(
            [str(test_reify_files / "nested_function.lp")],
            unifier=preds.AST_Facts)
        self.assertReifyReflectEqual(prog_str, facts)

    def test_reify_program_variable(self):
        """
        Test reification of normal rule with variables.
        """
        prog_str = "rel(Y,X) :- rel(X,Y)."
        facts = parse_fact_files(
            [str(test_reify_files / "variable.lp")],
            unifier=preds.AST_Facts)
        self.assertReifyReflectEqual(prog_str, facts)

    def test_reify_program_string(self):
        """
        Test reification of normal rule with string.
        """
        prog_str = 'yummy("carrot").'
        facts = parse_fact_files(
            [str(test_reify_files / "string.lp")],
            unifier=preds.AST_Facts)
        self.assertReifyReflectEqual(prog_str, facts)

    def test_reify_program_constant_term(self):
        """
        Test reification of normal rule with constant term.
        """
        prog_str = "good(human)."
        facts = parse_fact_files(
            [str(test_reify_files / "constant.lp")],
            unifier=preds.AST_Facts)
        self.assertReifyReflectEqual(prog_str, facts)

    def test_reify_program_binary_operator(self):
        prog_str = "equal((1+1),2)."
        facts = parse_fact_files(
            [str(test_reify_files / "binary_operation.lp")],
            unifier=preds.AST_Facts)
        self.assertReifyReflectEqual(prog_str, facts)

