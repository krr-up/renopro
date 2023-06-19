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

    def get_test_facts(self, fact_file_str: str):
        """Parse fact file from test directory."""
        facts = parse_fact_files(
            [str(test_reify_files / fact_file_str)],
            unifier=preds.AST_Facts)
        return facts

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
                    rast.reify_string(prog_str)
                    self.assertSetEqual(rast._reified, ast_facts)
                elif operation == "reflection":
                    rast = ReifiedAST()
                    rast.add_reified_facts(ast_facts)
                    rast.reflect()
                    expected_string = self.base_str + prog_str
                    self.assertEqual(rast.program_string, expected_string)


class TestReifyReflectSimplePrograms(TestReifyReflect):
    """Test cases for simple programs containing only a couple
    statements."""

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
        facts = self.get_test_facts("prop_fact.lp")
        self.assertReifyReflectEqual(prog_str, facts)

    def test_reify_program_prop_normal_rule(self):
        """
        Test reification of a normal rule containing only propositional atoms.
        """
        prog_str = "a :- b; not c."
        facts = self.get_test_facts("prop_normal_rule.lp")
        self.assertReifyReflectEqual(prog_str, facts)

    def test_reify_program_function(self):
        """
        Test reification of a variable-free normal rule with function symbols.
        """
        prog_str = "rel(2,1) :- rel(1,2)."
        facts = self.get_test_facts("function.lp")
        self.assertReifyReflectEqual(prog_str, facts)

    def test_reify_program_nested_function(self):
        prog_str = "next(move(a))."
        facts = self.get_test_facts("nested_function.lp")
        self.assertReifyReflectEqual(prog_str, facts)

    def test_reify_program_variable(self):
        """
        Test reification of normal rule with variables.
        """
        prog_str = "rel(Y,X) :- rel(X,Y)."
        facts = self.get_test_facts("variable.lp")
        self.assertReifyReflectEqual(prog_str, facts)

    def test_reify_program_string(self):
        """
        Test reification of normal rule with string.
        """
        prog_str = 'yummy("carrot").'
        facts = self.get_test_facts("string.lp")
        self.assertReifyReflectEqual(prog_str, facts)

    def test_reify_program_constant_term(self):
        """
        Test reification of normal rule with constant term.
        """
        prog_str = "good(human)."
        facts = self.get_test_facts("constant.lp")
        self.assertReifyReflectEqual(prog_str, facts)

    def test_reify_program_binary_operator(self):
        prog_str = "equal((1+1),2)."
        facts = self.get_test_facts("binary_operation.lp")
        self.assertReifyReflectEqual(prog_str, facts)

