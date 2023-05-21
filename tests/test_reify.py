"""Test cases for reification functionality."""
from unittest import TestCase
from typing import Iterable

from clorm import FactBase

from renopro.reify import ReifiedAST
import renopro.clorm_predicates as preds


class TestReifiedAST(TestCase):
    """Test cases for ReifiedAST class."""

    def test_add_ast_facts(self):
        rast = ReifiedAST()
        ast_facts = [preds.Variable("X")]
        rast.add_ast_facts(ast_facts)
        self.assertEqual(rast.factbase, FactBase(ast_facts))

    def assertReifiedFactsEqual(self, prog_str: str, facts:
                                Iterable[preds.AST_Predicate]):
        """Assert that factbase contained in reified AST equals
        factbase constructed from list of AST Predicates."""
        rast = ReifiedAST.from_str(prog_str)
        self.assertEqual(rast.factbase, FactBase(facts))

    def assertConversionsEqual(self, prog_str: str,
                               ast_facts: Iterable[preds.AST_Predicate]):
        """Assert that reification of prog_str results in ast_facts,
        and that reflection of ast_facts result in prog_str."""
        for operation in ["reification", "reflection"]:
            with self.subTest(operation=operation):
                if operation == "reification":
                    rast = ReifiedAST.from_str(prog_str)
                    self.assertEqual(rast.factbase, FactBase(ast_facts))
                elif operation == "reflection":
                    rast = ReifiedAST.from_facts(ast_facts)
                    self.assertEqual(rast.reflect(), prog_str)

    def test_reify_program_prop_fact(self):

        """Test reification of a propositional fact, and the string
        representation of the associated clorm predicate.

        """
        prog_str = "a."
        atom_a = preds.Function(name="a", arguments=preds.Term_Tuple_Id(0))
        ast_facts = [
            preds.Rule(head=preds.Literal(sign_=0, atom=atom_a),
                       body=preds.Literal_Tuple_Id(identifier=0))
        ]
        ast_fact_str = "rule(literal(0,function(\"a\",term_tuple(0))),literal_tuple(0)).\n"
        self.assertConversionsEqual(prog_str, ast_facts)
        self.assertEqual(str(ReifiedAST.from_str(prog_str)), ast_fact_str)

    def test_reify_program_prop_normal_rule(self):
        """
        Test reification of a normal rule containing only propositional atoms.
        """
        prog_str = "a :- b; not c."
        atom_a = preds.Function(name="a", arguments=preds.Term_Tuple_Id(0))
        atom_b = preds.Function(name="b", arguments=preds.Term_Tuple_Id(1))
        atom_c = preds.Function(name="c", arguments=preds.Term_Tuple_Id(2))
        expected_facts = [
            preds.Rule(head=preds.Literal(sign_=0, atom=atom_a),
                       body=preds.Literal_Tuple_Id(identifier=0)),
            preds.Literal_Tuple(identifier=0, position=0,
                                element=preds.Literal(sign_=0, atom=atom_b)),
            preds.Literal_Tuple(identifier=0, position=1,
                                element=preds.Literal(sign_=1, atom=atom_c))
        ]
        self.assertConversionsEqual(prog_str, expected_facts)

    def test_reify_program_function(self):
        """
        Test reification of a variable-free normal rule with function symbols.
        """
        prog_str = "rel(2,1) :- rel(1,2)."
        rel21 = preds.Function(name="rel", arguments=preds.Term_Tuple_Id(0))
        rel12 = preds.Function(name="rel", arguments=preds.Term_Tuple_Id(1))
        expected_facts = [
            preds.Rule(head=preds.Literal(sign_=0, atom=rel21),
                       body=preds.Literal_Tuple_Id(identifier=0)),
            preds.Term_Tuple(identifier=0, position=0, element=preds.Integer(2)),
            preds.Term_Tuple(identifier=0, position=1, element=preds.Integer(1)),
            preds.Literal_Tuple(identifier=0, position=0,
                                element=preds.Literal(sign_=0, atom=rel12)),
            preds.Term_Tuple(identifier=1, position=0, element=preds.Integer(1)),
            preds.Term_Tuple(identifier=1, position=1, element=preds.Integer(2))
        ]
        self.assertConversionsEqual(prog_str, expected_facts)

    def test_reify_program_variable(self):
        """
        Test reification of normal rule with variables.
        """
        prog_str = "rel(Y,X) :- rel(X,Y)."
        relyx = preds.Function(name="rel", arguments=preds.Term_Tuple_Id(0))
        relxy = preds.Function(name="rel", arguments=preds.Term_Tuple_Id(1))
        expected_facts = [
            preds.Rule(head=preds.Literal(sign_=0, atom=relyx),
                       body=preds.Literal_Tuple_Id(identifier=0)),
            preds.Term_Tuple(identifier=0, position=0, element=preds.Variable("Y")),
            preds.Term_Tuple(identifier=0, position=1, element=preds.Variable("X")),
            preds.Literal_Tuple(identifier=0, position=0,
                                element=preds.Literal(sign_=0, atom=relxy)),
            preds.Term_Tuple(identifier=1, position=0, element=preds.Variable("X")),
            preds.Term_Tuple(identifier=1, position=1, element=preds.Variable("Y"))
        ]
        self.assertConversionsEqual(prog_str, expected_facts)

    def test_reify_program_string(self):
        """
        Test reification of normal rule with string.
        """
        prog_str = "yummy(\"carrot\")."
        yummy = preds.Function(name="yummy", arguments=preds.Term_Tuple_Id(0))
        expected_facts = [
            preds.Rule(head=preds.Literal(sign_=0, atom=yummy),
                       body=preds.Literal_Tuple_Id(identifier=0)),
            preds.Term_Tuple(identifier=0, position=0,
                             element=preds.String("carrot"))
        ]
        self.assertConversionsEqual(prog_str, expected_facts)

    def test_reify_program_constant_term(self):
        """
        Test reification of normal rule with constant term.
        """
        prog_str = "good(human)."
        good = preds.Function(name="good", arguments=preds.Term_Tuple_Id(0))
        human = preds.Function(name="human", arguments=preds.Term_Tuple_Id(1))
        expected_facts = [
            preds.Rule(head=preds.Literal(sign_=0, atom=good),
                       body=preds.Literal_Tuple_Id(identifier=0)),
            preds.Term_Tuple(identifier=0, position=0, element=human)
        ]
        self.assertConversionsEqual(prog_str, expected_facts)
