"""Test cases for reification functionality."""
from itertools import count
from pathlib import Path
from typing import List
from unittest import TestCase

from clingo.ast import parse_string
from clorm import FactBase, Predicate, UnifierNoMatchError

import renopro.predicates as preds
from renopro.reify import ChildQueryError, ReifiedAST

reify_files = Path("src", "renopro", "asp", "tests", "reify")
good_reify_files = reify_files / "good_ast"
malformed_reify_files = reify_files / "malformed_ast"


class TestReifiedAST(TestCase):
    """Common base class for tests involving the ReifiedAST class."""

    def setUp(self):
        # reset id generator between test cases so reification
        # auto-generates the expected integers
        preds.id_count = count()


class TestReifiedASTInterface(TestReifiedAST):
    """Test interfaces of ReifiedAST class"""

    class NotAnASTFact(Predicate):
        """This is, in fact, not an ast fact."""

    def test_update_ast_facts(self):
        """Test updating of reified facts of a ReifiedAST by list of
        ast facts."""
        rast = ReifiedAST()
        ast_facts = [preds.Variable(id=0, name="X")]
        rast.add_reified_facts(ast_facts)
        self.assertSetEqual(rast.reified_facts, FactBase(ast_facts))
        rast = ReifiedAST()
        with self.assertRaises(UnifierNoMatchError):
            rast.add_reified_facts([self.NotAnASTFact()])

    def test_add_reified_string(self):
        """Test adding string representation of reified facts to
        reified facts of a ReifiedAST."""
        rast = ReifiedAST()
        fact = 'variable(0,"X").\n'
        rast.add_reified_string(fact)
        fb = FactBase([preds.Variable(id=0, name="X")])
        self.assertSetEqual(rast.reified_facts, fb)
        self.assertEqual(rast.reified_string, fact)
        rast = ReifiedAST()
        with self.assertRaises(UnifierNoMatchError):
            rast.add_reified_string('variance(0,"X").')

    def test_unification_error_message(self):
        """Test that ReifiedAST rejects adding of facts that do not
        unify against any ast predicate definition, with an
        informative error message.

        """
        rast = ReifiedAST()
        # first case: signature does not match any ast facts
        # should show the closest matching signature in the error message.
        fact_str = 'litteral(1,"pos",atom(2)).'
        regex = r"(?s).*'litteral\(1,\"pos\",atom\(2\)\)'\..*literal/3."
        with self.assertRaisesRegex(UnifierNoMatchError, expected_regex=regex):
            rast.add_reified_string(fact_str)
        # second case: argument of ast fact fails to unify
        # should show the argument that failed to unify with the specified field
        fact_str = 'literal(1,"pos",attom(2)).'
        regex = r"(?s).*'literal\(1,\"pos\",attom\(2\)\)'.*'attom\(2\).*Atom1Field"
        with self.assertRaisesRegex(UnifierNoMatchError, expected_regex=regex):
            rast.add_reified_string(fact_str)

    def test_reified_files(self):
        """Test adding of reified facts from files to reified facts of a ReifiedAST."""
        rast = ReifiedAST()
        rast.add_reified_files([malformed_reify_files / "ast_fact.lp"])
        fb = FactBase([preds.Atom(id=12, symbol=preds.Function1(id=13))])
        self.assertSetEqual(rast.reified_facts, fb)


class TestReifyReflect(TestReifiedAST):
    """Base class for tests for reification and reflection of
    non-ground programs."""

    base_str = ""

    def assertReifyReflectEqual(
        self, prog_str: str, ast_fact_files: List[str]
    ):  # pylint: disable=invalid-name
        """Assert that reification of prog_str results in ast_facts,
        and that reflection of ast_facts result in prog_str."""

        ast_fact_files_str = [(good_reify_files / f) for f in ast_fact_files]
        for operation in ["reification", "reflection"]:
            with self.subTest(operation=operation):
                if operation == "reification":
                    rast1 = ReifiedAST()
                    rast1.reify_string(prog_str)
                    reified_facts = rast1.reified_facts
                    rast2 = ReifiedAST()
                    rast2.add_reified_files(ast_fact_files_str)
                    expected_facts = rast2.reified_facts
                    self.assertSetEqual(reified_facts, expected_facts)  # type: ignore
                elif operation == "reflection":
                    rast = ReifiedAST()
                    rast.add_reified_files(ast_fact_files_str)
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

    def test_reify_prop_fact(self):
        "Test reification and reflection of a propositional fact."
        self.assertReifyReflectEqual("a.", ["prop_fact.lp"])
        rast = ReifiedAST()
        rast.reify_string("a.")
        statements = []
        parse_string(
            "a.", lambda s: statements.append(s)  # pylint: disable=unnecessary-lambda
        )
        rast.reflect()
        self.assertEqual(rast.program_ast, statements)

    def test_reify_prop_normal_rule(self):
        "Test reification and reflection of a normal rule containing only propositional atoms."
        self.assertReifyReflectEqual("a :- b; not c.", ["prop_normal_rule.lp"])

    def test_reify_function(self):
        "Test reification and reflection of a variable-free normal rule with a symbolic atom."
        self.assertReifyReflectEqual("rel(2,1) :- rel(1,2).", ["atom.lp"])

    def test_reify_nested_function(self):
        "Test reification and reflection of a rule with a function term."
        self.assertReifyReflectEqual("next(move(a)).", ["function.lp"])

    def test_reify_variable(self):
        "Test reification and reflection of normal rule with variable terms."
        self.assertReifyReflectEqual("rel(Y,X) :- rel(X,Y).", ["variable.lp"])

    def test_reify_string(self):
        "Test reification and reflection of normal rule with a string term."
        self.assertReifyReflectEqual('yummy("carrot").', ["string.lp"])

    def test_reify_constant_term(self):
        "Test reification and reflection of a normal rule with constant term."
        self.assertReifyReflectEqual("good(human).", ["constant_term.lp"])

    def test_reify_interval(self):
        "Test reification and reflection of a normal rule with an interval term."
        self.assertReifyReflectEqual("a((1..3)).", ["interval.lp"])

    def test_reify_binary_operation(self):
        "Test reification and reflection of a normal rule with a binary operation."
        self.assertReifyReflectEqual("equal((1+1),2).", ["binary_operation.lp"])

    def test_reify_external_false(self):
        "Test reification of an external statement with default value false."
        self.assertReifyReflectEqual(
            "#external a(X) : c(X); d(e(X)). [false]", ["external.lp"]
        )

    def test_reify_program_params(self):
        "Test reification and reflection of a program statement with parameters."
        self.assertReifyReflectEqual("#program acid(k).", ["program_acid.lp"])

    def test_reify_node_failure(self):
        """Reification for any object not of type clingo.ast.AST or
        clingo.symbol.Symbol should raise a TypeError."""
        rast = ReifiedAST()
        not_node = {"not": "ast"}
        regex = "(?s).*AST or Symbol.*dict"
        with self.assertRaisesRegex(TypeError, expected_regex=regex):
            rast.reify_node(not_node)

    def test_child_query_error_none_found(self):
        """Reflection of a parent fact that expects a singe child fact
        but finds none should fail with an informative error message.

        """
        rast = ReifiedAST()
        rast.add_reified_files([malformed_reify_files / "missing_child.lp"])
        regex = r"(?s).*atom\(4,function\(5\)\).*function\(5\).*found none.*"
        with self.assertRaisesRegex(ChildQueryError, expected_regex=regex):
            rast.reflect()

    def test_child_query_error_multiple_found(self):
        """Reflection of a parent fact that expects a singe child fact
        but finds multiple should fail with an informative error
        message.

        """
        rast = ReifiedAST()
        rast.add_reified_files([malformed_reify_files / "multiple_child.lp"])
        regex = r"(?s).*atom\(4,function\(5\)\).*function\(5\).*found multiple.*"
        with self.assertRaisesRegex(ChildQueryError, expected_regex=regex):
            rast.reflect()
