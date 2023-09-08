# pylint: disable=too-many-public-methods
"""Test cases for reification functionality."""
from itertools import count
from pathlib import Path
from typing import List
from unittest import TestCase

from clingo.ast import parse_string
from clorm import FactBase, Predicate, UnifierNoMatchError

import renopro.predicates as preds
from renopro.rast import ChildQueryError, ChildrenQueryError, ReifiedAST

test_files = Path("tests", "asp", "reify_reflect")
well_formed_ast_files = test_files / "well_formed_ast"
malformed_ast_files = test_files / "malformed_ast"


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
        regex = r"(?s).*'literal\(1,\"pos\",attom\(2\)\)'.*'attom\(2\).*AtomField"
        with self.assertRaisesRegex(UnifierNoMatchError, expected_regex=regex):
            rast.add_reified_string(fact_str)

    def test_reified_files(self):
        """Test adding of reified facts from files to reified facts of a ReifiedAST."""
        rast = ReifiedAST()
        rast.add_reified_files([malformed_ast_files / "ast_fact.lp"])
        fb = FactBase([preds.Symbolic_Atom(id=12, symbol=preds.Function1(id=13))])
        self.assertSetEqual(rast.reified_facts, fb)


class TestReifyReflect(TestReifiedAST):
    """Base class for tests for reification and reflection of
    non-ground programs."""

    base_str = "#program base.\n"

    def assertReifyEqual(
        self, prog_str: str, ast_fact_files: List[str]
    ):  # pylint: disable=invalid-name
        "Assert that reification of prog_str results in ast_facts."
        ast_fact_files_str = [(well_formed_ast_files / f) for f in ast_fact_files]
        rast1 = ReifiedAST()
        rast1.reify_string(prog_str)
        reified_facts = rast1.reified_facts
        rast2 = ReifiedAST()
        rast2.add_reified_files(ast_fact_files_str)
        expected_facts = rast2.reified_facts
        self.assertSetEqual(reified_facts, expected_facts)  # type: ignore

    def assertReflectEqual(
        self, prog_str: str, ast_fact_files: List[str]
    ):  # pylint: disable=invalid-name
        "Assert that reflection of ast_facts results in prog_str."
        ast_fact_files_str = [(well_formed_ast_files / f) for f in ast_fact_files]
        rast = ReifiedAST()
        rast.add_reified_files(ast_fact_files_str)
        rast.reflect()
        expected_string = self.base_str + prog_str
        self.assertEqual(rast.program_string, expected_string)

    def assertReifyReflectEqual(
        self, prog_str: str, ast_fact_files: List[str]
    ):  # pylint: disable=invalid-name
        """Assert that reification of prog_str results in ast_facts,
        and that reflection of ast_facts result in prog_str."""

        for operation in ["reification", "reflection"]:
            with self.subTest(operation=operation):
                if operation == "reification":
                    self.assertReifyEqual(prog_str, ast_fact_files)
                elif operation == "reflection":
                    self.assertReflectEqual(prog_str, ast_fact_files)


class TestReifyReflectNormalPrograms(TestReifyReflect):
    """Test cases for reification and reflection of normal logic programs."""

    def test_rast_prop_fact(self):
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

    def test_rast_prop_normal_rule(self):
        """Test reification and reflection of a normal rule containing
        only propositional atoms."""
        self.assertReifyReflectEqual("a :- b; not c.", ["prop_normal_rule.lp"])

    def test_rast_function(self):
        """Test reification and reflection of a variable-free normal
        rule with a symbolic atom."""
        self.assertReifyReflectEqual("rel(2,1) :- rel(1,2).", ["atom.lp"])

    def test_rast_nested_function(self):
        "Test reification and reflection of a rule with a function term."
        self.assertReifyReflectEqual("next(move(a)).", ["function.lp"])

    def test_rast_variable(self):
        "Test reification and reflection of normal rule with variable terms."
        self.assertReifyReflectEqual("rel(Y,X) :- rel(X,Y).", ["variable.lp"])

    def test_rast_string(self):
        "Test reification and reflection of normal rule with a string term."
        self.assertReifyReflectEqual('yummy("carrot").', ["string.lp"])

    def test_rast_constant_term(self):
        "Test reification and reflection of a normal rule with constant term."
        self.assertReifyReflectEqual("good(human).", ["constant_term.lp"])

    def test_rast_interval(self):
        "Test reification and reflection of a normal rule with an interval term."
        self.assertReifyReflectEqual("a((1..3)).", ["interval.lp"])

    def test_rast_unary_operation(self):
        "Test reification and reflection of a normal rule with a unary operation."
        self.assertReifyReflectEqual("neg(-1).", ["unary_operation.lp"])

    def test_rast_binary_operation(self):
        "Test reification and reflection of a normal rule with a binary operation."
        self.assertReifyReflectEqual("equal((1+1),2).", ["binary_operation.lp"])

    def test_rast_pool(self):
        "Test reification and reflection of a normal rule with a pool."
        self.assertReifyReflectEqual("pool((1;a)).", ["pool.lp"])

    def test_rast_comparison(self):
        "Test reification and reflection of a comparison operator"
        self.assertReifyReflectEqual("1 < 2 != 3 > 4.", ["comparison.lp"])

    def test_rast_boolean_constant(self):
        "Test reification and reflection of a boolean constant."
        self.assertReifyReflectEqual("#false.\n#true.", ["bool_const.lp"])

    def test_rast_integrity_constraint(self):
        "Test reification and reflection of an integrity constraint."
        self.assertReifyEqual(":- a.", ["integrity_constraint.lp"])

    def test_rast_conditional_literal(self):
        "Test reification and reflection of a conditional literal."
        self.assertReifyReflectEqual("a :- b: c, d.", ["conditional_literal.lp"])

    def test_rast_disjunction(self):
        "Test reification and reflection of a disjunction."
        self.assertReifyReflectEqual("a; b: c, not d.", ["disjunction.lp"])

    def test_rast_node_failure(self):
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
        rast.add_reified_files([malformed_ast_files / "missing_child.lp"])
        regex = r"(?s).*atom\(4,function\(5\)\).*function\(5\).*found 0.*"
        with self.assertRaisesRegex(ChildQueryError, expected_regex=regex):
            rast.reflect()

    def test_child_query_error_multiple_found(self):
        """Reflection of a parent fact that expects a singe child fact
        but finds multiple should fail with an informative error
        message.

        """
        rast = ReifiedAST()
        rast.add_reified_files([malformed_ast_files / "multiple_child.lp"])
        regex = r"(?s).*atom\(4,function\(5\)\).*function\(5\).*found 2.*"
        with self.assertRaisesRegex(ChildQueryError, expected_regex=regex):
            rast.reflect()

    def test_children_query_error_multiple_in_same_position(self):
        """Refection of a child tuple with multiple elements in the
        same position should fail"""
        rast = ReifiedAST()
        rast.add_reified_files([malformed_ast_files / "multiple_in_same_pos.lp"])
        regex = (
            r"(?s).*comparison\(4,number\(5\),guards\(6\)\).*"
            r"multiple child facts in the same position.*guards\(6\)"
        )
        with self.assertRaisesRegex(ChildrenQueryError, expected_regex=regex):
            rast.reflect()

    def test_comparison_no_guard_found(self):
        """Reflection of a comparison fact with an empty guard tuple should fail."""
        rast = ReifiedAST()
        rast.add_reified_files([malformed_ast_files / "missing_guard_in_comparison.lp"])
        regex = (
            r"(?s).*comparison\(4,number\(5\),guards\(6\)\).*"
            # r".*Expected 1 or more.*guards\(6\).*."
        )
        with self.assertRaisesRegex(ChildrenQueryError, expected_regex=regex):
            rast.reflect()


class TestReifyReflectAggTheory(TestReifyReflect):
    """Test cases for reification and reflection of aggregates and
    theory atoms."""

    def test_rast_aggregate(self):
        "Test reification and reflection of a simple count aggregate."
        with self.subTest(operation="reify"):
            self.assertReifyEqual("1 {a: b; c}.", ["aggregate.lp"])
        with self.subTest(operation="reflect"):
            self.assertReflectEqual("1 <= { a: b; c }.", ["aggregate.lp"])

    def test_rast_simple_theory_atom(self):
        """Test reification and reflection of simple theory atoms
        consisting only of symbolic part."""
        with self.subTest(operation="reify"):
            self.assertReifyEqual("&a.", ["theory_atom_simple.lp"])
        with self.subTest(operation="reflect"):
            self.assertReflectEqual("&a { }.", ["theory_atom_simple.lp"])
        self.setUp()
        self.assertReifyReflectEqual('&a("b") { }.', ["theory_atom_simple_arg.lp"])
        self.setUp()
        self.assertReifyReflectEqual("&a { } | b.", ["theory_atom_simple_guard.lp"])

    def test_rast_theory_sequence(self):
        "Test reification and reflection of a theory sequence."
        self.assertReifyReflectEqual("&a { [1,b] }.", ["theory_sequence.lp"])

    def test_rast_theory_sequence_type(self):
        "Test that all theory sequence types are accepted in reified representation."

    def test_rast_theory_function(self):
        "Test reification and reflection of a theory function."
        self.assertReifyReflectEqual("&a { f(1) }.", ["theory_function.lp"])

    def test_rast_theory_term(self):
        "Test that all theory terms are accepted in reified representation."

    def test_rast_theory_unparsed_term(self):
        "Test reification and reflection of an unparsed theory term."
        self.assertReifyReflectEqual('&a { +1!-"b" }.', ["theory_unparsed_term.lp"])

    def test_rast_head_aggregate(self):
        "Test reification and reflection of a head aggregate."
        with self.subTest(operation="reify"):
            self.assertReifyEqual(
                "#sum { 1,2: a: not b; 3: c } = 4.", ["head_aggregate.lp"]
            )
        with self.subTest(operation="reflect"):
            self.assertReflectEqual(
                "4 = #sum { 1,2: a: not b; 3: c }.", ["head_aggregate.lp"]
            )

    def test_rast_body_aggregate(self):
        "Test reification and reflection of a body aggregate."
        with self.subTest(operation="reify"):
            self.assertReifyEqual(
                ":- #sum+ { 1,2: not a; 3: b } != 4.", ["body_aggregate.lp"]
            )
        with self.subTest(operation="reflect"):
            self.assertReflectEqual(
                "#false :- 4 != #sum+ { 1,2: not a; 3: b }.", ["body_aggregate.lp"]
            )


class TestReifyReflectStatements(TestReifyReflect):
    """Test cases for reification and reflection of statements."""

    def test_rast_external_false(self):
        "Test reification of an external statement with default value false."
        self.assertReifyReflectEqual(
            "#external a(X) : c(X); d(e(X)). [false]", ["external.lp"]
        )

    def test_rast_program_params(self):
        "Test reification and reflection of a program statement with parameters."
        self.assertReifyReflectEqual("#program acid(k).", ["program_acid.lp"])
