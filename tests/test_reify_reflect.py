# pylint: disable=too-many-public-methods
"""Test cases for reification functionality."""
from itertools import count
from pathlib import Path
from unittest import TestCase

from clingo import ast
from clorm import FactBase, Predicate, UnifierNoMatchError

import renopro.predicates as preds
from renopro.rast import ChildQueryError, ChildrenQueryError, ReifiedAST

test_files = Path("tests", "asp", "reify_reflect")
reified_files = test_files / "reified"
reflected_files = test_files / "reflected"
malformed_reified_files = test_files / "malformed"


class TestReifiedAST(TestCase):
    """Common base class for tests involving the ReifiedAST class."""

    maxDiff = 930

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
        rast.add_reified_files([reified_files / "ast_fact.lp"])
        fb = FactBase([preds.SymbolicAtom(id=12, symbol=preds.Function.unary(id=13))])
        self.assertSetEqual(rast.reified_facts, fb)

    def test_program_ast(self):
        "Test accessibility of program AST nodes."
        prog_str = "a."
        stms = []
        ast.parse_string(prog_str, stms.append)
        rast = ReifiedAST()
        rast.reify_string(prog_str)
        rast.reflect()
        self.assertListEqual(stms, rast.program_ast)
        reified = rast.reified_facts
        rast.reify_ast(stms)
        self.assertSetEqual(reified, rast.reified_facts)


class TestReifyReflect(TestReifiedAST):
    """Base class for tests for reification and reflection of
    non-ground programs."""

    base_str = "#program base.\n"

    def assertReifyEqual(
        self, file_name: str, reify_location: bool = False
    ):  # pylint: disable=invalid-name
        """Assert that reification of file_name under reflected_files
        results in file_name under reified_files."""
        reified_file = reified_files / file_name
        reflected_file = reflected_files / file_name
        reflected_str = reflected_file.read_text().strip()
        rast1 = ReifiedAST(reify_location=reify_location)
        rast1.reify_string(reflected_str)
        reified_facts = rast1.reified_facts
        rast2 = ReifiedAST()
        rast2.add_reified_files([reified_file])
        expected_facts = rast2.reified_facts
        self.assertSetEqual(reified_facts, expected_facts)  # type: ignore

    def assertReflectEqual(self, file_name: str):  # pylint: disable=invalid-name
        """Assert that reflection of file_name under reified_files
        results in file_name under reflected_files."""
        reified_file = reified_files / file_name
        reflected_file = reflected_files / file_name
        reflected_str = reflected_file.read_text().strip()
        rast = ReifiedAST()
        rast.add_reified_files([reified_file])
        rast.reflect()
        expected_string = self.base_str + reflected_str
        self.assertEqual(rast.program_string, expected_string)

    def assertReifyReflectEqual(self, file_name: str):  # pylint: disable=invalid-name
        """Assert that reification of prog_str results in ast_facts,
        and that reflection of ast_facts result in prog_str."""

        for operation in ["reification", "reflection"]:
            with self.subTest(operation=operation):
                if operation == "reification":
                    self.assertReifyEqual(file_name)
                elif operation == "reflection":
                    self.assertReflectEqual(file_name)


class TestReifyReflectNormalPrograms(TestReifyReflect):
    """Test cases for reification and reflection of disjunctive logic
    program AST nodes."""

    def test_rast_location(self):
        "Test reification of AST node locations"
        self.assertReifyEqual("location.lp", reify_location=True)

    def test_rast_prop_fact(self):
        "Test reification and reflection of a propositional fact."
        self.assertReifyReflectEqual("prop_fact.lp")

    def test_rast_prop_normal_rule(self):
        """Test reification and reflection of a normal rule containing
        only propositional atoms."""
        self.assertReifyReflectEqual("prop_normal_rule.lp")

    def test_rast_function(self):
        """Test reification and reflection of a variable-free normal
        rule with a symbolic atom."""
        self.assertReifyReflectEqual("atom.lp")

    def test_rast_external_function(self):
        """Test reification and reflection of an external function term."""
        self.assertReifyReflectEqual("external_function.lp")

    def test_rast_nested_function(self):
        "Test reification and reflection of a rule with a function term."
        self.assertReifyReflectEqual("function.lp")

    def test_rast_variable(self):
        "Test reification and reflection of normal rule with variable terms."
        self.assertReifyReflectEqual("variable.lp")

    def test_rast_string(self):
        "Test reification and reflection of normal rule with a string term."
        self.assertReifyReflectEqual("string.lp")

    def test_rast_constant_term(self):
        "Test reification and reflection of a normal rule with constant term."
        self.assertReifyReflectEqual("constant_term.lp")

    def test_rast_interval(self):
        "Test reification and reflection of a normal rule with an interval term."
        self.assertReifyReflectEqual("interval.lp")

    def test_rast_unary_operation(self):
        "Test reification and reflection of a normal rule with a unary operation."
        self.assertReifyReflectEqual("unary_operation.lp")

    def test_rast_binary_operation(self):
        "Test reification and reflection of a normal rule with a binary operation."
        self.assertReifyReflectEqual("binary_operation.lp")

    def test_rast_pool(self):
        "Test reification and reflection of a normal rule with a pool."
        self.assertReifyReflectEqual("pool.lp")

    def test_rast_comparison(self):
        "Test reification and reflection of a comparison operator"
        self.assertReifyReflectEqual("comparison.lp")

    def test_rast_boolean_constant(self):
        "Test reification and reflection of a boolean constant."
        self.assertReifyReflectEqual("bool_const.lp")

    def test_rast_integrity_constraint(self):
        "Test reification and reflection of an integrity constraint."
        self.assertReifyEqual("integrity_constraint.lp")

    def test_rast_conditional_literal(self):
        "Test reification and reflection of a conditional literal."
        self.assertReifyReflectEqual("conditional_literal.lp")

    def test_rast_disjunction(self):
        "Test reification and reflection of a disjunction."
        self.assertReifyReflectEqual("disjunction.lp")


class TestReifyReflectErrors(TestReifyReflect):
    "Test cases where reification or reflection should fail."

    def test_rast_node_failure(self):
        """Reification for any object not of type clingo.ast.AST or
        clingo.symbol.Symbol should raise a TypeError."""
        rast = ReifiedAST()
        not_node = {"not": "ast"}
        regex = "(?s).*AST or Symbol.*dict"
        with self.assertRaisesRegex(TypeError, expected_regex=regex):
            rast.reify_node(not_node)

    def test_child_query_error_one_expected_zero_found(self):
        """Reflection of a parent fact that expects a singe child fact
        but finds none should fail with an informative error message.

        """
        rast = ReifiedAST()
        rast.add_reified_files([malformed_reified_files / "one_expected_zero_found.lp"])
        regex = r"(?s).*atom\(5,function\(6\)\).*function\(6\).*found 0.*"
        with self.assertRaisesRegex(ChildQueryError, expected_regex=regex):
            rast.reflect()

    def test_child_query_error_one_expected_multiple_found(self):
        """Reflection of a parent fact that expects a singe child fact
        but finds multiple should fail with an informative error
        message.

        """
        rast = ReifiedAST()
        rast.add_reified_files(
            [malformed_reified_files / "one_expected_multiple_found.lp"]
        )
        regex = r"(?s).*atom\(5,function\(6\)\).*function\(6\).*found 2.*"
        with self.assertRaisesRegex(ChildQueryError, expected_regex=regex):
            rast.reflect()

    def test_children_query_error_multiple_in_same_position(self):
        """Refection of a child tuple with multiple elements in the
        same position should fail"""
        rast = ReifiedAST()
        rast.add_reified_files([malformed_reified_files / "multiple_in_same_pos.lp"])
        regex = (
            r"(?s).*comparison\(5,number\(6\),guards\(7\)\).*"
            r"multiple child facts in the same position.*guards\(7\)"
        )
        with self.assertRaisesRegex(ChildrenQueryError, expected_regex=regex):
            rast.reflect()

    def test_children_query_error_one_or_more_expected_zero_found(self):
        """Reflection of a child facts where one or more facts are expected
        should fail with an informative error message when zero are found."""
        rast = ReifiedAST()
        rast.add_reified_files(
            [malformed_reified_files / "one_or_more_expected_found_zero.lp"]
        )
        regex = (
            r"(?s).*comparison\(5,number\(6\),guards\(7\)\).*"
            r".*Expected 1 or more.*guards\(7\).*."
        )
        with self.assertRaisesRegex(ChildrenQueryError, expected_regex=regex):
            rast.reflect()

    def test_child_query_error_zero_or_one_expected_multiple_found(self):
        """Reflection of a child fact where zero or one fact is expected
        should fail with an informative error message when multiple are found."""
        rast = ReifiedAST()
        rast.add_reified_files(
            [malformed_reified_files / "zero_or_more_expected_multiple_found.lp"]
        )
        regex = (
            r"(?s).*aggregate\(4,guard\(5\),aggregate_elements\(8\),guard\(9\)\).*"
            r".*Expected 0 or 1.*guard\(5\).*found 2."
        )
        with self.assertRaisesRegex(ChildQueryError, expected_regex=regex):
            rast.reflect()


class TestReifyReflectAggTheory(TestReifyReflect):
    """Test cases for reification and reflection of aggregates and
    theory atoms."""

    def test_rast_aggregate(self):
        "Test reification and reflection of a simple count aggregate."
        self.assertReifyReflectEqual("aggregate.lp")
        self.setUp()
        self.assertReifyReflectEqual("aggregate2.lp")

    def test_rast_simple_theory_atom(self):
        """Test reification and reflection of simple theory atoms
        consisting only of symbolic part."""

        self.assertReifyReflectEqual("theory_atom_simple.lp")
        self.setUp()
        self.assertReifyReflectEqual("theory_atom_simple_arg.lp")
        self.setUp()
        self.assertReifyReflectEqual("theory_atom_simple_guard.lp")

    def test_rast_theory_sequence(self):
        "Test reification and reflection of a theory sequence."
        self.assertReifyReflectEqual("theory_sequence.lp")

    def test_rast_theory_sequence_type(self):
        "Test that all theory sequence types are accepted in reified representation."

    def test_rast_theory_function(self):
        "Test reification and reflection of a theory function."
        self.assertReifyReflectEqual("theory_function.lp")

    def test_rast_theory_term(self):
        "Test that all theory terms are accepted in reified representation."

    def test_rast_theory_unparsed_term(self):
        "Test reification and reflection of an unparsed theory term."
        self.assertReifyReflectEqual("theory_unparsed_term.lp")

    def test_rast_head_aggregate(self):
        "Test reification and reflection of a head aggregate."
        self.assertReifyReflectEqual("head_aggregate.lp")

    def test_rast_body_aggregate(self):
        "Test reification and reflection of a body aggregate."
        self.assertReifyReflectEqual("body_aggregate.lp")


class TestReifyReflectStatements(TestReifyReflect):
    """Test cases for reification and reflection of statements."""

    def test_rast_const_definition(self):
        "Test reification and reflections of a constant definition."
        self.assertReifyReflectEqual("definition.lp")

    def test_rast_show_signature(self):
        "Test reification and reflection of a show signature statement."
        self.assertReifyReflectEqual("show_signature.lp")

    def test_rast_defined(self):
        "Test reification and reflection of a defined statement."
        self.assertReifyReflectEqual("defined.lp")

    def test_rast_show_term(self):
        "Test reification and reflection of a show term statement."
        self.assertReifyReflectEqual("show_term.lp")

    def test_rast_minimize(self):
        "Test reification and reflection of a minimize statement."
        self.assertReifyReflectEqual("minimize.lp")

    def test_rast_script(self):
        "Test reification and reflection of an embedded script statement."
        self.assertReifyReflectEqual("script.lp")

    def test_rast_external_false(self):
        """Test reification and reflection of an external statement
        with default value false."""
        self.assertReifyReflectEqual("external.lp")

    def test_rast_edge(self):
        "Test reification and reflection of an edge statement."
        self.assertReifyReflectEqual("edge.lp")

    def test_rast_heuristic(self):
        "Test reification and reflection of a heuristic statement."
        self.assertReifyReflectEqual("heuristic.lp")

    def test_rast_project_atom(self):
        "Test reification and reflection of a project atom statement."
        self.assertReifyReflectEqual("project_atom.lp")

    def test_rast_project_signature(self):
        "Test reification and reflection of a project signature statement."
        self.assertReifyReflectEqual("project_signature.lp")

    def test_rast_program_params(self):
        "Test reification and reflection of a program statement with parameters."
        self.assertReifyReflectEqual("program_acid.lp")

    def test_rast_theory_definition(self):
        "Test reification and reflection of a theory definition."
        self.assertReifyReflectEqual("theory_definition.lp")
