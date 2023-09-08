# pylint: disable=too-many-lines
"""Module implementing reification and de-reification of non-ground programs"""
import inspect
import logging
import re
from contextlib import AbstractContextManager
from functools import singledispatchmethod
from pathlib import Path
from typing import Iterator, List, Literal, Optional, Sequence, Type, Union, overload

from clingo import Control, ast, symbol
from clingo.ast import (
    AST,
    ASTSequence,
    ASTType,
    Location,
    Position,
    parse_files,
    parse_string,
)
from clingo.symbol import Symbol, SymbolType
from clorm import (
    BaseField,
    FactBase,
    Predicate,
    Unifier,
    UnifierNoMatchError,
    control_add_facts,
    parse_fact_files,
    parse_fact_string,
)
from thefuzz import process  # type: ignore

import renopro.predicates as preds
from renopro.utils import assert_never
from renopro.utils.logger import get_clingo_logger_callback

logger = logging.getLogger(__name__)

DUMMY_LOC = Location(Position("<string>", 1, 1), Position("<string>", 1, 1))


class ChildQueryError(Exception):
    """Exception raised when a required child fact of an AST fact
    cannot be found.

    """


class ChildrenQueryError(Exception):
    """Exception raised when the expected number child facts of an AST
    fact cannot be found.

    """


class TryUnify(AbstractContextManager):
    """Context manager to try some operation that requires unification
    of some set of ast facts. Enhance error message if unification fails.
    """

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is UnifierNoMatchError:
            self.handle_unify_error(exc_value)

    @staticmethod
    def handle_unify_error(error):
        """Enhance UnifierNoMatchError with some more
        useful error messages to help debug the reason unification failed.

        """
        unmatched = error.symbol
        name2arity2pred = {
            pred.meta.name: {pred.meta.arity: pred} for pred in preds.AstPreds
        }
        candidate = name2arity2pred.get(unmatched.name, {}).get(
            len(unmatched.arguments)
        )
        if candidate is None:
            fuzzy_name = process.extractOne(unmatched.name, name2arity2pred.keys())[0]
            signatures = [
                f"{fuzzy_name}/{arity}." for arity in name2arity2pred[fuzzy_name]
            ]
            msg = f"""No AST fact of matching signature found for symbol
            '{unmatched}'.
            Similar AST fact signatures are:
            """ + "\n".join(
                signatures
            )
            raise UnifierNoMatchError(
                inspect.cleandoc(msg), unmatched, error.predicates
            ) from None
        for idx, arg in enumerate(unmatched.arguments):
            # This is very hacky. Should ask Dave for a better
            # solution, if there is one.
            arg_field = candidate[idx]._field  # pylint: disable=protected-access
            arg_field_str = re.sub(r"\(.*?\)", "", str(arg_field))
            try:
                arg_field.cltopy(arg)
            except (TypeError, ValueError):
                msg = f"""Cannot unify symbol
                '{unmatched}'
                to only candidate AST fact of matching signature
                {candidate.meta.name}/{candidate.meta.arity}
                due to failure to unify symbol's argument
                '{arg}'
                against the corresponding field
                '{arg_field_str}'."""
                raise UnifierNoMatchError(
                    inspect.cleandoc(msg), unmatched, (candidate,)
                ) from None
        raise RuntimeError("Code should be unreachable")  # nocoverage


def dispatch_on_node_type(meth):
    """ "Dispatch method on node.ast_type if node is of type AST
    or dispatch on node.type if node is of type Symbol.

    """
    registry = {}

    def dispatch(value):
        try:
            return registry[value]
        except KeyError:
            return meth

    def register(value, func=None):
        if func is None:
            return lambda f: register(value, f)

        registry[value] = func
        return func

    def wrapper(self, node, *args, **kw):
        if isinstance(node, AST):
            return dispatch(node.ast_type)(self, node, *args, **kw)
        if isinstance(node, Symbol):
            return dispatch(node.type)(self, node, *args, **kw)
        return dispatch(type(node))(self, node, *args, **kw)

    wrapper.register = register
    wrapper.dispatch = dispatch
    wrapper.registry = registry

    return wrapper


class ReifiedAST:
    """Class for converting between reified and non-reified
    representation of ASP programs.
    """

    def __init__(self):
        self._reified = FactBase()
        self._program_ast = []
        self._id_counter = -1
        self._statement_tup_id = None
        self._statement_pos = 0
        self._program_string = ""

    def add_reified_facts(self, reified_facts: Iterator[preds.AstPred]) -> None:
        """Add iterator of reified AST facts to internal factbase."""
        unifier = Unifier(preds.AstPreds)
        # couldn't find a way in clorm to directly add a set of facts
        # while checking unification, so we have to unify against the
        # underlying symbols
        with TryUnify():
            unified_facts = unifier.iter_unify(
                [fact.symbol for fact in reified_facts], raise_nomatch=True
            )
            self._reified.update(unified_facts)

    def add_reified_string(self, reified_string: str) -> None:
        """Add string of reified facts into internal factbase."""
        unifier = preds.AstPreds
        with TryUnify():
            facts = parse_fact_string(
                reified_string, unifier=unifier, raise_nomatch=True, raise_nonfact=True
            )
        self._reified.update(facts)

    def add_reified_files(self, reified_files: Sequence[Path]) -> None:
        """Add files containing reified facts into internal factbase."""
        reified_files_str = [str(f) for f in reified_files]
        with TryUnify():
            facts = parse_fact_files(
                reified_files_str,
                unifier=preds.AstPreds,
                raise_nomatch=True,
                raise_nonfact=True,
            )
        self._reified.update(facts)

    def reify_string(self, prog_str: str) -> None:
        """Reify input program string, adding reified facts to the
        internal factbase."""
        parse_string(prog_str, self.reify_node)

    def reify_files(self, files: Sequence[Path]) -> None:
        """Reify input program files, adding reified facts to the
        internal factbase."""
        for f in files:
            if not f.is_file():  # nocoverage
                raise IOError(f"File {f} does not exist.")
        files_str = [str(f) for f in files]
        parse_files(files_str, self.reify_node)

    @property
    def program_string(self) -> str:
        """String representation of reflected AST facts."""
        return self._program_string

    @property
    def program_ast(self) -> List[AST]:
        """AST nodes attained via reflection of AST facts."""
        return self._program_ast

    @property
    def reified_facts(self) -> FactBase:
        """Set of reified AST facts, encoding a non-ground ASP program."""
        return self._reified

    @property
    def reified_string(self) -> str:
        """String representation of reified AST facts, encoding a
        non-ground ASP program."""
        return self._reified.asp_str()

    @property
    def reified_string_doc(self) -> str:  # nocoverage
        """String representation of reified AST facts, encoding a
        non-ground ASP program, with comments describing the schema of
        occurring facts.

        """
        return self._reified.asp_str(commented=True)

    @dispatch_on_node_type
    def reify_node(self, node):
        """Reify the input ast node by adding it's clorm fact
        representation to the internal fact base, and recursively
        reify child nodes.

        """
        if hasattr(node, "ast_type"):  # nocoverage
            raise NotImplementedError(
                (
                    "Reification not implemented for AST nodes of type: "
                    f"{node.ast_type.name}."
                )
            )
        if hasattr(node, "type"):  # nocoverage
            raise NotImplementedError(
                (
                    "Reification not implemented for symbol of type: "
                    f"{node.typle.name}."
                )
            )
        raise TypeError(f"Nodes should be of type AST or Symbol, got: {type(node)}")

    def _reify_ast_seqence(
        self, seq: ASTSequence, tup_id: BaseField, tup_pred: Type[preds.AstPred]
    ):
        """Reify ast sequence into a tuple of predicates of type
        tup_pred with identifier tup_id."""
        reified_seq = [
            tup_pred(tup_id, pos, self.reify_node(item))
            for pos, item in enumerate(seq, start=0)
        ]
        self._reified.add(reified_seq)

    @reify_node.register(ASTType.SymbolicTerm)
    def _reify_symbolic_term(self, node):
        """Reify symbolic term.

        Note that the only possible child of a symbolic term is a
        clingo symbol denoting a number, variable, or constant, so we
        don't represent this ast node in our reification.

        """
        return self.reify_node(node.symbol)

    @reify_node.register(SymbolType.String)
    def _reify_symbol_string(self, symb):
        string1 = preds.String1()
        self._reified.add(preds.String(id=string1.id, value=symb.string))
        return string1

    @reify_node.register(SymbolType.Number)
    def _reify_symbol_number(self, symb):
        number1 = preds.Number1()
        self._reified.add(preds.Number(id=number1.id, value=symb.number))
        return number1

    @reify_node.register(ASTType.Variable)
    def _reify_variable(self, node):
        variable1 = preds.Variable1()
        self._reified.add(preds.Variable(id=variable1.id, name=node.name))
        return variable1

    @reify_node.register(ASTType.UnaryOperation)
    def _reify_unary_operation(self, node):
        clorm_operator = preds.convert_enum(
            ast.UnaryOperator(node.operator_type), preds.UnaryOperator
        )
        unop1 = preds.Unary_Operation1()
        unop = preds.Unary_Operation(
            id=unop1.id,
            operator=clorm_operator,
            argument=self.reify_node(node.argument),
        )
        self._reified.add(unop)
        return unop1

    @reify_node.register(ASTType.BinaryOperation)
    def _reify_binary_operation(self, node):
        clorm_operator = preds.convert_enum(
            ast.BinaryOperator(node.operator_type), preds.BinaryOperator
        )
        binop1 = preds.Binary_Operation1()
        binop = preds.Binary_Operation(
            id=binop1.id,
            operator=clorm_operator,
            left=self.reify_node(node.left),
            right=self.reify_node(node.right),
        )
        self._reified.add(binop)
        return binop1

    @reify_node.register(ASTType.Interval)
    def _reify_interval(self, node):
        interval1 = preds.Interval1()
        left = self.reify_node(node.left)
        right = self.reify_node(node.right)
        interval = preds.Interval(id=interval1.id, left=left, right=right)
        self._reified.add(interval)
        return interval1

    @reify_node.register(SymbolType.Function)
    def _reify_symbol_function(self, symb):
        """Reify constant term.

        Note that clingo represents constant terms as a
        clingo.Symbol.Function with empty argument list.

        """
        func1 = preds.Function1()
        self._reified.add(
            preds.Function(id=func1.id, name=symb.name, arguments=preds.Terms1())
        )
        return func1

    @reify_node.register(ASTType.Function)
    def _reify_function(self, node):
        """Reify an ast node with node.ast_type of ASTType.Function.

        Note that clingo's ast also represents propositional constants
        as nodes with node.type of ASTType.Function and an empty
        node.arguments list; thus some additional care must be taken
        to create the correct clorm predicate.

        """
        function1 = preds.Function1()
        function = preds.Function(
            id=function1.id, name=node.name, arguments=preds.Terms1()
        )
        self._reified.add(function)
        self._reify_ast_seqence(node.arguments, function.arguments.id, preds.Terms)
        return function1

    @reify_node.register(ASTType.Pool)
    def _reify_pool(self, node):
        pool1 = preds.Pool1()
        pool = preds.Pool(id=pool1.id, arguments=preds.Terms1())
        self._reified.add(pool)
        self._reify_ast_seqence(node.arguments, pool.arguments.id, preds.Terms)
        return pool1

    @reify_node.register(ASTType.TheorySequence)
    def _reify_theory_sequence(self, node):
        theory_seq1 = preds.Theory_Sequence1()
        clorm_theory_seq_type = preds.convert_enum(
            ast.TheorySequenceType(node.sequence_type), preds.TheorySequenceType
        )
        theory_seq = preds.Theory_Sequence(
            id=theory_seq1.id,
            sequence_type=clorm_theory_seq_type,
            terms=preds.Theory_Terms1(),
        )
        self._reified.add(theory_seq)
        self._reify_ast_seqence(node.terms, theory_seq.terms.id, preds.Theory_Terms)
        return theory_seq1

    @reify_node.register(ASTType.TheoryFunction)
    def _reify_theory_function(self, node):
        theory_func1 = preds.Theory_Function1()
        theory_func = preds.Theory_Function(
            id=theory_func1.id, name=node.name, arguments=preds.Theory_Terms1()
        )
        self._reified.add(theory_func)
        self._reify_ast_seqence(
            seq=node.arguments,
            tup_id=theory_func.arguments.id,
            tup_pred=preds.Theory_Terms,
        )
        return theory_func1

    @reify_node.register(ASTType.TheoryUnparsedTerm)
    def _reify_theory_unparsed_term(self, node):
        reified_unparsed_theory_term1 = preds.Theory_Unparsed_Term1()
        reified_unparsed_elements1 = preds.Theory_Unparsed_Term_Elements1()
        # reified_unparsed_theory_term = preds.Theory_Unparsed_Term(
        #     id=reified_unparsed_theory_term1.id,
        #     elements=preds.Theory_Unparsed_Term_Elements1(),
        # )
        for pos, element in enumerate(node.elements):
            operators = preds.Theory_Operators1()
            reified_operators = [
                preds.Theory_Operators(id=operators.id, position=p, operator=op)
                for p, op in enumerate(element.operators)
            ]
            self._reified.add(reified_operators)
            reified_theory_term1 = self.reify_node(element.term)
            reified_unparsed_elements = preds.Theory_Unparsed_Term_Elements(
                id=reified_unparsed_elements1.id,
                position=pos,
                operators=operators,
                term=reified_theory_term1,
            )
            self._reified.add(reified_unparsed_elements)
        reified_unparsed = preds.Theory_Unparsed_Term(
            id=reified_unparsed_theory_term1.id, elements=reified_unparsed_elements1
        )
        self._reified.add(reified_unparsed)
        return reified_unparsed_theory_term1

    @reify_node.register(ASTType.Guard)
    def _reify_guard(self, node):
        guard1 = preds.Guard1()
        clorm_operator = preds.convert_enum(
            ast.ComparisonOperator(node.comparison), preds.ComparisonOperator
        )
        guard = preds.Guard(
            id=guard1.id, comparison=clorm_operator, term=self.reify_node(node.term)
        )
        self._reified.add(guard)
        return guard1

    @reify_node.register(ASTType.Comparison)
    def _reify_comparison(self, node):
        comparison1 = preds.Comparison1()
        comparison = preds.Comparison(
            id=comparison1.id, term=self.reify_node(node.term), guards=preds.Guards1()
        )
        self._reified.add(comparison)
        self._reify_ast_seqence(node.guards, comparison.guards.id, preds.Guards)
        return comparison1

    @reify_node.register(ASTType.BooleanConstant)
    def _reify_boolean_constant(self, node):
        bool_const1 = preds.Boolean_Constant1()
        bool_str = ""
        if node.value == 1:
            bool_str = "true"
        elif node.value == 0:
            bool_str = "false"
        else:  # nocoverage
            raise RuntimeError("Code should be unreachable")
        bool_const = preds.Boolean_Constant(id=bool_const1.id, value=bool_str)
        self._reified.add(bool_const)
        return bool_const1

    @reify_node.register(ASTType.SymbolicAtom)
    def _reify_symbolic_atom(self, node):
        atom1 = preds.Symbolic_Atom1()
        atom = preds.Symbolic_Atom(id=atom1.id, symbol=self.reify_node(node.symbol))
        self._reified.add(atom)
        return atom1

    @reify_node.register(ASTType.Literal)
    def _reify_literal(self, node):
        clorm_sign = preds.convert_enum(ast.Sign(node.sign), preds.Sign)
        lit1 = preds.Literal1()
        lit = preds.Literal(id=lit1.id, sig=clorm_sign, atom=self.reify_node(node.atom))
        self._reified.add(lit)
        return lit1

    @reify_node.register(ASTType.ConditionalLiteral)
    def _reify_conditional_literal(self, node) -> preds.Conditional_Literal1:
        cond_lit1 = preds.Conditional_Literal1()
        cond_lit = preds.Conditional_Literal(
            id=cond_lit1.id,
            literal=self.reify_node(node.literal),
            condition=preds.Literals1(),
        )
        self._reified.add(cond_lit)
        self._reify_ast_seqence(node.condition, cond_lit.condition.id, preds.Literals)
        return cond_lit1

    @reify_node.register(ASTType.Aggregate)
    def _reify_aggregate(self, node) -> preds.Aggregate1:
        count_agg1 = preds.Aggregate1()
        left_guard = (
            preds.Guard1()
            if node.left_guard is None
            else self.reify_node(node.left_guard)
        )
        elements1 = preds.Agg_Elements1()
        self._reify_ast_seqence(node.elements, elements1.id, preds.Agg_Elements)
        right_guard = (
            preds.Guard1()
            if node.right_guard is None
            else self.reify_node(node.right_guard)
        )
        count_agg = preds.Aggregate(
            id=count_agg1.id,
            left_guard=left_guard,
            elements=elements1,
            right_guard=right_guard,
        )
        self._reified.add(count_agg)
        return count_agg1

    @reify_node.register(ASTType.TheoryAtom)
    def _reify_theory_atom(self, node) -> preds.Theory_Atom1:
        theory_atom1 = preds.Theory_Atom1()
        # we make a slight modification in the reified representation
        # vs the AST, wrapping the function into a symbolic atom, as
        # that's what it really is IMO.
        theory_symbolic_atom1 = self.reify_node(ast.SymbolicAtom(node.term))
        theory_atom_elements1 = preds.Theory_Atom_Elements1()
        reified_elements = []
        for pos, element in enumerate(node.elements):
            theory_terms1 = preds.Theory_Terms1()
            self._reify_ast_seqence(element.terms, theory_terms1.id, preds.Theory_Terms)
            literals1 = preds.Literals1()
            self._reify_ast_seqence(element.condition, literals1.id, preds.Literals)
            reified_element = preds.Theory_Atom_Elements(
                id=theory_atom_elements1.id,
                position=pos,
                terms=theory_terms1,
                condition=literals1,
            )
            reified_elements.append(reified_element)
        self._reified.add(reified_elements)
        theory_guard1 = preds.Theory_Guard1()
        if node.guard is not None:
            guard_theory_term = self.reify_node(node.guard.term)
            theory_guard = preds.Theory_Guard(
                id=theory_guard1.id,
                operator_name=node.guard.operator_name,
                term=guard_theory_term,
            )
            self._reified.add(theory_guard)
        theory_atom = preds.Theory_Atom(
            id=theory_atom1.id,
            atom=theory_symbolic_atom1,
            elements=theory_atom_elements1,
            guard=theory_guard1,
        )
        self._reified.add(theory_atom)
        return theory_atom1

    @reify_node.register(ASTType.BodyAggregate)
    def _reify_body_aggregate(self, node) -> preds.Body_Aggregate1:
        agg1 = preds.Body_Aggregate1()
        left_guard = (
            preds.Guard1()
            if node.left_guard is None
            else self.reify_node(node.left_guard)
        )
        clorm_agg_func = preds.convert_enum(
            ast.AggregateFunction(node.function), preds.AggregateFunction
        )
        elements1 = preds.Body_Agg_Elements1()
        reified_elements = []
        for pos, element in enumerate(node.elements):
            terms1 = preds.Terms1()
            self._reify_ast_seqence(element.terms, terms1.id, preds.Terms)
            literals1 = preds.Literals1()
            self._reify_ast_seqence(element.condition, literals1.id, preds.Literals)
            reified_element = preds.Body_Agg_Elements(
                id=elements1.id, position=pos, terms=terms1, condition=literals1
            )
            reified_elements.append(reified_element)
        self._reified.add(reified_elements)
        right_guard = (
            preds.Guard1()
            if node.right_guard is None
            else self.reify_node(node.right_guard)
        )
        agg = preds.Body_Aggregate(
            id=agg1.id,
            left_guard=left_guard,
            function=clorm_agg_func,
            elements=elements1,
            right_guard=right_guard,
        )
        self._reified.add(agg)
        return agg1

    def _reify_body_literals(self, body_lits: Sequence[ast.AST], body_id):
        reified_body_lits = []
        for pos, lit in enumerate(body_lits, start=0):
            if lit.ast_type is ast.ASTType.ConditionalLiteral:
                cond_lit1 = self.reify_node(lit)
                reified_body_lits.append(
                    preds.Body_Literals(
                        id=body_id, position=pos, body_literal=cond_lit1
                    )
                )
            else:
                body_lit1 = preds.Body_Literal1()
                reified_body_lits.append(
                    preds.Body_Literals(
                        id=body_id, position=pos, body_literal=body_lit1
                    )
                )
                clorm_sign = preds.convert_enum(ast.Sign(lit.sign), preds.Sign)
                body_lit = preds.Body_Literal(
                    id=body_lit1.id, sig=clorm_sign, atom=self.reify_node(lit.atom)
                )
                self._reified.add(body_lit)
        self._reified.add(reified_body_lits)

    @reify_node.register(ASTType.HeadAggregate)
    def _reify_head_aggregate(self, node) -> preds.Head_Aggregate1:
        agg1 = preds.Head_Aggregate1()
        left_guard = (
            preds.Guard1()
            if node.left_guard is None
            else self.reify_node(node.left_guard)
        )
        clorm_agg_func = preds.convert_enum(
            ast.AggregateFunction(node.function), preds.AggregateFunction
        )
        elements1 = preds.Head_Agg_Elements1()
        reified_elements = []
        for pos, element in enumerate(node.elements):
            terms1 = preds.Terms1()
            self._reify_ast_seqence(element.terms, terms1.id, preds.Terms)
            cond_lit1 = self.reify_node(element.condition)
            reified_element = preds.Head_Agg_Elements(
                id=elements1.id, position=pos, terms=terms1, condition=cond_lit1
            )
            reified_elements.append(reified_element)
        self._reified.add(reified_elements)
        right_guard = (
            preds.Guard1()
            if node.right_guard is None
            else self.reify_node(node.right_guard)
        )
        agg = preds.Head_Aggregate(
            id=agg1.id,
            left_guard=left_guard,
            function=clorm_agg_func,
            elements=elements1,
            right_guard=right_guard,
        )
        self._reified.add(agg)
        return agg1

    @reify_node.register(ASTType.Disjunction)
    def _reify_disjunction(self, node) -> preds.Disjunction1:
        disj1 = preds.Disjunction1()
        cond_lits1 = preds.Conditional_Literals1()
        self._reify_ast_seqence(
            node.elements, cond_lits1.id, preds.Conditional_Literals
        )
        disj = preds.Disjunction(id=disj1.id, elements=cond_lits1)
        self._reified.add(disj)
        return disj1

    @reify_node.register(ASTType.Rule)
    def _reify_rule(self, node):
        rule1 = preds.Rule1()
        head = self.reify_node(node.head)
        rule = preds.Rule(id=rule1.id, head=head, body=preds.Body_Literals1())
        self._reified.add(rule)
        statement_tup = preds.Statements(
            id=self._statement_tup_id, position=self._statement_pos, statement=rule1
        )
        self._reified.add(statement_tup)
        self._statement_pos += 1
        self._reify_body_literals(node.body, rule.body.id)

    @reify_node.register(ASTType.Program)
    def _reify_program(self, node):
        const_tup_id = preds.Constants1()
        for pos, param in enumerate(node.parameters, start=0):
            const_tup = preds.Constants(
                id=const_tup_id.id, position=pos, constant=preds.Function1()
            )
            const = preds.Function(
                id=const_tup.constant.id, name=param.name, arguments=preds.Terms1()
            )
            self._reified.add([const_tup, const])
        program = preds.Program(
            name=node.name, parameters=const_tup_id, statements=preds.Statements1()
        )
        self._reified.add(program)
        self._statement_tup_id = program.statements.id
        self._statement_pos = 0

    @reify_node.register(ASTType.External)
    def _reify_external(self, node):
        ext_type = node.external_type.symbol.name
        external1 = preds.External1()
        external = preds.External(
            id=external1.id,
            atom=self.reify_node(node.atom),
            body=preds.Body_Literals1(),
            external_type=ext_type,
        )
        self._reified.add(external)
        statement_tup = preds.Statements(
            id=self._statement_tup_id, position=self._statement_pos, statement=external1
        )
        self._reified.add(statement_tup)
        self._statement_pos += 1
        self._reify_body_literals(node.body, external.body.id)

    ExpectedNum = Literal["1", "?", "+", "*"]

    @overload
    def _reflect_child(
        self,
        parent_fact: preds.AstPred,
        child_id_fact,
        expected_children_num: Literal["1"],
    ) -> AST:  # nocoverage
        ...

    # for handling the default argument "1"

    @overload
    def _reflect_child(
        self, parent_fact: preds.AstPred, child_id_fact
    ) -> AST:  # nocoverage
        ...

    @overload
    def _reflect_child(
        self,
        parent_fact: preds.AstPred,
        child_id_fact,
        expected_children_num: Literal["?"],
    ) -> Optional[AST]:  # nocoverage
        ...

    @overload
    def _reflect_child(
        self,
        parent_fact: preds.AstPred,
        child_id_fact,
        expected_children_num: Literal["*", "+"],
    ) -> Sequence[AST]:  # nocoverage
        ...

    def _reflect_child(
        self,
        parent_fact: preds.AstPred,
        child_id_fact,
        expected_children_num: ExpectedNum = "1",
    ) -> Union[None, AST, Sequence[AST]]:
        """Utility function that takes a unary ast predicate
        identifying a child predicate, queries reified factbase for
        child predicate, and returns the child node obtained by
        reflecting the child predicate.

        """
        identifier = child_id_fact.id
        child_ast_pred = getattr(preds, type(child_id_fact).__name__.rstrip("1"))
        query = self._reified.query(child_ast_pred).where(
            child_ast_pred.id == identifier
        )
        base_msg = f"Error finding child fact of fact '{parent_fact}':\n"
        if expected_children_num == "1":
            child_facts = list(query.all())
            num_child_facts = len(child_facts)
            if num_child_facts == 1:
                return self.reflect_predicate(child_facts[0])
            else:
                msg = (
                    f"Expected 1 child fact for identifier '{child_id_fact}'"
                    f", found {num_child_facts}."
                )
                raise ChildQueryError(base_msg + msg)
        elif expected_children_num == "?":
            child_facts = list(query.all())
            num_child_facts = len(child_facts)
            if num_child_facts == 0:
                return None
            elif num_child_facts == 1:
                return self.reflect_predicate(child_facts[0])
            else:
                msg = (
                    f"Expected 0 or 1 child fact for identifier "
                    f"'{child_id_fact}', found {num_child_facts}."
                )
                raise ChildQueryError(base_msg + msg)
        elif expected_children_num == "*" or expected_children_num == "+":
            query = query.order_by(child_ast_pred.position)
            child_facts = list(query.all())
            num_child_facts = len(child_facts)
            # check that there are no tuple elements in the same position
            if num_child_facts != len(set(tup.position for tup in child_facts)):
                msg = (
                    "Found multiple child facts in the same position for "
                    f"identifier '{child_id_fact}'."
                )
                raise ChildrenQueryError(base_msg + msg)
            if expected_children_num == "+" and num_child_facts == 0:
                msg = (
                    f"Expected 1 or more child facts for identifier "
                    f"'{child_id_fact}', found 0."
                )
                raise ChildrenQueryError(base_msg + msg)
            child_nodes = [self.reflect_predicate(fact) for fact in child_facts]
            return child_nodes
        assert_never(expected_children_num)

    @singledispatchmethod
    def reflect_predicate(self, pred: preds.AstPred):  # nocoverage
        """Convert the input AST element's reified fact representation
        back into a the corresponding member of clingo's abstract
        syntax tree, recursively reflecting all child facts.

        """
        raise NotImplementedError(
            f"Reflection not implemented for predicate of type {type(pred)}."
        )

    @reflect_predicate.register
    def _reflect_string(self, string: preds.String) -> AST:
        """Reflect a String fact into a String symbol wrapped in SymbolicTerm."""
        return ast.SymbolicTerm(
            location=DUMMY_LOC, symbol=symbol.String(string=str(string.value))
        )

    @reflect_predicate.register
    def _reflect_number(self, number: preds.Number) -> AST:
        """Reflect a Number fact into a Number node."""
        return ast.SymbolicTerm(
            location=DUMMY_LOC,
            symbol=symbol.Number(number=number.value),  # type: ignore
        )

    @reflect_predicate.register
    def _reflect_variable(self, var: preds.Variable) -> AST:
        """Reflect a Variable fact into a Variable node."""
        return ast.Variable(location=DUMMY_LOC, name=str(var.name))

    @reflect_predicate.register
    def _reflect_unary_operation(self, operation: preds.Unary_Operation) -> AST:
        """Reflect a Unary_Operation fact into a UnaryOperation node."""
        clingo_operator = preds.convert_enum(
            preds.UnaryOperator(operation.operator), ast.UnaryOperator
        )
        return ast.UnaryOperation(
            location=DUMMY_LOC,
            operator_type=clingo_operator,
            argument=self._reflect_child(operation, operation.argument),
        )

    @reflect_predicate.register
    def _reflect_binary_operation(self, operation: preds.Binary_Operation) -> AST:
        """Reflect a Binary_Operation fact into a BinaryOperation node."""
        clingo_operator = preds.convert_enum(
            preds.BinaryOperator(operation.operator), ast.BinaryOperator
        )
        reflected_left = self._reflect_child(operation, operation.left)
        reflected_right = self._reflect_child(operation, operation.right)
        return ast.BinaryOperation(
            location=DUMMY_LOC,
            operator_type=clingo_operator,
            left=reflected_left,
            right=reflected_right,
        )

    @reflect_predicate.register
    def _reflect_interval(self, interval: preds.Interval) -> AST:
        reflected_left = self._reflect_child(interval, interval.left)
        reflected_right = self._reflect_child(interval, interval.right)
        return ast.Interval(
            location=DUMMY_LOC, left=reflected_left, right=reflected_right
        )

    @reflect_predicate.register
    def _reflect_terms(self, terms: preds.Terms) -> AST:
        return self._reflect_child(terms, terms.term)

    @reflect_predicate.register
    def _reflect_function(self, func: preds.Function) -> AST:
        """Reflect a Function fact into a Function node.

        Note that a Function fact is used to represent a propositional
        constant, predicate, function symbol, or constant term. All of
        these can be validly represented by a Function node in the
        clingo AST and so we can return a Function node in each case.
        Constant terms are parsed a Symbol by the parser, thus we need
        to handle them differently when reifying.

        """
        arg_nodes = self._reflect_child(func, func.arguments, "*")
        return ast.Function(
            location=DUMMY_LOC, name=str(func.name), arguments=arg_nodes, external=0
        )

    @reflect_predicate.register
    def _reflect_pool(self, pool: preds.Pool) -> AST:
        arg_nodes = self._reflect_child(pool, pool.arguments, "*")
        return ast.Pool(location=DUMMY_LOC, arguments=arg_nodes)

    @reflect_predicate.register
    def _reflect_theory_terms(self, theory_terms: preds.Theory_Terms) -> AST:
        return self._reflect_child(theory_terms, theory_terms.theory_term)

    @reflect_predicate.register
    def _reflect_theory_sequence(self, theory_seq: preds.Theory_Sequence) -> AST:
        clingo_theory_sequence_type = preds.convert_enum(
            preds.TheorySequenceType(theory_seq.sequence_type), ast.TheorySequenceType
        )
        theory_term_nodes = self._reflect_child(theory_seq, theory_seq.terms, "*")
        return ast.TheorySequence(
            location=DUMMY_LOC,
            sequence_type=clingo_theory_sequence_type,
            terms=theory_term_nodes,
        )

    @reflect_predicate.register
    def _reflect_theory_function(self, theory_func: preds.Theory_Function) -> AST:
        arguments = self._reflect_child(theory_func, theory_func.arguments, "*")
        return ast.TheoryFunction(
            location=DUMMY_LOC, name=str(theory_func.name), arguments=arguments
        )

    @reflect_predicate.register
    def _reflect_theory_operators(
        self, theory_operators: preds.Theory_Operators
    ) -> str:
        return str(theory_operators.operator)

    @reflect_predicate.register
    def _reflect_theory_unparsed_term_elements(
        self, elements: preds.Theory_Unparsed_Term_Elements
    ) -> AST:
        reflected_operators = self._reflect_child(elements, elements.operators, "*")
        reflected_term = self._reflect_child(elements, elements.term)
        return ast.TheoryUnparsedTermElement(
            operators=reflected_operators, term=reflected_term
        )

    @reflect_predicate.register
    def _reflect_theory_unparsed_term(
        self, theory_unparsed_term: preds.Theory_Unparsed_Term
    ) -> AST:
        reflected_elements = self._reflect_child(
            theory_unparsed_term, theory_unparsed_term.elements, "*"
        )
        return ast.TheoryUnparsedTerm(location=DUMMY_LOC, elements=reflected_elements)

    @reflect_predicate.register
    def _reflect_guard(self, guard: preds.Guard) -> AST:
        clingo_operator = preds.convert_enum(
            preds.ComparisonOperator(guard.comparison), ast.ComparisonOperator
        )
        reflected_guard = self._reflect_child(guard, guard.term)
        return ast.Guard(comparison=clingo_operator, term=reflected_guard)

    @reflect_predicate.register
    def _reflect_guards(self, guards: preds.Guards) -> AST:
        return self._reflect_child(guards, guards.guard)

    @reflect_predicate.register
    def _reflect_comparison(self, comparison: preds.Comparison) -> AST:
        term_node = self._reflect_child(comparison, comparison.term)
        guard_nodes = self._reflect_child(comparison, comparison.guards, "+")
        return ast.Comparison(term=term_node, guards=guard_nodes)

    @reflect_predicate.register
    def _reflect_boolean_constant(self, bool_const: preds.Boolean_Constant) -> AST:
        bool_const_term = bool_const.value
        if bool_const_term == "true":
            b = 1
        elif bool_const_term == "false":
            b = 0
        else:  # nocoverage
            raise RuntimeError("Code should be unreachable")
        return ast.BooleanConstant(value=b)

    @reflect_predicate.register
    def _reflect_symbolic_atom(self, atom: preds.Symbolic_Atom) -> AST:
        reflected_symbol = self._reflect_child(atom, atom.symbol)
        return ast.SymbolicAtom(symbol=reflected_symbol)

    @reflect_predicate.register
    def _reflect_literal(self, lit: preds.Literal) -> AST:
        clingo_sign = preds.convert_enum(preds.Sign(lit.sig), ast.Sign)
        reflected_atom = self._reflect_child(lit, lit.atom)
        return ast.Literal(location=DUMMY_LOC, sign=clingo_sign, atom=reflected_atom)

    @reflect_predicate.register
    def _reflect_literals(self, literals: preds.Literals) -> AST:
        return self._reflect_child(literals, literals.literal)

    @reflect_predicate.register
    def _reflect_conditional_literal(self, cond_lit: preds.Conditional_Literal) -> AST:
        reflected_literal = self._reflect_child(cond_lit, cond_lit.literal)
        reflected_condition = self._reflect_child(cond_lit, cond_lit.condition, "*")
        return ast.ConditionalLiteral(
            location=DUMMY_LOC, literal=reflected_literal, condition=reflected_condition
        )

    @reflect_predicate.register
    def _reflect_agg_elements(self, agg_elements: preds.Agg_Elements) -> AST:
        return self._reflect_child(agg_elements, agg_elements.element)

    @reflect_predicate.register
    def _reflect_aggregate(self, aggregate: preds.Aggregate) -> AST:
        reflected_left_guard = self._reflect_child(aggregate, aggregate.left_guard, "?")
        reflected_right_guard = self._reflect_child(
            aggregate, aggregate.right_guard, "?"
        )
        reflected_elements = self._reflect_child(aggregate, aggregate.elements, "*")
        return ast.Aggregate(
            location=DUMMY_LOC,
            left_guard=reflected_left_guard,
            elements=reflected_elements,
            right_guard=reflected_right_guard,
        )

    @reflect_predicate.register
    def _reflect_theory_guard(self, theory_guard: preds.Theory_Guard) -> AST:
        reflected_operator_name = theory_guard.operator_name
        reflected_theory_term = self._reflect_child(theory_guard, theory_guard.term)
        return ast.TheoryGuard(
            operator_name=str(reflected_operator_name), term=reflected_theory_term
        )

    @reflect_predicate.register
    def _reflect_theory_atom_elements(
        self, elements: preds.Theory_Atom_Elements
    ) -> AST:
        reflected_terms = self._reflect_child(elements, elements.terms, "*")
        reflected_condition = self._reflect_child(elements, elements.condition, "*")
        return ast.TheoryAtomElement(
            terms=reflected_terms, condition=reflected_condition
        )

    @reflect_predicate.register
    def _reflect_theory_atom(self, theory_atom: preds.Theory_Atom) -> AST:
        reflected_syb_atom = self._reflect_child(theory_atom, theory_atom.atom)
        reflected_elements = self._reflect_child(theory_atom, theory_atom.elements, "*")
        reflected_guard = self._reflect_child(theory_atom, theory_atom.guard, "?")
        return ast.TheoryAtom(
            location=DUMMY_LOC,
            term=reflected_syb_atom.symbol,
            elements=reflected_elements,
            guard=reflected_guard,
        )

    @reflect_predicate.register
    def _reflect_body_agg_elements(self, elements: preds.Body_Agg_Elements) -> AST:
        reflected_terms = self._reflect_child(elements, elements.terms, "*")
        reflected_condition = self._reflect_child(elements, elements.condition, "*")
        return ast.BodyAggregateElement(
            terms=reflected_terms, condition=reflected_condition
        )

    @reflect_predicate.register
    def _reflect_body_aggregate(self, aggregate: preds.Body_Aggregate) -> AST:
        reflected_left_guard = self._reflect_child(aggregate, aggregate.left_guard, "?")
        reflected_agg_function = preds.convert_enum(
            preds.AggregateFunction(aggregate.function), ast.AggregateFunction
        )
        reflected_elements = self._reflect_child(aggregate, aggregate.elements, "*")
        reflected_right_guard = self._reflect_child(
            aggregate, aggregate.right_guard, "?"
        )
        return ast.BodyAggregate(
            location=DUMMY_LOC,
            left_guard=reflected_left_guard,
            function=reflected_agg_function,
            elements=reflected_elements,
            right_guard=reflected_right_guard,
        )

    @reflect_predicate.register
    def _reflect_body_literals(self, body_literals: preds.Body_Literals) -> AST:
        return self._reflect_child(body_literals, body_literals.body_literal)

    @reflect_predicate.register
    def _reflect_body_literal(self, body_lit: preds.Body_Literal) -> AST:
        return self._reflect_literal(body_lit)

    @reflect_predicate.register
    def _reflect_head_agg_elements(self, elements: preds.Head_Agg_Elements) -> AST:
        reflected_terms = self._reflect_child(elements, elements.terms, "*")
        reflected_condition = self._reflect_child(elements, elements.condition)
        return ast.HeadAggregateElement(
            terms=reflected_terms, condition=reflected_condition
        )

    @reflect_predicate.register
    def _reflect_head_aggregate(self, aggregate: preds.Head_Aggregate) -> AST:
        reflected_left_guard = self._reflect_child(aggregate, aggregate.left_guard, "?")
        reflected_agg_function = preds.convert_enum(
            preds.AggregateFunction(aggregate.function), ast.AggregateFunction
        )
        reflected_elements = self._reflect_child(aggregate, aggregate.elements, "*")
        reflected_right_guard = self._reflect_child(
            aggregate, aggregate.right_guard, "?"
        )
        return ast.HeadAggregate(
            location=DUMMY_LOC,
            left_guard=reflected_left_guard,
            function=reflected_agg_function,
            elements=reflected_elements,
            right_guard=reflected_right_guard,
        )

    @reflect_predicate.register
    def _reflect_conditional_literals(
        self, cond_lits: preds.Conditional_Literals
    ) -> AST:
        return self._reflect_child(cond_lits, cond_lits.conditional_literal)

    @reflect_predicate.register
    def _reflect_disjunction(self, disjunction: preds.Disjunction) -> AST:
        reflected_elements = self._reflect_child(disjunction, disjunction.elements, "*")
        return ast.Disjunction(location=DUMMY_LOC, elements=reflected_elements)

    @reflect_predicate.register
    def _reflect_rule(self, rule: preds.Rule) -> AST:
        """Reflect a Rule fact into a Rule node."""
        reflected_head = self._reflect_child(rule, rule.head)
        reflected_body = self._reflect_child(rule, rule.body, "*")
        return ast.Rule(location=DUMMY_LOC, head=reflected_head, body=reflected_body)

    @reflect_predicate.register
    def _reflect_statements(self, statements: preds.Statements) -> AST:
        return self._reflect_child(statements, statements.statement)

    @reflect_predicate.register
    def _reflect_constants(self, constants: preds.Constants) -> AST:
        return self._reflect_child(constants, constants.constant)

    @reflect_predicate.register
    def _reflect_program(self, program: preds.Program) -> Sequence[AST]:
        """Reflect a (sub)program fact into sequence of AST nodes, one
        node per each statement in the (sub)program.

        """
        subprogram = []
        parameter_nodes = self._reflect_child(program, program.parameters, "*")
        subprogram.append(
            ast.Program(
                location=DUMMY_LOC, name=str(program.name), parameters=parameter_nodes
            )
        )
        statement_nodes = self._reflect_child(program, program.statements, "*")
        subprogram.extend(statement_nodes)
        return subprogram

    @reflect_predicate.register
    def _reflect_external(self, external: preds.External) -> AST:
        """Reflect an External fact into an External node."""
        symb_atom_node = self._reflect_child(external, external.atom)
        body_nodes = self._reflect_child(external, external.body, "*")
        ext_type = ast.SymbolicTerm(
            location=DUMMY_LOC,
            symbol=symbol.Function(name=str(external.external_type), arguments=[]),
        )
        return ast.External(
            location=DUMMY_LOC,
            atom=symb_atom_node,
            body=body_nodes,
            external_type=ext_type,
        )

    def reflect(self):
        """Convert stored reified ast facts into a (sequence of) AST
        node(s), and it's string representation.

        """
        # reset list of program statements before population via reflect
        self._program_ast = []
        # should probably define an order in which programs are queried
        for prog in self._reified.query(preds.Program).all():
            subprogram = self.reflect_predicate(prog)
            self._program_ast.extend(subprogram)
        self._program_string = "\n".join(
            [str(statement) for statement in self._program_ast]
        )
        logger.debug("Reflected program string:\n%s", self.program_string)

    def transform(
        self,
        meta_str: Optional[str] = None,
        meta_files: Optional[Sequence[Path]] = None,
        clingo_options: Optional[Sequence[str]] = None,
    ) -> None:
        """Transform the reified AST using meta encoding.

        Parameter meta_prog may be a string path to file containing
        meta-encoding, or the meta-encoding in string form.

        """
        if len(self._reified) == 0:
            logger.warning("Reified AST to be transformed is empty.")
        if meta_str is None and meta_files is None:
            raise ValueError("No meta-program provided for transformation.")
        meta_prog = ""
        if meta_str is not None:
            meta_prog += meta_str
        if meta_files is not None:
            for meta_file in meta_files:
                with meta_file.open() as f:
                    meta_prog += f.read()
        logger.debug(
            "Applying transformation defined in following meta-encoding:\n%s", meta_prog
        )
        clingo_logger = get_clingo_logger_callback(logger)
        clingo_options = [] if clingo_options is None else clingo_options
        ctl = Control(clingo_options, logger=clingo_logger)
        logger.debug(
            "Reified facts before applying transformation:\n%s", self.reified_string
        )
        control_add_facts(ctl, self._reified)
        ctl.add(meta_prog)
        ctl.load("./src/renopro/asp/transform.lp")
        ctl.ground()
        with ctl.solve(yield_=True) as handle:  # type: ignore
            model_iterator = iter(handle)
            model = next(model_iterator)
            ast_symbols = [final.arguments[0] for final in model.symbols(shown=True)]
            unifier = Unifier(preds.AstPreds)
            with TryUnify():
                ast_facts = unifier.iter_unify(ast_symbols, raise_nomatch=True)
                self._reified = FactBase(ast_facts)
            try:
                next(model_iterator)
            except StopIteration:
                pass
            else:  # nocoverage
                logger.warning(
                    (
                        "Transformation encoding produced multiple models, "
                        "ignoring additional ones."
                    )
                )
        logger.debug(
            "Reified facts after applying transformation:\n%s", self.reified_string
        )


if __name__ == "__main__":  # nocoverage
    pass

#  LocalWords:  nocoverage
