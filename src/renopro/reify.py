"""Module implementing reification and de-reification of non-ground programs"""
import inspect
import logging
import re
from contextlib import AbstractContextManager
from functools import singledispatchmethod
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Type

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
    of some set of ast facts. Enhance error message if unification fails"""

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
            pred.meta.name: {pred.meta.arity: pred} for pred in preds.AstPredicates
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
                inspect.cleandoc(msg),
                unmatched,
                error.predicates,
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
                    inspect.cleandoc(msg),
                    unmatched,
                    (candidate,),
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
    representation of ASP programs."""

    def __init__(self):
        self._reified = FactBase()
        self._program_ast = []
        self._id_counter = -1
        self._statement_tup_id = None
        self._statement_pos = 0
        self._program_string = ""

    def add_reified_facts(self, reified_facts: Iterator[preds.AstPredicate]) -> None:
        """Add iterator of reified AST facts to internal factbase."""
        unifier = Unifier(preds.AstPredicates)
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
        unifier = preds.AstPredicates
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
                unifier=preds.AstPredicates,
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
                    "f{node.ast_type.name}."
                )
            )
        if hasattr(node, "type"):  # nocoverage
            raise NotImplementedError(
                (
                    "Reification not implemented for symbol of type: "
                    "f{node.typle.name}."
                )
            )
        raise TypeError(f"Nodes should be of type AST or Symbol, got: {type(node)}")

    @reify_node.register(ASTType.Program)
    def _reify_program(self, node):
        const_tup_id = preds.Constant_Tuple1()
        for pos, param in enumerate(node.parameters, start=0):
            const_tup = preds.Constant_Tuple(
                id=const_tup_id.id, position=pos, element=preds.Function1()
            )
            const = preds.Function(
                id=const_tup.element.id, name=param.name, arguments=preds.Term_Tuple1()
            )
            self._reified.add([const_tup, const])
        program = preds.Program(
            name=node.name,
            parameters=const_tup_id,
            statements=preds.Statement_Tuple1(),
        )
        self._reified.add(program)
        self._statement_tup_id = program.statements.id
        self._statement_pos = 0

    def _reify_ast_seqence(
        self,
        seq: ASTSequence,
        tup_id: BaseField,
        tup_pred: Type[Predicate],
    ):
        """Reify ast sequence into a tuple of predicates of type
        tup_pred with identifier tup_id."""
        for pos, item in enumerate(seq, start=0):
            self._reified.add(
                tup_pred(id=tup_id, position=pos, element=self.reify_node(item))
            )

    @reify_node.register(ASTType.External)
    def _reify_external(self, node):
        ext_type = node.external_type.symbol.name
        external1 = preds.External1()
        external = preds.External(
            id=external1.id,
            atom=self.reify_node(node.atom),
            body=preds.Body_Literal_Tuple1(),
            external_type=ext_type,
        )
        self._reified.add(external)
        statement_tup = preds.Statement_Tuple(
            id=self._statement_tup_id, position=self._statement_pos, element=external1
        )
        self._reified.add(statement_tup)
        self._statement_pos += 1
        self._reify_ast_seqence(node.body, external.body.id, preds.Body_Literal_Tuple)

    @reify_node.register(ASTType.Rule)
    def _reify_rule(self, node):
        rule1 = preds.Rule1()

        # assumption: head can only be a Literal
        head = self.reify_node(node.head)
        rule = preds.Rule(id=rule1.id, head=head, body=preds.Body_Literal_Tuple1())
        self._reified.add(rule)
        statement_tup = preds.Statement_Tuple(
            id=self._statement_tup_id, position=self._statement_pos, element=rule1
        )
        self._reified.add(statement_tup)
        self._statement_pos += 1
        self._reify_ast_seqence(node.body, rule.body.id, preds.Body_Literal_Tuple)

    @reify_node.register(ASTType.ConditionalLiteral)
    def _reify_conditional_literal(self, node) -> preds.Conditional_Literal1:
        cond_lit1 = preds.Conditional_Literal1()
        cond_lit = preds.Conditional_Literal(
            id=cond_lit1.id,
            literal=self.reify_node(node.literal),
            condition=preds.Literal_Tuple1(),
        )
        self._reified.add(cond_lit)
        self._reify_ast_seqence(
            node.condition, cond_lit.condition.id, preds.Literal_Tuple
        )
        return cond_lit1

    @reify_node.register(ASTType.Literal)
    def _reify_literal(self, node):
        lit1 = preds.Literal1()
        clorm_sign = preds.sign_ast2cl[node.sign]
        lit = preds.Literal(id=lit1.id, sig=clorm_sign, atom=self.reify_node(node.atom))
        self._reified.add(lit)
        return lit1

    @reify_node.register(ASTType.SymbolicAtom)
    def _reify_symbolic_atom(self, node):
        atom1 = preds.Symbolic_Atom1()
        atom = preds.Symbolic_Atom(id=atom1.id, symbol=self.reify_node(node.symbol))
        self._reified.add(atom)
        return atom1

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

    @reify_node.register(ASTType.Comparison)
    def _reify_comparison(self, node):
        comparison1 = preds.Comparison1()
        comparison = preds.Comparison(
            id=comparison1.id,
            term=self.reify_node(node.term),
            guards=preds.Guard_Tuple1(),
        )
        self._reified.add(comparison)
        for pos, guard in enumerate(node.guards, start=0):
            self._reified.add(
                preds.Guard_Tuple(
                    id=comparison.guards.id,
                    position=pos,
                    comparison=preds.comp_operator_ast2cl[guard.comparison],
                    term=self.reify_node(guard.term),
                )
            )
        return comparison1

    @reify_node.register(ASTType.Interval)
    def _reify_interval(self, node):
        interval1 = preds.Interval1()
        left = self.reify_node(node.left)
        right = self.reify_node(node.right)
        interval = preds.Interval(id=interval1.id, left=left, right=right)
        self._reified.add(interval)
        return interval1

    @reify_node.register(ASTType.BinaryOperation)
    def _reify_binary_operation(self, node):
        clorm_operator = preds.binary_operator_ast2cl[node.operator_type]
        binop1 = preds.Binary_Operation1()
        binop = preds.Binary_Operation(
            id=binop1.id,
            operator=clorm_operator,
            left=self.reify_node(node.left),
            right=self.reify_node(node.right),
        )
        self._reified.add(binop)
        return binop1

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
            id=function1.id,
            name=node.name,
            arguments=preds.Term_Tuple1(),
        )
        self._reified.add(function)
        self._reify_ast_seqence(node.arguments, function.arguments.id, preds.Term_Tuple)
        return function1

    @reify_node.register(ASTType.Variable)
    def _reify_variable(self, node):
        variable1 = preds.Variable1()
        self._reified.add(preds.Variable(id=variable1.id, name=node.name))
        return variable1

    @reify_node.register(ASTType.SymbolicTerm)
    def _reify_symbolic_term(self, node):
        """Reify symbolic term.

        Note that the only possible child of a symbolic term is a
        clingo symbol denoting a number, variable, or constant, so we
        don't represent this ast node in our reification.

        """
        return self.reify_node(node.symbol)

    @reify_node.register(SymbolType.Number)
    def _reify_symbol_number(self, symb):
        number1 = preds.Number1()
        self._reified.add(preds.Number(id=number1.id, value=symb.number))
        return number1

    @reify_node.register(SymbolType.Function)
    def _reify_symbol_function(self, symb):
        """Reify constant term.

        Note that clingo represents constant terms as a
        clingo.Symbol.Function with empty argument list.

        """
        func1 = preds.Function1()
        self._reified.add(
            preds.Function(id=func1.id, name=symb.name, arguments=preds.Term_Tuple1())
        )
        return func1

    @reify_node.register(SymbolType.String)
    def _reify_symbol_string(self, symb):
        string1 = preds.String1()
        self._reified.add(preds.String(id=string1.id, value=symb.string))
        return string1

    def _reflect_child_pred(self, parent_fact, child_id_fact):
        """Utility function that takes a unary ast predicate
        containing only an identifier pointing to a child predicate,
        queries reified factbase for child predicate, and returns the
        child node obtained by reflecting the child predicate.

        """
        identifier = child_id_fact.id
        child_ast_pred = getattr(preds, type(child_id_fact).__name__.rstrip("1"))
        query = self._reified.query(child_ast_pred).where(
            child_ast_pred.id == identifier
        )
        child_preds = list(query.all())
        if len(child_preds) == 0:
            msg = (
                f"Error finding child fact of fact '{parent_fact}':\n"
                f"Expected single child fact for identifier '{child_id_fact}'"
                ", found none."
            )
            raise ChildQueryError(msg)
        if len(child_preds) > 1:
            child_pred_strings = [str(pred) for pred in child_preds]
            msg = (
                f"Error finding child fact of fact '{parent_fact}':\n"
                f"Expected single child fact for identifier '{child_id_fact}'"
                ", found multiple:\n" + "\n".join(child_pred_strings)
            )
            raise ChildQueryError(msg)
        child_pred = child_preds[0]
        return self.reflect_predicate(child_pred)

    def _get_child_tuple_preds(self, parent_fact, children_id_fact):
        """Query factbase to retrieve all tuple facts identified by children_id_fact."""
        identifier = children_id_fact.id
        child_ast_pred = getattr(preds, type(children_id_fact).__name__.rstrip("1"))
        query = (
            self._reified.query(child_ast_pred)
            .where(child_ast_pred.id == identifier)
            .order_by(child_ast_pred.position)
        )
        tuples = list(query.all())
        # check that there are no tuple elements in the same position
        if len(tuples) != len(set(tup.position for tup in tuples)):
            msg = (
                f"Error finding children facts of fact '{parent_fact}':\n"
                f"Found multiple tuple elements in same position for identifier '{children_id_fact}'."
            )
            raise ChildrenQueryError(msg)
        return tuples

    def _reflect_child_preds(self, parent_fact, children_id_fact):
        """Utility function that takes a unary ast fact containing
        only an identifier pointing to a tuple of child facts, and
        returns a list of the child nodes obtained by reflecting all
        child facts.

        """
        tuples = self._get_child_tuple_preds(parent_fact, children_id_fact)
        child_nodes = []
        for tup in tuples:
            child_nodes.append(self._reflect_child_pred(tup, tup.element))
        return child_nodes

    @singledispatchmethod
    def reflect_predicate(self, pred: preds.AstPredicate):  # nocoverage
        """Convert the input AST element's reified fact representation
        back into a the corresponding member of clingo's abstract
        syntax tree, recursively reflecting all child facts.

        """
        raise NotImplementedError(
            f"Reflection not implemented for predicate of type {type(pred)}."
        )

    @reflect_predicate.register
    def _reflect_program(self, program: preds.Program) -> Sequence[AST]:
        """Reflect a (sub)program fact into sequence of AST nodes, one
        node per each statement in the (sub)program.

        """
        subprogram = []
        parameter_nodes = self._reflect_child_preds(program, program.parameters)
        subprogram.append(
            ast.Program(
                location=DUMMY_LOC, name=str(program.name), parameters=parameter_nodes
            )
        )
        statement_nodes = self._reflect_child_preds(program, program.statements)
        subprogram.extend(statement_nodes)
        return subprogram

    @reflect_predicate.register
    def _reflect_external(self, external: preds.External) -> AST:
        """Reflect an External fact into an External node."""
        symb_atom_node = self._reflect_child_pred(external, external.atom)
        body_nodes = self._reflect_child_preds(external, external.body)
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

    @reflect_predicate.register
    def _reflect_rule(self, rule: preds.Rule) -> AST:
        """Reflect a Rule fact into a Rule node."""
        head_node = self._reflect_child_pred(rule, rule.head)
        body_nodes = self._reflect_child_preds(rule, rule.body)
        return ast.Rule(location=DUMMY_LOC, head=head_node, body=body_nodes)

    @reflect_predicate.register
    def _reflect_literal(self, lit: preds.Literal) -> AST:
        """Reflect a Literal fact into a Literal node."""
        sign = preds.sign_cl2ast[preds.Sign(lit.sig)]
        atom_node = self._reflect_child_pred(lit, lit.atom)
        return ast.Literal(location=DUMMY_LOC, sign=sign, atom=atom_node)

    @reflect_predicate.register
    def _reflect_conditional_literal(self, cond_lit: preds.Conditional_Literal) -> AST:
        return ast.ConditionalLiteral(
            location=DUMMY_LOC,
            literal=self._reflect_child_pred(cond_lit, cond_lit.literal),
            condition=self._reflect_child_preds(cond_lit, cond_lit.condition),
        )

    @reflect_predicate.register
    def _reflect_symbolic_atom(self, atom: preds.Symbolic_Atom) -> AST:
        """Reflect a Symbolic_Atom fact into a SymbolicAtom node."""
        return ast.SymbolicAtom(symbol=self._reflect_child_pred(atom, atom.symbol))

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
    def _reflect_comparison(self, comparison: preds.Comparison) -> AST:
        term_node = self._reflect_child_pred(comparison, comparison.term)
        guard_tuples = self._get_child_tuple_preds(comparison, comparison.guards)
        guard_nodes = [
            ast.Guard(
                comparison=preds.comp_operator_cl2ast[g.comparison],
                term=self._reflect_child_pred(g, g.term),
            )
            for g in guard_tuples
        ]
        if len(guard_nodes) == 0:
            msg = (
                f"Error finding child facts of predicate '{comparison}'.\n"
                "Found no child guard_tuple facts with identifier matching "
                f"'{comparison.guards}', expected at least one."
            )
            raise ChildrenQueryError(msg)
        return ast.Comparison(
            term=term_node,
            guards=guard_nodes,
        )

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
        arg_nodes = self._reflect_child_preds(func, func.arguments)
        return ast.Function(
            location=DUMMY_LOC, name=str(func.name), arguments=arg_nodes, external=0
        )

    @reflect_predicate.register
    def _reflect_variable(self, var: preds.Variable) -> AST:
        """Reflect a Variable fact into a Variable node."""
        return ast.Variable(location=DUMMY_LOC, name=str(var.name))

    @reflect_predicate.register
    def _reflect_number(self, number: preds.Number) -> AST:
        """Reflect a Number fact into a Number node."""
        return ast.SymbolicTerm(
            location=DUMMY_LOC, symbol=symbol.Number(number=number.value)  # type: ignore
        )

    @reflect_predicate.register
    def _reflect_string(self, string: preds.String) -> AST:
        """Reflect a String fact into a String symbol wrapped in SymbolicTerm."""
        return ast.SymbolicTerm(
            location=DUMMY_LOC, symbol=symbol.String(string=str(string.value))
        )

    @reflect_predicate.register
    def _reflect_binary_operation(self, operation: preds.Binary_Operation) -> AST:
        """Reflect a Binary_Operation fact into a BinaryOperation node."""
        ast_operator = preds.binary_operator_cl2ast[
            preds.BinaryOperator(operation.operator)
        ]
        return ast.BinaryOperation(
            location=DUMMY_LOC,
            operator_type=ast_operator,
            left=self._reflect_child_pred(operation, operation.left),
            right=self._reflect_child_pred(operation, operation.right),
        )

    @reflect_predicate.register
    def _reflect_interval(self, interval: preds.Interval) -> AST:
        return ast.Interval(
            location=DUMMY_LOC,
            left=self._reflect_child_pred(interval, interval.left),
            right=self._reflect_child_pred(interval, interval.right),
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
        logger.info("Reflected program string:\n%s", self.program_string)

    def transform(
        self,
        meta_str: Optional[str] = None,
        meta_files: Optional[Sequence[Path]] = None,
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

        ctl = Control(["--warn=none"])
        control_add_facts(ctl, self._reified)
        ctl.add(meta_prog)
        ctl.load("./src/renopro/asp/encodings/transform.lp")
        ctl.ground()
        with ctl.solve(yield_=True) as handle:  # type: ignore
            model = next(iter(handle))
            ast_symbols = [final.arguments[0] for final in model.symbols(shown=True)]
            unifier = Unifier(preds.AstPredicates)
            with TryUnify():
                ast_facts = unifier.iter_unify(ast_symbols, raise_nomatch=True)
                self._reified = FactBase(ast_facts)


if __name__ == "__main__":  # nocoverage
    pass

#  LocalWords:  nocoverage
