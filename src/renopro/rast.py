# pylint: disable=too-many-lines
"""Module implementing reification and de-reification of non-ground programs"""
import inspect
import logging
import re
from contextlib import AbstractContextManager
from functools import singledispatchmethod
from itertools import count
from pathlib import Path
from types import TracebackType
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    overload,
)

from clingo import Control, ast, symbol
from clingo.ast import (
    AST,
    ASTSequence,
    ASTType,
    Location,
    Position,
    StrSequence,
    parse_files,
    parse_string,
)
from clingo.script import enable_python
from clingo.symbol import Symbol, SymbolType
from clorm import (
    BaseField,
    FactBase,
    Unifier,
    UnifierNoMatchError,
    control_add_facts,
    parse_fact_files,
    parse_fact_string,
)
from thefuzz import process  # type: ignore

import renopro.enum_fields as enums
import renopro.predicates as preds
from renopro.enum_fields import convert_enum
from renopro.utils import assert_never
from renopro.utils.logger import get_clingo_logger_callback

logger = logging.getLogger(__name__)

enable_python()


class ChildQueryError(Exception):
    """Exception raised when a required child fact of an AST fact
    cannot be found.

    """


class ChildrenQueryError(Exception):
    """Exception raised when the expected number child facts of an AST
    fact cannot be found.

    """


class TransformationError(Exception):
    """Exception raised when a transformation meta-encoding derives an
    error or is unsatisfiable."""


class TryUnify(AbstractContextManager):  # type: ignore
    """Context manager to try some operation that requires unification
    of some set of ast facts. Enhance error message if unification fails.
    """

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is UnifierNoMatchError:
            self.handle_unify_error(cast(UnifierNoMatchError, exc_value))

    @staticmethod
    def handle_unify_error(error: UnifierNoMatchError) -> None:
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
            ) from None  # type: ignore
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
                ) from None  # type: ignore
        raise RuntimeError("Code should be unreachable")  # nocoverage


DUMMY_LOC = Location(Position("<string>", 1, 1), Position("<string>", 1, 1))

Node = AST | Symbol
NodeConstructor = Callable[..., None]
NodeAttr = Union[Node, Sequence[Symbol], ASTSequence, str, int, StrSequence]

log_lvl_str2int = {"debug": 10, "info": 20, "warning": 30, "error": 40}


def location_symb2str(location: Symbol) -> str:
    pairs = []
    for pair in zip(location.arguments[1].arguments, location.arguments[2].arguments):
        pairs.append(
            str(pair[0]) if pair[0] == pair[1] else str(pair[0]) + "-" + str(pair[1])
        )
    return ":".join(pairs)


class ReifiedAST:
    """Class for converting between reified and non-reified
    representation of ASP programs.
    """

    def __init__(self) -> None:
        self._reified = FactBase()
        self._program_ast: List[AST] = []
        self._current_statement: Tuple[Optional[preds.IdentifierPredicate], int] = (
            None,
            0,
        )
        self._tuple_pos: Iterator[int] = count()
        self._init_overrides()
        self._parent_id_term: Optional[preds.IdentifierPredicate] = None

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
        self._program_ast = []
        parse_string(prog_str, self._program_ast.append)
        self.reify_ast(self._program_ast)

    def reify_files(self, files: Sequence[Path]) -> None:
        """Reify input program files, adding reified facts to the
        internal factbase."""
        self._program_ast = []
        for f in files:
            if not f.is_file():  # nocoverage
                raise IOError(f"File {f} does not exist.")
        files_str = [str(f) for f in files]
        parse_files(files_str, self.program_ast.append)
        self.reify_ast(self._program_ast)

    def reify_ast(self, asts: List[AST]) -> None:
        """Reify input sequence of AST nodes, adding reified facts to
        the internal factbase."""
        self._program_ast = asts
        for statement in self._program_ast:
            self.reify_node(statement)

    @property
    def program_string(self) -> str:
        """String representation of reflected AST facts."""
        return "\n".join([str(statement) for statement in self._program_ast])

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

    def _init_overrides(self) -> None:
        """Initialize override functions that change the default
        behavior when reifying or reflecting"""
        self._reify_overrides = {
            "node": {
                ASTType.SymbolicTerm: self._override_symbolic_term,
                ASTType.Id: self._override_id,
                ASTType.Function: self._override_function,
            },
            "attr": {"position": lambda _: next(self._tuple_pos)},
            "pred_attr": {
                ASTType.BooleanConstant: {
                    "value": lambda node: self._reify_bool(node.value)
                },
                ASTType.Program: {"statements": lambda _: preds.Statements.unary()},
                ASTType.Definition: {
                    "is_default": lambda node: self._reify_bool(node.is_default)
                },
                ASTType.ShowSignature: {
                    "positive": lambda node: self._reify_bool(node.positive)
                },
                ASTType.Defined: {
                    "positive": lambda node: self._reify_bool(node.positive)
                },
                ASTType.External: {
                    "external_type": lambda node: node.external_type.symbol.name
                },
                ASTType.ProjectSignature: {
                    "positive": lambda node: self._reify_bool(node.positive)
                },
                ASTType.Id: {"arguments": lambda _: preds.Terms.unary()},
            },
        }
        self._reflect_overrides = {
            "pred": {
                preds.String: lambda fact: ast.SymbolicTerm(
                    DUMMY_LOC, symbol.String(fact.string)
                ),
                preds.Number: lambda fact: ast.SymbolicTerm(
                    DUMMY_LOC, symbol.Number(fact.number)
                ),
                preds.Terms: lambda fact: self._reflect_child(fact, fact.term),
                preds.TheoryTerms: lambda fact: self._reflect_child(
                    fact, fact.theory_term
                ),
                preds.TheoryOperators: lambda fact: fact.operator,
                preds.Guards: lambda fact: self._reflect_child(fact, fact.guard),
                preds.Literals: lambda fact: self._reflect_child(fact, fact.literal),
                preds.AggregateElements: lambda fact: self._reflect_child(
                    fact, fact.element
                ),
                preds.BodyLiterals: lambda fact: self._reflect_child(
                    fact, fact.body_literal
                ),
                preds.ConditionalLiterals: lambda fact: self._reflect_child(
                    fact, fact.conditional_literal
                ),
                preds.Statements: lambda fact: self._reflect_child(
                    fact, fact.statement
                ),
                preds.Constants: lambda fact: self._reflect_child(fact, fact.constant),
                preds.Program: self._reflect_program,
            },
            "attr": {"location": lambda _: DUMMY_LOC},
            "node_attr": {
                preds.Comparison: {
                    "guards": lambda fact: self._reflect_child(fact, fact.guards, "+")
                },
                preds.TheoryUnparsedTerm: {
                    "elements": lambda fact: self._reflect_child(
                        fact, fact.elements, "+"
                    )
                },
                # we represent regular function and external (that is script functions)
                # with different predicates.
                preds.Function: {"external": lambda _: 0},
                preds.ExternalFunction: {"external": lambda _: 1},
                preds.BooleanConstant: {
                    "value": lambda fact: self._reflect_bool(fact.value)
                },
                preds.Definition: {
                    "is_default": lambda fact: self._reflect_bool(fact.is_default)
                },
                preds.ShowSignature: {
                    "positive": lambda fact: self._reflect_bool(fact.positive)
                },
                preds.Defined: {
                    "positive": lambda fact: self._reflect_bool(fact.positive)
                },
                preds.ProjectSignature: {
                    "positive": lambda fact: self._reflect_bool(fact.positive)
                },
                preds.External: {
                    "external_type": lambda fact: ast.SymbolicTerm(
                        DUMMY_LOC, symbol.Function(fact.external_type, [])
                    )
                },
            },
        }

    @staticmethod
    def _get_node_type(node: Node) -> Union[ASTType, SymbolType]:
        """Return the type (enum member) of input ast node."""
        if isinstance(node, Symbol):
            ast_type = node.type
            return ast_type
        if isinstance(node, AST):
            symbol_type = node.ast_type
            return symbol_type

    @staticmethod
    def _get_node_constructor_annotations(node: Node) -> dict[str, Any]:
        "Return the constructor annotations of input ast node."
        if isinstance(node, Symbol):
            ast_type = node.type
            ast_constructor: NodeConstructor = getattr(symbol, ast_type.name)
            return ast_constructor.__annotations__
        if isinstance(node, AST):
            symbol_type = node.ast_type
            symbol_constructor: NodeConstructor = getattr(ast, symbol_type.name)
            return symbol_constructor.__annotations__
        raise TypeError(f"Node must be of type AST or Symbol, got: {type(node)}")

    def _reify_ast_sequence(
        self,
        ast_seq: Union[ASTSequence, Sequence[Symbol], Sequence[str]],
        tuple_predicate: Type[preds.AstPredicate],
    ) -> preds.IdentifierPredicate:
        "Reify an ast sequence into a list of facts of type tuple_predicate."
        tuple_id_term = tuple_predicate.unary()
        self._reified.add(preds.Child(self._parent_id_term, tuple_id_term))
        old_parent_id_term = self._parent_id_term
        self._parent_id_term = tuple_id_term
        reified_tuple_facts = []
        # Flattened tuples have arity > 3, so we reify it's attributes
        # and then construct the tuple.
        if tuple_predicate in preds.FlattenedTuples:
            tmp = self._tuple_pos
            self._tuple_pos = count()
            for node in ast_seq:
                kwargs_dict = self._reify_node_attrs(
                    node, {"id": tuple_id_term.id}, tuple_predicate
                )
                reified_tuple_fact = tuple_predicate(**kwargs_dict)
                reified_tuple_facts.append(reified_tuple_fact)
            self._tuple_pos = tmp
        # Theory operators are just a list of strings in clingo AST,
        # we need to wrap these in TheoryOperators predicate
        elif tuple_predicate is preds.TheoryOperators:
            for position, operator in enumerate(ast_seq):
                reified_tuple_fact = tuple_predicate(
                    tuple_id_term.id, position, operator
                )
                reified_tuple_facts.append(reified_tuple_fact)
        # We represent body literals explicitly in our ASP
        # representation, and categorize conditional literals separately,
        elif tuple_predicate is preds.BodyLiterals:
            for position, lit in enumerate(ast_seq):
                if lit.ast_type is ast.ASTType.ConditionalLiteral:
                    cond_lit1 = self.reify_node(lit)
                    reified_tuple_fact = preds.BodyLiterals(
                        tuple_id_term.id, position, cond_lit1
                    )
                    reified_tuple_facts.append(reified_tuple_fact)
                else:
                    body_lit1 = preds.BodyLiteral.unary()
                    reified_tuple_fact = preds.BodyLiterals(
                        tuple_id_term.id, position, body_lit1
                    )
                    self._reified.add(preds.Child(self._parent_id_term, body_lit1))
                    self._parent_id_term = body_lit1
                    clorm_sign = convert_enum(ast.Sign(lit.sign), enums.Sign)
                    body_lit = preds.BodyLiteral(
                        body_lit1.id, clorm_sign, self.reify_node(lit.atom)
                    )
                    self._reified.add(body_lit)
                    reified_tuple_facts.append(reified_tuple_fact)
                    self._parent_id_term = tuple_id_term
        # All other tuples we can reify in the usual way.
        else:
            for position, node in enumerate(ast_seq):
                child_fact_unary = self.reify_node(node)
                reified_tuple_fact = tuple_predicate(
                    tuple_id_term.id, position, child_fact_unary
                )
                reified_tuple_facts.append(reified_tuple_fact)
        self._reified.add(reified_tuple_facts)
        self._parent_id_term = old_parent_id_term
        return tuple_id_term

    def _reify_node_attr(
        self, annotation: Type[NodeAttr], attr: NodeAttr, field: BaseField
    ) -> Any:
        """Reify an AST node's attribute attr based on the type hint
        for the respective argument in the AST node's constructor.
        This default behavior is overridden in certain cases; see reify_node."""
        if annotation in [AST, Symbol, Optional[AST], Optional[Symbol]]:
            if annotation in [Optional[AST], Optional[Symbol]] and attr is None:
                return field.complex()
            return self.reify_node(attr)  # type: ignore
        if annotation in [Sequence[Symbol], Sequence[AST], Sequence[str]]:
            return self._reify_ast_sequence(
                attr, field.complex.non_unary  # type: ignore
            )
        if hasattr(field, "enum"):
            ast_enum = getattr(ast, field.enum.__name__)
            return convert_enum(ast_enum(attr), field.enum)
        if annotation in [str, int]:
            return attr
        raise RuntimeError("Code should be unreachable.")  # nocoverage

    def _reify_node_attrs(
        self,
        node: Node,
        kwargs_dict: Dict[str, Any],
        predicate: Type[preds.AstPredicate],
    ) -> Dict[str, Any]:
        """Reify the attributes of an AST node.

        The attributes are reified into a kwarg dictionary, to be
        consumed by instantiating the reified predicate.

        """
        node_type = self._get_node_type(node)
        annotations = self._get_node_constructor_annotations(node)
        for key in predicate.meta.keys():
            # the id field has no corresponding attribute in nodes to be reified
            if key == "id":
                continue
            if (
                pred_attr_overrides := self._reify_overrides["pred_attr"].get(node_type)
            ) and (attr_override_func := pred_attr_overrides.get(key)):
                kwargs_dict.update({key: attr_override_func(node)})
                continue
            if attr_override_func := self._reify_overrides["attr"].get(key):
                kwargs_dict.update({key: attr_override_func(node)})
                continue
            # sign is a reserved field name in clorm, so we used sign_ instead
            if key == "sign_":
                annotation, attr = annotations["sign"], getattr(node, "sign")
            else:
                annotation, attr = annotations.get(key), getattr(node, key)
            field = getattr(predicate, key).meta.field
            reified_attr = self._reify_node_attr(annotation, attr, field)
            kwargs_dict.update({key: reified_attr})
        return kwargs_dict

    def reify_node(self, node: Node) -> preds.IdentifierPredicate:
        """Reify the input ast node by adding it's clorm fact
        representation to the internal fact base, and recursively
        reify child nodes.

        """
        if not (isinstance(node, AST) or isinstance(node, Symbol)):
            raise TypeError("Node to be reified must be of type AST or Symbol.")
        node_type = self._get_node_type(node)
        if node_override_func := self._reify_overrides["node"].get(node_type):
            node, predicate, location = node_override_func(node)
        else:
            predicate = getattr(preds, node_type.name)
            location = getattr(node, "location", None)
        id_term = predicate.unary()
        if self._parent_id_term is not None:
            self._reified.add(preds.Child(self._parent_id_term, id_term))
        old_parent_id_term = self._parent_id_term
        self._parent_id_term = id_term
        if location:
            self._reify_location(id_term, location)
        kwargs_dict = {"id": id_term.id}
        kwargs_dict = self._reify_node_attrs(node, kwargs_dict, predicate)
        reified_fact = predicate(**kwargs_dict)
        self._reified.add(reified_fact)
        if predicate is preds.Program:
            stms_id_term = reified_fact.statements
            self._current_statement = (stms_id_term, 0)
            self._reified.add(preds.Child(id_term, stms_id_term))
            old_parent_id_term = stms_id_term
        elif predicate in preds.SubprogramStatements:
            stms_id_term, position = self._current_statement
            statement = preds.Statements(stms_id_term.id, position, id_term)
            self._reified.add(statement)
            self._current_statement = (stms_id_term, position + 1)
        self._parent_id_term = old_parent_id_term
        return id_term

    def _reify_location(
        self, id_term: preds.AstPredicate, location: ast.Location
    ) -> None:
        begin = preds.Position(
            location.begin.filename, location.begin.line, location.begin.column
        )
        end = preds.Position(
            location.end.filename, location.end.line, location.end.column
        )
        self._reified.add(preds.Location(id_term, begin, end))

    def _override_symbolic_term(
        self, node: AST
    ) -> Tuple[Node, preds.AstPredicate, ast.Location]:
        """Override reification of a symbolic term node.

        We don't include this node in our reified representation as it
        doesn't add much from a knowledge representation standpoint."""
        location = node.location
        node = node.symbol
        node_type = self._get_node_type(node)
        predicate = getattr(preds, node_type.name)
        return node, predicate, location

    def _override_function(self, node: AST) -> Tuple[Node, Any, ast.Location]:
        """Override reification of a function AST node.

        Default node reification is overridden by this method, as we
        represent external function (functions that are script calls)
        as a distinct predicate, while the clingo AST just uses a
        flag.

        """
        pred: Type[preds.Function] | Type[preds.ExternalFunction]
        pred = preds.Function if node.external == 0 else preds.ExternalFunction
        return node, pred, node.location

    def _override_id(self, node: AST) -> Tuple[Node, Any, None]:
        """Override reification of a program parameter.

        We reify a program parameter as a constant (a function with
        empty arguments) instead of using a distinct predicate (as the
        clingo AST does), as this representation is simpler and more consistent.

        """
        return node, preds.Function, None

    def _reify_bool(self, boolean: int) -> str:
        return "true" if boolean == 1 else "false"

    def _get_children(
        self,
        parent_fact: preds.AstPred,
        child_id_fact: Any,
        expected_children_num: Literal["1", "?", "+", "*"] = "1",
    ) -> Sequence[AST]:
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
                return child_facts
            msg = (
                f"Expected 1 child fact for identifier '{child_id_fact}'"
                f", found {num_child_facts}."
            )
            raise ChildQueryError(base_msg + msg)
        if expected_children_num == "?":
            child_facts = list(query.all())
            num_child_facts = len(child_facts)
            if num_child_facts in [0, 1]:
                return child_facts
            msg = (
                f"Expected 0 or 1 child fact for identifier "
                f"'{child_id_fact}', found {num_child_facts}."
            )
            raise ChildQueryError(base_msg + msg)
        #  pylint: disable=consider-using-in
        if expected_children_num == "*" or expected_children_num == "+":
            child_facts = list(query.all())
            child_facts.sort(key=lambda fact: fact.position)
            num_child_facts = len(child_facts)
            # tuple elements in the same position are not allowed.
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
            return child_facts
        assert_never(expected_children_num)

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
        expected_children_num: Literal["1", "?", "+", "*"] = "1",
    ) -> Union[None, AST, Sequence[AST]]:
        """Utility function that takes a unary ast predicate
        identifying a child predicate, queries reified factbase for
        child predicate, and returns the child node obtained by
        reflecting the child predicate.

        """
        child_facts = self._get_children(
            parent_fact, child_id_fact, expected_children_num
        )
        num_child_facts = len(child_facts)
        #  pylint: disable=consider-using-in
        if expected_children_num == "1" or expected_children_num == "?":
            if expected_children_num == "?" and num_child_facts == 0:
                return None
            return self.reflect_fact(child_facts[0])
        if expected_children_num == "+" or expected_children_num == "*":
            child_nodes = [self.reflect_fact(fact) for fact in child_facts]
            return child_nodes
        assert_never(expected_children_num)

    @staticmethod
    def _node_constructor_from_pred(ast_pred: Type[preds.AstPred]) -> NodeConstructor:
        "Return the constructor function from clingo.ast corresponding to ast_pred."
        type_name = ast_pred.__name__.rstrip("s")
        if ast_pred is preds.ExternalFunction:
            return ast.Function
        if ast_pred is preds.BodyLiteral:
            return ast.Literal
        ast_constructor = getattr(ast, type_name, None)
        if ast_constructor is not None:
            return ast_constructor
        # currently this never gets executed, as symbol predicates get overridden
        symbol_constructor = getattr(symbol, type_name, None)  # nocoverage
        if symbol_constructor is not None:  # nocoverage
            return symbol_constructor
        raise ValueError(
            f"AST Predicate '{ast_pred}' has no associated node constructor."
        )  # nocoverage

    @singledispatchmethod
    def reflect_fact(self, fact: preds.AstPred) -> AST:  # nocoverage
        """Convert the input AST element's reified fact representation
        back into a the corresponding member of clingo's abstract
        syntax tree, recursively reflecting all child facts.

        """
        predicate = type(fact)
        pred_override_func: Optional[Callable[[Any], AST]] = self._reflect_overrides[
            "pred"
        ].get(predicate)
        if pred_override_func:
            return pred_override_func(fact)
        node_constructor = self._node_constructor_from_pred(predicate)

        annotations = node_constructor.__annotations__
        kwargs_dict = {}
        for key, child_type in annotations.items():
            # sign is a reserved field name in clorm, so we had use
            # sign_ instead
            if key == "sign":
                field_val = getattr(fact, "sign_")
                field = getattr(predicate, "sign_").meta.field
            else:
                field_val = getattr(fact, key, None)
                field = (
                    getattr(predicate, key).meta.field
                    if field_val is not None
                    else None
                )
            if key == "return":
                continue
            if (
                node_attr_overrides := self._reflect_overrides["node_attr"].get(
                    predicate
                )
            ) and (attr_override_func := node_attr_overrides.get(key)):
                kwargs_dict.update({key: attr_override_func(fact)})
            elif attr_override_func := self._reflect_overrides["attr"].get(key):
                kwargs_dict.update({key: attr_override_func(fact)})
            elif clorm_enum := getattr(field, "enum", None):
                ast_enum = getattr(ast, clorm_enum.__name__)
                ast_enum_member = convert_enum(field_val, ast_enum)
                # temporary fix due to bug in clingo ast CommentType
                # https://github.com/potassco/clingo/issues/506
                if ast_enum == ast.CommentType:
                    if ast_enum_member == ast.CommentType.Block:
                        ast_enum_member = 1
                    elif ast_enum_member == ast.CommentType.Line:
                        ast_enum_member = 0
                kwargs_dict.update({key: ast_enum_member})
            elif child_type in [str, int]:
                kwargs_dict.update({key: field_val})
            elif child_type in [AST, Symbol]:
                child_node = self._reflect_child(fact, field_val, "1")
                kwargs_dict.update({key: child_node})
            elif child_type in [Optional[AST], Optional[Symbol]]:
                optional_child_node = self._reflect_child(fact, field_val, "?")
                kwargs_dict.update({key: optional_child_node})
            elif child_type in [Sequence[AST], Sequence[Symbol], Sequence[str]]:
                child_nodes = self._reflect_child(fact, field_val, "*")
                kwargs_dict.update({key: child_nodes})
        reflected_node = node_constructor(**kwargs_dict)
        return reflected_node

    def _reflect_bool(self, boolean: Literal["true", "false"]) -> int:
        return 1 if boolean == "true" else 0

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

    def reflect(self) -> None:
        """Convert stored reified ast facts into a (sequence of) AST
        node(s), and it's string representation.

        """
        # reset list of program statements before population via reflect
        self._program_ast = []
        # should probably define an order in which programs are queried
        for prog in self._reified.query(preds.Program).all():
            subprogram = self.reflect_fact(prog)
            self._program_ast.extend(subprogram)
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
        clingo_logger = get_clingo_logger_callback(logger)
        clingo_options = [] if clingo_options is None else clingo_options
        ctl = Control(clingo_options, logger=clingo_logger)
        if meta_files:
            for meta_file in meta_files:
                ctl.load(str(meta_file))
        logger.debug(
            "Reified facts before applying transformation:\n%s", self.reified_string
        )
        control_add_facts(ctl, self._reified)
        if meta_str:
            ctl.add(meta_str)
        ctl.load("./src/renopro/asp/transform.lp")
        ctl.ground()
        with ctl.solve(yield_=True) as handle:  # type: ignore
            model_iterator = iter(handle)
            try:
                model = next(model_iterator)
            except StopIteration as e:
                raise TransformationError(
                    "Transformation encoding is unsatisfiable."
                ) from e
            ast_symbols = []
            logs = {40: [], 30: [], 20: [], 10: []}
            logger.debug(
                "Stable model obtained via transformation:\n%s",
                "\n".join([str(s) for s in model.symbols(shown=True)]),
            )
            for symb in model.symbols(shown=True):
                if (
                    symb.type == SymbolType.Function
                    and symb.positive is True
                    and symb.name == "log"
                    and len(symb.arguments) > 1
                ):
                    log_lvl_symb = symb.arguments[0]
                    msg_format_str = str(symb.arguments[1])
                    log_lvl_strings = log_lvl_str2int.keys()
                    if (
                        log_lvl_symb.type != SymbolType.String
                        or log_lvl_symb.string not in log_lvl_strings
                    ):
                        raise TransformationError(
                            "First argument of log term must be one of the string symbols: '"
                            + "', '".join(log_lvl_strings)
                            + "'"
                        )
                    level = log_lvl_str2int[log_lvl_symb.string]
                    log_strings = [
                        (
                            location_symb2str(s)
                            if s.match("location", 3)
                            else str(s).strip('"')
                        )
                        for s in symb.arguments[2:]
                    ]
                    msg_str = msg_format_str.format(*log_strings)
                    logs[level].append(msg_str)
                elif symb.match("final", 1):
                    ast_symbols.append(symb.arguments[0])
            for level, msgs in logs.items():
                for msg in msgs:
                    if level == 40:
                        logger.error(
                            msg, exc_info=logger.getEffectiveLevel() == logging.DEBUG
                        )
                    else:
                        logger.log(level, msg)
            if msgs := logs[40]:
                raise TransformationError("\n".join(msgs))
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
