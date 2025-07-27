# pylint: disable=too-many-lines
"""Module implementing reification and de-reification of non-ground programs"""
import inspect
import logging
import re
from contextlib import AbstractContextManager
from functools import singledispatchmethod
from itertools import count, groupby
from pathlib import Path
from types import TracebackType
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    cast,
    overload,
)

from clingo import Control, ast, symbol
from clingo.application import clingo_main
from clingo.script import enable_python
from clingo.solving import Model
from clingo.symbol import Function, Symbol, SymbolType
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

logger = logging.getLogger(__name__)


class ChildQueryError(Exception):
    """Exception raised when a required child fact of an AST fact
    cannot be found.

    """


class ChildrenQueryError(Exception):
    """Exception raised when the expected number child facts of an AST
    fact cannot be found.

    """


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
            pred.meta.name: {pred.meta.arity: pred} for pred in preds.Asts
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
            # solution, for figuring out which sub-field did not unify,
            # if there is one.
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


DUMMY_LOC = ast.Location(ast.Position("<string>", 1, 1), ast.Position("<string>", 1, 1))

Node = ast.AST | Symbol
NodeConstructor = Callable[..., None]
NodeAttr = Node | Sequence[Symbol] | ast.ASTSequence | str | int | ast.StrSequence

log_lvl_str2int = {"debug": 10, "info": 20, "warning": 30, "error": 40}


class ReifiedAST:
    """Class for converting between reified and non-reified
    representation of ASP programs.
    """

    _unifier_preds = preds.ASTFacts
    _unifier = Unifier(_unifier_preds)

    def __init__(self) -> None:
        self._reified = FactBase()
        self.program_stms: List[ast.AST] = []
        self._init_overrides()

    def add_reified_symbols(self, reified_symbols: Iterable[Symbol]) -> None:
        # couldn't find a way in clorm to directly add a set of facts
        # while checking unification, so we have to unify against the
        # underlying symbols
        with TryUnify():
            unified_facts = self._unifier.iter_unify(
                reified_symbols, raise_nomatch=True
            )
            self._reified.update(unified_facts)

    def add_reified_facts(self, reified_facts: Iterator[preds.ASTFact]) -> None:
        """Add iterator of reified AST facts to internal factbase."""
        # couldn't find a way in clorm to directly add a set of facts
        # while checking unification, so we have to unify against the
        # underlying symbols
        self.add_reified_symbols([fact.symbol for fact in reified_facts])

    def add_reified_string(self, reified_string: str) -> None:
        """Add string of reified facts into internal factbase."""
        with TryUnify():
            facts = parse_fact_string(
                reified_string,
                unifier=self._unifier_preds,
                raise_nomatch=True,
                raise_nonfact=True,
            )
        self._reified.update(facts)

    def add_reified_files(self, reified_files: Sequence[Path]) -> None:
        """Add files containing reified facts into internal factbase."""
        reified_files_str = [str(f) for f in reified_files]
        with TryUnify():
            facts = parse_fact_files(
                reified_files_str,
                unifier=self._unifier_preds,
                raise_nomatch=True,
                raise_nonfact=True,
            )
        self._reified.update(facts)

    def reify_string(self, prog_str: str) -> None:
        """Reify input program string, adding reified facts to the
        internal factbase."""
        self.program_stms = []
        ast.parse_string(prog_str, self.program_stms.append)
        self.reify_program(self.program_stms)

    def reify_files(self, files: Sequence[Path]) -> None:
        """Reify input program files, adding reified facts to the
        internal factbase."""
        self.program_stms = []
        for f in files:
            if not f.is_file():  # nocoverage
                raise IOError(f"File {f} does not exist.")
        files_str = [str(f) for f in files]
        ast.parse_files(files_str, self.program_ast.append)
        self.reify_program(self.program_stms)

    def reify_program(self, program: List[ast.AST]) -> None:
        """Reify input sequence of AST nodes, adding reified facts to
        the internal factbase."""
        self.program_stms = program
        if program[0].ast_type is not ast.Program:
            raise ValueError(
                "Program to be reified must start with a #program statement."
            )
        else:
            facts: List[preds.ASTFact] = []
            groups = groupby(program, lambda s: s.ast_type is ast.ASTType.Program)
            for is_prog, stms in groups:
                if is_prog:
                    prog = list(stms)[-1]
                    params = [preds.Function() for p in prog.parameters]
                else:
                    reified_stms = [self.reify_node(stm) for stm in stms]
                preds.Program(prog.name)
            self._add_ast_facts(facts)

    @property
    def program_string(self) -> str:
        """String representation of reflected AST facts."""
        return "\n".join([str(statement) for statement in self.program_stms])

    @property
    def program_ast(self) -> List[ast.AST]:
        """AST nodes attained via reflection of AST facts."""
        return self.program_stms

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

    def _add_ast_facts(self, f: preds.ASTFact | Iterable[preds.ASTFact]) -> None:
        self._reified.add(f)

    def _init_overrides(self) -> None:
        """Initialize override functions that change the default
        behavior when reifying or reflecting"""
        self._reify_overrides = {
            "node": {
                ast.ASTType.SymbolicTerm: self._override_symbolic_term,
                ast.ASTType.Id: self._override_id,
                ast.ASTType.Function: self._override_function,
            },
            "attr": {"body": lambda node: None},
            "pred_attr": {
                ast.ASTType.TheoryUnparsedTermElement: {
                    "operators": lambda ops: self._override_theory_ops(ops)
                },
                ast.ASTType.BooleanConstant: {
                    "value": lambda node: self._reify_bool(node.value)
                },
                ast.ASTType.Program: {"statements": lambda _: preds.Statements.unary()},
                ast.ASTType.Definition: {
                    "is_default": lambda node: self._reify_bool(node.is_default)
                },
                ast.ASTType.ShowSignature: {
                    "positive": lambda node: self._reify_bool(node.positive)
                },
                ast.ASTType.Defined: {
                    "positive": lambda node: self._reify_bool(node.positive)
                },
                ast.ASTType.External: {
                    "external_type": lambda node: node.external_type.symbol.name
                },
                ast.ASTType.ProjectSignature: {
                    "positive": lambda node: self._reify_bool(node.positive)
                },
                ast.ASTType.Id: {"arguments": lambda _: preds.Nil()},
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
                preds.Theory_Unparsed_Term: {
                    "elements": lambda fact: self._reflect_child(
                        fact, fact.elements, "+"
                    )
                },
                # we represent regular function and external (that is script functions)
                # with different predicates.
                preds.Function: {"external": lambda _: 0},
                preds.External_Function: {"external": lambda _: 1},
                preds.Boolean_Constant: {
                    "value": lambda fact: self._reflect_bool(fact.value)
                },
                preds.Definition: {
                    "is_default": lambda fact: self._reflect_bool(fact.is_default)
                },
                preds.Show_Signature: {
                    "positive": lambda fact: self._reflect_bool(fact.positive)
                },
                preds.Defined: {
                    "positive": lambda fact: self._reflect_bool(fact.positive)
                },
                preds.Project_Signature: {
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
    def _get_node_type(node: Node) -> ast.ASTType | SymbolType:
        """Return the type (enum member) of input ast node."""
        if isinstance(node, Symbol):
            ast_type = node.type
            return ast_type
        if isinstance(node, ast.AST):
            symbol_type = node.ast_type
            return symbol_type

    @staticmethod
    def _get_node_constructor_annotations(node: Node) -> dict[str, Any]:
        "Return the constructor annotations of input ast node."
        if isinstance(node, Symbol):
            ast_type = node.type
            ast_constructor: NodeConstructor = getattr(symbol, ast_type.name)
            return ast_constructor.__annotations__
        if isinstance(node, ast.AST):
            symbol_type = node.ast_type
            symbol_constructor: NodeConstructor = getattr(ast, symbol_type.name)
            return symbol_constructor.__annotations__
        raise TypeError(f"Node must be of type AST or Symbol, got: {type(node)}")

    def _reify_node_attr(
        self, annotation: Type[NodeAttr], attr: NodeAttr, field: BaseField
    ) -> Any:
        """Reify an AST node's attribute attr based on the type hint
        for the respective argument in the AST node's constructor.
        This default behavior is overridden in certain cases; see reify_node."""
        if annotation in [ast.AST, Symbol, Optional[ast.AST], Optional[Symbol]]:
            if annotation in [Optional[ast.AST], Optional[Symbol]] and attr is None:
                return preds.Nil()
            return self.reify_node(attr)  # type: ignore
        if annotation in [Sequence[Symbol], Sequence[ast.AST], Sequence[str]]:
            return [self.reify_node(a) for a in attr]  # type: ignore
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
        predicate: Type[preds.AstTermType],
    ) -> Dict[str, Any]:
        """Reify the attributes of an AST node.

        The attributes are reified into a kwarg dictionary, to be
        consumed by instantiating the reified predicate.

        """
        node_type = self._get_node_type(node)
        annotations = self._get_node_constructor_annotations(node)
        for key in predicate.meta.keys():
            # check if there are any overrides for the attribute and apply them
            # and continue if so
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
            # the base case
            else:
                annotation, attr = annotations.get(key), getattr(node, key)
            field = getattr(predicate, key).meta.field
            reified_attr = self._reify_node_attr(annotation, attr, field)
            kwargs_dict.update({key: reified_attr})
        return kwargs_dict

    def reify_node(self, node: Node) -> preds.AST:
        """Reify the input ast node by adding it's clorm fact
        representation to the internal fact base, and recursively
        reify child nodes.

        """
        if not (isinstance(node, ast.AST) or isinstance(node, Symbol)):
            raise TypeError("Node to be reified must be of type AST or Symbol.")
        node_type = self._get_node_type(node)
        if node_override_func := self._reify_overrides["node"].get(node_type):
            node, predicate, location = node_override_func(node)
        else:
            predicate = getattr(preds, preds.clingo2clorm_name[node_type.name])
            location = getattr(node, "location", None)
        facts: List[preds.ASTFact] = []
        kwargs_dict: dict[str, Any] = {}
        kwargs_dict = self._reify_node_attrs(node, kwargs_dict, predicate)
        reified_term = predicate(**kwargs_dict)
        if location:
            self._reify_location(reified_term, location)
        for name, arg in kwargs_dict.items():
            if isinstance(arg, preds.AST):  # type: ignore
                facts.append(preds.Child(name, reified_term, arg))
            elif isinstance(arg, list):
                for idx, child_node in enumerate(arg):
                    facts.append(preds.Child((name, idx), reified_term, child_node))
        facts.append(preds.Node(reified_term))
        self._add_ast_facts(facts)
        return reified_term

    def _reify_location(self, reified_term: preds.AST, location: ast.Location) -> None:
        begin = preds.Position(
            location.begin.filename, location.begin.line, location.begin.column
        )
        end = preds.Position(
            location.end.filename, location.end.line, location.end.column
        )
        self._add_ast_facts(preds.Location(reified_term, begin, end))

    def _override_symbolic_term(
        self, node: ast.AST
    ) -> Tuple[Node, preds.AstTermType, ast.ast.Location]:
        """Override reification of a symbolic term node.

        We don't include this node in our reified representation as it
        doesn't add much from a knowledge representation standpoint."""
        location = node.location
        node = node.symbol
        node_type = self._get_node_type(node)
        predicate = getattr(preds, node_type.name)
        return node, predicate, location

    def _override_function(self, node: ast.AST) -> Tuple[Node, Any, ast.ast.Location]:
        """Override reification of a function AST node.

        Default node reification is overridden by this method, as we
        represent external function (functions that are script calls)
        as a distinct predicate, while the clingo AST just uses a
        flag.

        """
        pred: Type[preds.Function] | Type[preds.External_Function]
        pred = preds.Function if node.external == 0 else preds.External_Function
        return node, pred, node.location

    def _override_id(self, node: ast.AST) -> Tuple[Node, Any, None]:
        """Override reification of a program parameter.

        We reify a program parameter as a constant (a function with
        empty arguments) instead of using a distinct predicate (as the
        clingo AST does), as this representation is simpler and more consistent.

        """
        return node, preds.Function, None

    def _reify_bool(self, boolean: int) -> str:
        return "true" if boolean == 1 else "false"

    def _override_theory_ops(self, ops: list[str]) -> list[preds.Theory_Operator]:
        return [preds.Theory_Operator(op) for op in ops]

    def _get_children(
        self,
        parent_fact: preds.AstTermType,
        child_id_term: Any,
        expected_children_num: Literal["1", "?", "+", "*"] = "1",
    ) -> Sequence[ast.AST]:
        identifier = child_id_term.id
        child_ast_pred = getattr(preds, type(child_id_term).__name__.rstrip("1"))
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
                f"Expected 1 child fact for identifier '{child_id_term}'"
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
                f"'{child_id_term}', found {num_child_facts}."
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
                    f"identifier '{child_id_term}'."
                )
                raise ChildrenQueryError(base_msg + msg)
            if expected_children_num == "+" and num_child_facts == 0:
                msg = (
                    f"Expected 1 or more child facts for identifier "
                    f"'{child_id_term}', found 0."
                )
                raise ChildrenQueryError(base_msg + msg)
            return child_facts
        assert_never(expected_children_num)

    @overload
    def _reflect_child(
        self,
        parent_fact: preds.AstTermType,
        child_id_fact: Any,
        expected_children_num: Literal["1"],
    ) -> ast.AST:  # nocoverage
        ...

    # for handling the default argument "1"

    @overload
    def _reflect_child(
        self, parent_fact: preds.AstTermType, child_id_fact: Any
    ) -> ast.AST:  # nocoverage
        ...

    @overload
    def _reflect_child(
        self,
        parent_fact: preds.AstTermType,
        child_id_fact: Any,
        expected_children_num: Literal["?"],
    ) -> Optional[ast.AST]:  # nocoverage
        ...

    @overload
    def _reflect_child(
        self,
        parent_fact: preds.AstTermType,
        child_id_fact: Any,
        expected_children_num: Literal["*", "+"],
    ) -> Sequence[ast.AST]:  # nocoverage
        ...

    def _reflect_child(
        self,
        parent_fact: preds.AstTermType,
        child_id_fact: Any,
        expected_children_num: Literal["1", "?", "+", "*"] = "1",
    ) -> None | ast.AST | Sequence[ast.AST]:
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
    def _node_constructor_from_pred(
        ast_pred: Type[preds.AstTermType],
    ) -> NodeConstructor:
        "Return the constructor function from clingo.ast corresponding to ast_pred."
        type_name = ast_pred.__name__
        if ast_pred is preds.External_Function:
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
    def reflect_fact(self, fact: preds.AstTermType) -> ast.AST:  # nocoverage
        """Convert the input AST element's reified fact representation
        back into a the corresponding member of clingo's abstract
        syntax tree, recursively reflecting all child facts.

        """
        predicate = type(fact)
        pred_override_func: Optional[Callable[[Any], ast.AST]] = (
            self._reflect_overrides["pred"].get(predicate)
        )
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
            elif child_type in [ast.AST, Symbol]:
                child_node = self._reflect_child(fact, field_val, "1")
                kwargs_dict.update({key: child_node})
            elif child_type in [Optional[ast.AST], Optional[Symbol]]:
                optional_child_node = self._reflect_child(fact, field_val, "?")
                kwargs_dict.update({key: optional_child_node})
            elif child_type in [Sequence[ast.AST], Sequence[Symbol], Sequence[str]]:
                child_nodes = self._reflect_child(fact, field_val, "*")
                kwargs_dict.update({key: child_nodes})
        reflected_node = node_constructor(**kwargs_dict)
        return reflected_node

    def _reflect_bool(self, boolean: Literal["true", "false"]) -> int:
        return 1 if boolean == "true" else 0

    def _reflect_program(self, program: preds.Program) -> Sequence[ast.AST]:
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
        self.program_stms = []
        # should probably define an order in which programs are queried
        for prog in self._reified.query(preds.Program).all():
            subprogram = self.reflect_fact(prog)
            self.program_stms.extend(subprogram)
        logger.debug("Reflected program string:\n%s", self.program_string)

    # def transform(
    #     input_files: List[Path] meta_files: List[Path], options: Optional[List[str]] = None
    # ) -> List[FactBase]:
    #     """Transform the reified AST using meta encoding."""
    #     options = [] if options is None else options
    #     options += [f"-m {str(meta_file.resolve())}" for meta_file in meta_files]
    #     if len(self._reified) == 0:
    #         logger.info("Reified AST to be transformed is empty.")
    #     meta_tf_app = MetaTransformerApp()
    #     args = options + ["--outf=3"]
    #     clingo_main(meta_tf_app, args)
    #     transformed_models: List[FactBase] = []
    #     unifier = Unifier(preds.AstTerms)
    #     for model in meta_tf_app.transformed_models:
    #         with TryUnify():
    #             ast_facts = unifier.iter_unify(model, raise_nomatch=True)
    #             transformed_models.append(FactBase(ast_facts))
    #     return transformed_models


if __name__ == "__main__":  # nocoverage
    pass

#  LocalWords:  nocoverage
