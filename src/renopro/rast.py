# pylint: disable=too-many-lines
"""Module implementing reification and de-reification of non-ground programs"""
import inspect
import logging
import re
from contextlib import AbstractContextManager
from functools import singledispatchmethod
from itertools import count
from pathlib import Path
from typing import (
    Callable,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    overload,
)

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
    FactBase,
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

NodeConstructor = Callable[..., Union[AST, Symbol]]


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


class ReifiedAST:
    """Class for converting between reified and non-reified
    representation of ASP programs.
    """

    def __init__(self):
        self._reified = FactBase()
        self._program_ast = []
        self._current_statement = (None, None)
        self._program_string = ""
        self._tuple_pos = None
        self._init_overrides()

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

    def reify_ast(self, asts: Sequence[AST]) -> None:
        """Reify input sequence of AST nodes, adding reified facts to
        the internal factbase."""
        for node in asts:
            self.reify_node(node)

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

    def _init_overrides(self):
        """Initialize override functions that change the default
        behavior when reifying or reflecting"""
        self._reify_overrides = {
            "node": {
                # we don't include this node in our reified representation
                # as it doesn't add much from a knowledge representation standpoint.
                ASTType.SymbolicTerm: lambda node: self.reify_node(node.symbol),
                ASTType.Id: self._reify_id,
                ASTType.Function: self._reify_function,
            },
            "attr": {
                # body literals need special treatment, as we distinguish
                # between them and regular literals while the clingo AST
                # does not.
                "body": lambda node: self._reify_body_literals(node.body),
                "position": lambda none: next(self._tuple_pos),
            },
            "node_attr": {
                ASTType.BooleanConstant: {
                    "value": lambda node: self._reify_bool(node.value)
                },
                ASTType.Program: {"statements": lambda none: preds.Statements.unary()},
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
                ASTType.TheoryUnparsedTermElement: {
                    "operators": lambda node: self._reify_theory_operators(
                        node.operators
                    )
                },
                ASTType.TheoryGuardDefinition: {
                    "operators": lambda node: self._reify_theory_operators(
                        node.operators
                    )
                },
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
            "attr": {"location": lambda none: DUMMY_LOC},
            "pred_attr": {
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
                preds.Function: {"external": lambda none: 0},
                preds.ExternalFunction: {"external": lambda none: 1},
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

    def _reify_ast_sequence(
        self, ast_seq: ASTSequence, tuple_predicate: Type[preds.AstPredicate]
    ):
        "Reify an ast sequence into a list of facts of type tuple_predicate."
        tuple_unary_fact = tuple_predicate.unary()
        reified_facts = []
        tmp = self._tuple_pos
        self._tuple_pos = count()
        if tuple_predicate in preds.FlattenedTuples:
            for node in ast_seq:
                self.reify_node(
                    node, predicate=tuple_predicate, id_term=tuple_unary_fact
                )
        else:
            for position, node in enumerate(ast_seq):
                child_fact_unary = self.reify_node(node)
                tuple_fact = tuple_predicate(
                    tuple_unary_fact.id, position, child_fact_unary
                )
                reified_facts.append(tuple_fact)
                self._reified.add(reified_facts)
        self._tuple_pos = tmp
        return tuple_unary_fact

    @staticmethod
    def _get_type_constructor_from_node(
        node: Union[AST, Symbol]
    ) -> Tuple[Union[ASTType, SymbolType], NodeConstructor]:
        "Return the type and constructor of input ast node."
        if isinstance(node, Symbol):
            ast_type = node.type
            ast_constructor: NodeConstructor = getattr(symbol, ast_type.name)
            return ast_type, ast_constructor
        if isinstance(node, AST):
            symbol_type = node.ast_type
            symbol_constructor: NodeConstructor = getattr(ast, symbol_type.name)
            return symbol_type, symbol_constructor
        raise TypeError(f"Node must be of type AST or Symbol, got: {type(node)}")

    def reify_node(self, node: Union[AST, Symbol], predicate=None, id_term=None):
        """Reify the input ast node by adding it's clorm fact
        representation to the internal fact base, and recursively
        reify child nodes.

        """
        node_type, node_constructor = self._get_type_constructor_from_node(node)

        if node_override_func := self._reify_overrides["node"].get(node_type):
            return node_override_func(node)

        annotations = node_constructor.__annotations__
        predicate = getattr(preds, node_type.name) if predicate is None else predicate
        id_term = predicate.unary() if id_term is None else id_term
        kwargs_dict = {"id": id_term.id}

        for key in predicate.meta.keys():
            # the id field has no corresponding attribute in nodes to be reified
            if key == "id":
                continue
            if (
                node_attr_overrides := self._reify_overrides["node_attr"].get(node_type)
            ) and (attr_override_func := node_attr_overrides.get(key)):
                kwargs_dict.update({key: attr_override_func(node)})
                continue
            if attr_override_func := self._reify_overrides["attr"].get(key):
                kwargs_dict.update({key: attr_override_func(node)})
                continue
            # sign is a reserved field name in clorm, so we had use
            # sign_ instead
            if key == "sign_":
                annotation = annotations["sign"]
                attr = getattr(node, "sign")
            else:
                annotation = annotations.get(key)
                attr = getattr(node, key)
            field = getattr(predicate, key).meta.field
            if annotation in [AST, Symbol, Optional[AST], Optional[Symbol]]:
                if annotation in [Optional[AST], Optional[Symbol]] and attr is None:
                    kwargs_dict.update({key: field.complex()})
                    continue
                child_fact_unary = self.reify_node(attr)
                kwargs_dict.update({key: child_fact_unary})
            elif annotation in [Sequence[Symbol], Sequence[AST]]:
                child_tuple_fact_unary = self._reify_ast_sequence(
                    attr, field.complex.non_unary
                )
                kwargs_dict.update({key: child_tuple_fact_unary})
            elif hasattr(field, "enum"):
                ast_enum = getattr(ast, field.enum.__name__)
                clorm_enum_member = preds.convert_enum(ast_enum(attr), field.enum)
                kwargs_dict.update({key: clorm_enum_member})
            elif annotation in [str, int]:
                kwargs_dict.update({key: attr})
            else:  # nocoverage
                raise RuntimeError("Code should be unreachable.")
        reified_fact = predicate(**kwargs_dict)
        self._reified.add(reified_fact)
        if predicate is preds.Program:
            self._current_statement = (reified_fact.statements.id, 0)
        if predicate in preds.SubprogramStatements:
            id_, position = self._current_statement
            statement_element = preds.Statements(
                id=id_, position=position, statement=id_term
            )
            self._reified.add(statement_element)
            self._current_statement = (id_, position + 1)
        return id_term

    def _reify_theory_operators(self, operators: Sequence[str]):
        operators1 = preds.TheoryOperators.unary()
        reified_operators = [
            preds.TheoryOperators(id=operators1.id, position=p, operator=op)
            for p, op in enumerate(operators)
        ]
        self._reified.add(reified_operators)
        return operators1

    def _reify_body_literals(self, nodes: Sequence[ast.AST]):
        body_lits1 = preds.BodyLiterals.unary()
        reified_body_lits = []
        for pos, lit in enumerate(nodes, start=0):
            if lit.ast_type is ast.ASTType.ConditionalLiteral:
                cond_lit1 = self.reify_node(lit)
                reified_body_lits.append(
                    preds.BodyLiterals(
                        id=body_lits1.id, position=pos, body_literal=cond_lit1
                    )
                )
            else:
                body_lit1 = preds.BodyLiteral.unary()
                reified_body_lits.append(
                    preds.BodyLiterals(
                        id=body_lits1.id, position=pos, body_literal=body_lit1
                    )
                )
                clorm_sign = preds.convert_enum(ast.Sign(lit.sign), preds.Sign)
                body_lit = preds.BodyLiteral(
                    id=body_lit1.id, sign_=clorm_sign, atom=self.reify_node(lit.atom)
                )
                self._reified.add(body_lit)
        self._reified.add(reified_body_lits)
        return body_lits1

    def _reify_function(self, node):
        if node.external == 0:
            pred = preds.Function
            func1 = pred.unary()
        else:
            pred = preds.ExternalFunction
            func1 = pred.unary()
        self._reified.add(
            pred(
                func1.id,
                node.name,
                arguments=self._reify_ast_sequence(node.arguments, preds.Terms),
            )
        )
        return func1

    def _reify_id(self, node):
        const1 = preds.Function.unary()
        self._reified.add(
            preds.Function(id=const1.id, name=node.name, arguments=preds.Terms.unary())
        )
        return const1

    def _reify_bool(self, boolean: int):
        return "true" if boolean == 1 else "false"

    def _get_children(
        self,
        parent_fact: preds.AstPred,
        child_id_fact,
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
    def reflect_fact(self, fact: preds.AstPred):  # nocoverage
        """Convert the input AST element's reified fact representation
        back into a the corresponding member of clingo's abstract
        syntax tree, recursively reflecting all child facts.

        """
        predicate = type(fact)
        if pred_override_func := self._reflect_overrides["pred"].get(predicate):
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
                pred_attr_overrides := self._reflect_overrides["pred_attr"].get(
                    predicate
                )
            ) and (attr_override_func := pred_attr_overrides.get(key)):
                kwargs_dict.update({key: attr_override_func(fact)})
            elif attr_override_func := self._reflect_overrides["attr"].get(key):
                kwargs_dict.update({key: attr_override_func(fact)})
            elif clorm_enum := getattr(field, "enum", None):
                ast_enum = getattr(ast, clorm_enum.__name__)
                ast_enum_member = preds.convert_enum(field_val, ast_enum)
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

    def _reflect_bool(self, boolean: Literal["true", "false"]):
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

    def reflect(self):
        """Convert stored reified ast facts into a (sequence of) AST
        node(s), and it's string representation.

        """
        # reset list of program statements before population via reflect
        self._program_ast = []
        # should probably define an order in which programs are queried
        for prog in self._reified.query(preds.Program).all():
            subprogram = self.reflect_fact(prog)
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
