# pylint: disable=too-many-lines
"""Definitions of AST elements as clorm predicates."""
import inspect
import re
from itertools import count
from types import new_class
from typing import TYPE_CHECKING, Any, Protocol, Sequence, Type, Union

import clingo
from clingo import Symbol
from clorm import (
    BaseField,
    ComplexTerm,
    ConstantField,
    IntegerField,
    Predicate,
    RawField,
    StringField,
    refine_field,
)
from clorm.orm.core import _PredicateMeta

from renopro.enum_fields import (
    AggregateFunctionField,
    BinaryOperatorField,
    CommentTypeField,
    ComparisonOperatorField,
    SignField,
    TheoryAtomTypeField,
    TheoryOperatorTypeField,
    TheorySequenceTypeField,
    UnaryOperatorField,
)

id_count = count()


def combine_fields(
    fields: Sequence[Type[BaseField]], *, name: str = ""
) -> Type[BaseField]:
    """Factory function that returns a field sub-class that combines
    other fields lazily.

    Essentially the same as the combine_fields defined in the clorm
    package, but exposes a 'fields' attrible, allowing us to add
    additional fields after the initial combination of fields by
    appending to the 'fields' attribute of the combined field.

    """
    subclass_name = name if name else "AnonymousCombinedBaseField"

    # Must combine at least two fields otherwise it doesn't make sense
    for f in fields:
        if not inspect.isclass(f) or not issubclass(f, BaseField):
            raise TypeError("{f} is not BaseField or a sub-class.")

    fields = list(fields)

    def _pytocl(value: Any) -> Symbol:
        for f in fields:
            try:
                return f.pytocl(value)  # type: ignore
            except (TypeError, ValueError, AttributeError):
                pass
        raise TypeError(f"No combined pytocl() match for value {value}.")

    def _cltopy(symbol: Symbol) -> Any:
        for f in fields:
            try:
                return f.cltopy(symbol)  # type: ignore
            except (TypeError, ValueError):
                pass
        raise TypeError(
            (
                f"Object '{symbol}' ({type(symbol)}) failed to unify "
                f"with {subclass_name}."
            )
        )

    def body(ns: dict[str, Any]) -> None:
        ns.update({"fields": fields, "pytocl": _pytocl, "cltopy": _cltopy})

    return new_class(subclass_name, (BaseField,), {}, body)


# by default we use integer identifiers, but allow arbitrary symbols as well
# for flexibility when these are user generated
IntegerOrRawField = combine_fields([IntegerField, RawField], name="IntegerOrRawField")
IntegerOrRawField = IntegerOrRawField(default_factory=lambda: clingo.Number(next(id_count)))  # type: ignore

# Metaclass shenanigans to dynamically create unary versions of AST predicates,
# which are used to identify child AST facts


class _AstPredicateMeta(_PredicateMeta):
    def __new__(
        mcs,
        cls_name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> "_AstPredicateMeta":
        pattern = re.compile(r"(?<!^)(?=[A-Z])")
        underscore_lower_cls_name = pattern.sub("_", cls_name).lower()

        def id_body(ns: dict[str, Any]) -> None:
            ns.update({"id": IntegerOrRawField})

        unary = new_class(
            cls_name,
            (ComplexTerm,),
            kwds={"name": underscore_lower_cls_name},
            exec_body=id_body,
        )
        cls = super().__new__(
            mcs, cls_name, bases, namespace, name=underscore_lower_cls_name, **kwargs
        )  # type: ignore
        cls.unary = unary
        cls.unary.non_unary = cls
        return cls


class IdentifierPredicate(Predicate):
    id = IntegerOrRawField


# define callback protocol to correctly type the default argument
# behaviour of IdentifierField


class IdentifierPredicateConstructor(Protocol):
    def __call__(self, id: Any = ..., /) -> IdentifierPredicate: ...


class AstPredicate(Predicate, metaclass=_AstPredicateMeta):
    """A predicate representing an AST node."""

    if TYPE_CHECKING:
        unary: IdentifierPredicateConstructor


class String(AstPredicate):
    """Predicate representing a string term.

    id: Identifier of the string term.
    string: Value of string term, a string term itself.
    """

    id = IntegerOrRawField
    string = StringField


class Number(AstPredicate):
    """Predicate representing an integer term.

    id: Identifier of the integer term.
    number: Value of integer term, an integer term itself.
    """

    id = IntegerOrRawField
    number = IntegerField


class Variable(AstPredicate):
    """Predicate representing a variable term.

    id: Identifier of variable term.
    value: Value of variable term, a string term.
    """

    id = IntegerOrRawField
    name = StringField


TermField = combine_fields(
    [String.unary.Field, Number.unary.Field, Variable.unary.Field], name="TermField"
)


class UnaryOperation(AstPredicate):
    """Predicate representing a unary operation term.

    id: Identifier of the unary operation.
    operator: A clingo unary operator, in string form.
    argument: The term argument the unary operator is applied to.
    """

    id = IntegerOrRawField
    operator_type = UnaryOperatorField
    argument = TermField


class BinaryOperation(AstPredicate):
    """Predicate representing a binary operation term.

    id: Identifier of the binary operation.
    operator: A clingo binary operator, in string form.
    left: Predicate identifying the term that is the left operand of the operation.
    right: Predicate identifying the term that is the right operand of the operation.
    """

    id = IntegerOrRawField
    operator_type = BinaryOperatorField
    left = TermField
    right = TermField


class Interval(AstPredicate):
    """Predicate representing an interval term.

    id: Identifier of the interval.
    left: Left bound of the interval.
    right: Right bound of the interval.
    """

    id = IntegerOrRawField
    left = TermField
    right = TermField


class Terms(AstPredicate):
    """Predicate representing an element of a tuple of terms.

    id: Identifier of the tuple.
    position: Integer representing position of the element the tuple, ordered by <.
    element: Term identifying the element.
    """

    id = IntegerOrRawField
    position = IntegerOrRawField
    term = TermField


class Function(AstPredicate):
    """Predicate representing a function symbol with term arguments.
    Note that we represent function terms and constant terms as well as symbolic
    atoms and propositional constants via this predicate.

    id: Identifier of the function.
    name: Symbolic name of the function, a constant term.
    arguments: Term  identifying the function's arguments.
               If there are no elements of the term tuple with a matching
               identifier, the function has no arguments and is thus a constant.

    """

    id = IntegerOrRawField
    name = ConstantField
    arguments = Terms.unary.Field


class ExternalFunction(AstPredicate):
    """Predicate representing an external function written in a
    scripting language to be evaluated during grounding.

    id: Identifier of the function.
    name: Symbolic name of the function, a constant term.
    arguments: Term  identifying the function's arguments.
    """

    id = IntegerOrRawField
    name = ConstantField
    arguments = Terms.unary.Field


class Pool(AstPredicate):
    """Predicate representing a pool of terms.

    id: Identifier of the pool.
    arguments: Terms forming the pool.
    """

    id = IntegerOrRawField
    arguments = Terms.unary.Field


TermField.fields.extend(
    [
        UnaryOperation.unary.Field,
        BinaryOperation.unary.Field,
        Interval.unary.Field,
        Function.unary.Field,
        ExternalFunction.unary.Field,
        Pool.unary.Field,
    ]
)


TheoryTermField = combine_fields(
    [
        String.unary.Field,
        Number.unary.Field,
        Function.unary.Field,
        Variable.unary.Field,
    ],
    name="TheoryTermField",
)


class TheoryTerms(AstPredicate):
    """Predicate representing an element of a tuple of theory terms.

    id: Identifier of the tuple of theory terms.
    position: Integer representing position of the element the tuple, ordered by <.
    element: Term identifying the element.
    """

    id = IntegerOrRawField
    position = IntegerOrRawField
    theory_term = TheoryTermField


class TheorySequence(AstPredicate):
    """Predicate representing a sequence of theory terms.

    id: The identifier of the theory sequence.
    sequence_type: The type of the theory sequence.
    terms: The tuple of terms forming the sequence.
    """

    id = IntegerOrRawField
    sequence_type = TheorySequenceTypeField
    terms = TheoryTerms.unary.Field


class TheoryFunction(AstPredicate):
    """Predicate representing a theory function term.

    id: The identifier of the theory function.
    name: The name of the theory function.
    terms: The tuple of theory terms forming the arguments of
           the theory function.
    """

    id = IntegerOrRawField
    name = StringField
    arguments = TheoryTerms.unary.Field


class TheoryOperators(AstPredicate):
    """Predicate representing an element of tuple of theory operators.

    id: The identifier of the tuple of theory operators.
    position: Integer representing position of the element the tuple, ordered by <.
    operator: A theory operator, represented as a string.
    """

    id = IntegerOrRawField
    position = IntegerOrRawField
    operator = StringField


class TheoryUnparsedTermElements(AstPredicate):
    """Predicate representing an element of a tuple of unparsed theory term elements.

    id: Identifier of the tuple of elements.
    position: Integer representing position of the element
              of the theory tuple, ordered by <.
    operators: A tuple of theory operators.
    term: The theory term.
    """

    id = IntegerOrRawField
    position = IntegerOrRawField
    operators = TheoryOperators.unary.Field
    term = TheoryTermField


class TheoryUnparsedTerm(AstPredicate):
    """Predicate representing an unparsed theory term.
    An unparsed theory term consists of a tuple, each element of which
    consists of a tuple of theory operators and a theory term. This
    predicate represents an element of an unparsed theory term.

    id: The identifier of the unparsed theory term.
    elements: The tuple of aforementioned elements
              forming the unparsed theory term.
    """

    id = IntegerOrRawField
    elements = TheoryUnparsedTermElements.unary.Field


TheoryTermField.fields.extend(
    [
        TheorySequence.unary.Field,
        TheoryFunction.unary.Field,
        TheoryUnparsedTerm.unary.Field,
    ]
)


# Literals


class Guard(AstPredicate):
    """Predicate representing a guard for a comparison or aggregate atom.

    id: identifier of the guard.

    """

    id = IntegerOrRawField
    comparison = ComparisonOperatorField
    term = TermField


class Guards(AstPredicate):
    """Predicate representing a tuple of guards for a comparison or aggregate atom.

    id: Identifier of the guard.
    position: Integer representing position of the element the tuple, ordered by <.
    element: Term identifying the element.
    """

    id = IntegerOrRawField
    position = IntegerOrRawField
    guard = Guard.unary.Field


class Comparison(AstPredicate):
    """Predicate representing a comparison atom.

    id: Identifier of the comparison.
    term: Leftmost term of the comparison atom.
    guards: tuple of guard predicates
    """

    id = IntegerOrRawField
    term = TermField
    guards = Guards.unary.Field


BoolField = refine_field(ConstantField, ["true", "false"], name="BoolField")


class BooleanConstant(AstPredicate):
    """Predicate representing a boolean constant, true or false.

    id: Identifier of the boolean constant
    value: Boolean value of the constant, represented by the constant
    terms 'true' and 'false'.
    """

    id = IntegerOrRawField
    value = BoolField


# For some reason, item(fox;beans) parses differently than
# item((fox;beans)). The latter gives a symbolic atom with a function
# argument with name item and the pool as the argument. The former
# gives a symbolic atom with a pool argument, the arguments of which
# are item(fox) and item(beans). Therefore, a symbolic atom can also
# have a pool argument.


FunctionOrPoolField = combine_fields(
    [Function.unary.Field, Pool.unary.Field], name="FunctionOrPoolField"
)


class SymbolicAtom(AstPredicate):
    """Predicate representing a symbolic atom.

    id: Identifier of the atom.
    symbol: The function symbol constituting the atom.
    """

    id = IntegerOrRawField
    symbol = FunctionOrPoolField


AtomField = combine_fields(
    [SymbolicAtom.unary.Field, Comparison.unary.Field, BooleanConstant.unary.Field],
    name="AtomField",
)


class Literal(AstPredicate):
    """Predicate representing a literal.

    id: Identifier of the literal.
    sign_: Sign of the literal, in string form. Possible values are
         "pos" "not" and "not not".
    atom: The atom constituting the literal.
    """

    id = IntegerOrRawField
    sign_ = SignField
    atom = AtomField


class Literals(AstPredicate):
    """Predicate representing an element of a tuple of literals.

    id: Identifier of the tuple.
    position: Integer representing position of the element the tuple, ordered by <.
    element: Term identifying the element.
    """

    id = IntegerOrRawField
    position = IntegerOrRawField
    literal = Literal.unary.Field


class ConditionalLiteral(AstPredicate):
    """Predicate representing a conditional literal.

    id: Identifier of the conditional literal.
    literal: the literal in the head of the conditional literal
    condition: the tuple of literals forming the condition
    """

    id = IntegerOrRawField
    literal = Literal.unary.Field
    condition = Literals.unary.Field


class AggregateElements(AstPredicate):
    """Predicate representing an element of an implicit count aggregate.

    id: Identifier of the tuple.
    position: Integer representing position of the element the tuple, ordered by <.
    element: The conditional literal constituting the element of the aggregate.
    """

    id = IntegerOrRawField
    position = IntegerOrRawField
    element = ConditionalLiteral.unary.Field


class Aggregate(AstPredicate):
    """Predicate representing an simple (count) aggregate atom with
    conditional literal elements.

    id: Identifier of the count aggregate.
    left_guard: The (optional) left guard of the aggregate.
    elements: The tuple of conditional literals forming the elements of
              the aggregate.
    right_guard: The (optional) right guard of the aggregate.
    """

    id = IntegerOrRawField
    left_guard = Guard.unary.Field
    elements = AggregateElements.unary.Field
    right_guard = Guard.unary.Field


class TheoryAtomElements(AstPredicate):
    """Predicate representing an element of a tuple forming a theory atom's
    aggregate-like part.

    id: Identifier of the tuple of theory atom elements.
    position: Integer representing position of the element in the tuple,
              ordered by <.
    terms: The tuple of theory terms which form an element to be conditionally
           aggregated in the theory atom.
    condition: The tuple of literals forming a conjunction on which the
               tuple of theory terms is conditioned.
    """

    id = IntegerOrRawField
    position = IntegerOrRawField
    terms = TheoryTerms.unary.Field
    condition = Literals.unary.Field


class TheoryGuard(AstPredicate):
    """Predicate representing a theory guard.

    id: The identifier of the theory guard.
    operator_name: The name of the binary theory operator applied to
                   the aggregated elements of the theory atom and the theory term.
    term: The theory term to which the theory operator is applied.
    """

    id = IntegerOrRawField
    operator_name = StringField
    term = TheoryTermField


class TheoryAtom(AstPredicate):
    """Predicate representing a theory atom.

    id: Identifier of the theory atom.
    term: The forming the symbolic part of the theory atom.
    elements: The tuple of elements aggregated in the theory atom.
    guard: The (optional) theory guard applied the aggregated elements
           of the theory atom.
    """

    id = IntegerOrRawField
    term = Function.unary.Field
    elements = TheoryAtomElements.unary.Field
    guard = TheoryGuard.unary.Field


class BodyAggregateElements(AstPredicate):
    """Predicate representing an element of a body aggregate.

    id: Identifier of the tuple of body aggregate elements.
    position: Integer representing position of the element in the tuple, ordered by <.
    terms: The tuple of terms to which the aggregate function will be applied.
    condition: The tuple of literals forming a conjunction on which the
               tuple of terms is conditioned.
    """

    id = IntegerOrRawField
    position = IntegerOrRawField
    terms = Terms.unary.Field
    condition = Literals.unary.Field


class BodyAggregate(AstPredicate):
    """Predicate representing an aggregate atom occurring in a body.

    id: Identifier of the body aggregate.
    left_guard: The (optional) left guard of the aggregate..
    function: The aggregate function applied to the terms of the aggregate
              remaining after evaluation of the conditions.
    elements: The elements of the aggregate,
    right_guard: The (optional) right guard of the aggregate.
    """

    id = IntegerOrRawField
    left_guard = Guard.unary.Field
    function = AggregateFunctionField
    elements = BodyAggregateElements.unary.Field
    right_guard = Guard.unary.Field


BodyAtomField = combine_fields(
    AtomField.fields
    + [  # noqa: W503
        Aggregate.unary.Field,
        BodyAggregate.unary.Field,
        TheoryAtom.unary.Field,
    ],
    name="BodyAtomField",
)


class BodyLiteral(AstPredicate):
    """Predicate representing a literal occurring in the body of a
    rule.

    id: Identifier of the literal.
    sig: Sign of the literal, in string form. Possible values are
         "pos" "not" and "not not".
    atom: The atom constituting the literal.

    """

    id = IntegerOrRawField
    sign_ = SignField
    atom = BodyAtomField


BodyLiteralField = combine_fields(
    [BodyLiteral.unary.Field, ConditionalLiteral.unary.Field], name="BodyLiteralField"
)


class BodyLiterals(AstPredicate):
    """Predicate representing an element of a tuple of literals
    occurring in the body of a statement.

    id: Identifier of the tuple.
    position: Integer representing position of the element the tuple, ordered by <.
    element:  Term identifying the element.
    """

    id = IntegerOrRawField
    position = IntegerOrRawField
    body_literal = BodyLiteralField


class HeadAggregateElements(AstPredicate):
    """Predicate representing an element of a head aggregate.

    id: Identifier of the tuple of head aggregate elements.
    position: Integer representing position of the element the tuple, ordered by <.
    terms: The tuple of terms to which the aggregate function will be applied.
    condition: The conditional literal who's condition determines if
               the tuple of terms are added to the aggregate's set. When the condition
               holds, the head of the conditional literal is also derived.
    """

    id = IntegerOrRawField
    position = IntegerOrRawField
    terms = Terms.unary.Field
    condition = ConditionalLiteral.unary.Field


class HeadAggregate(AstPredicate):
    """Predicate representing an aggregate atom occuring in a head.

    id: Identifier of the head aggregate.
    left_guard: The (optional) left guard of the aggregate.
    function: The aggregate function applied to the term elements of the aggregate
              remaining after evaluation of their conditions.
    elements: The elements of the aggregate,
    right_guard: The (optional) right guard of the aggregate.
    """

    id = IntegerOrRawField
    left_guard = Guard.unary.Field
    function = AggregateFunctionField
    elements = HeadAggregateElements.unary.Field
    right_guard = Guard.unary.Field


class ConditionalLiterals(AstPredicate):
    """Predicate representing an element of a tuple of conditional literals.

    id: Identifier of the tuple of conditional literals.
    position: Integer representing position of the element the tuple, ordered by <.
    conditional_literal: Term identifying the conditional literal element.
    """

    id = IntegerOrRawField
    position = IntegerOrRawField
    conditional_literal = ConditionalLiteral.unary.Field


class Disjunction(AstPredicate):
    """Predicate representing a disjunction of (conditional) literals.

    id: Identifier of the disjunction.
    elements: The elements of the disjunction, a tuple of conditional literals.
             A literal in a disjunction is represented as a conditional literal
             with an empty condition.
    """

    id = IntegerOrRawField
    elements = ConditionalLiterals.unary.Field


HeadField = combine_fields(
    [
        Literal.unary.Field,
        Aggregate.unary.Field,
        HeadAggregate.unary.Field,
        Disjunction.unary.Field,
        TheoryAtom.unary.Field,
    ],
    name="HeadField",
)


class Rule(AstPredicate):
    """Predicate representing a rule statement.

    id: Identifier of the rule.
    head: The head of the rule.
    body: The body of the rule, a tuple of body literals.
    """

    id = IntegerOrRawField
    head = HeadField
    body = BodyLiterals.unary.Field


class Definition(AstPredicate):
    """Predicate representing a definition statement (defining a constant).

    id: Identifier of the definition.
    name: Name of the constant defined.
    value: Default value of the constant.
    is_default: true if the statement gives the default value of the
                constant, false if it's overriding it."""

    id = IntegerOrRawField
    name = ConstantField
    # the term in reality must not contain variables, pools or intervals
    # this should probably be encoded in the clorm representation.
    value = TermField
    is_default = BoolField


class ShowSignature(AstPredicate):
    """Predicate representing a show statement given by a predicate signature.

    id: Identifier of the show signature statement.
    name: Name of the predicate.
    arity: Arity of the predicate.
    positive: true if predicate is positive, false if strongly negated."""

    id = IntegerOrRawField
    name = ConstantField
    arity = IntegerField
    positive = BoolField


class Defined(AstPredicate):
    """Predicate representing a defined statement.

    id: Identifier of the defined statement.
    name: Name of the predicate to be marked as defined.
    arity: Arity of the predicate.
    positive: true if predicate is positive, false if strongly negated."""

    id = IntegerOrRawField
    name = ConstantField
    arity = IntegerField
    positive = BoolField


class ShowTerm(AstPredicate):
    """Predicate representing a show term statement.

    id: Identifier of the show term statement.
    term: The term to be shown.
    body: The body literals the term is conditioned on."""

    id = IntegerOrRawField
    term = TermField
    body = BodyLiterals.unary.Field


class Minimize(AstPredicate):
    """Predicate representing a minimize statement.

    id: Identifier of the minimize statement.
    weight: The weight associated with the tuple of terms.
    priority: The priority assigned to the weight.
    terms: The tuple of terms, which contribute their associated weight
           to the minimization. Similar to aggregates, a tuple (across all
           minimize statements) can only contribute a weight once (as in a set).
    body: The body literals on which the tuple of terms are conditioned."""

    id = IntegerOrRawField
    weight = TermField
    priority = TermField
    terms = Terms.unary.Field
    body = BodyLiterals.unary.Field


class Script(AstPredicate):
    """Predicate representing a script statement.

    id: Identifier of the script statement.
    name: Name of the embedded script.
    code: The code of the script."""

    id = IntegerOrRawField
    name = ConstantField
    code = StringField


StatementField = combine_fields(
    [
        Rule.unary.Field,
        Definition.unary.Field,
        ShowSignature.unary.Field,
        Defined.unary.Field,
        ShowTerm.unary.Field,
        Minimize.unary.Field,
        Script.unary.Field,
    ],
    name="StatementField",
)


class Statements(AstPredicate):
    """Predicate representing an element of a tuple of statements.

    id: Identifier of the tuple.
    position: Integer representing position of the element the tuple, ordered by <.
    element: Predicate identifying the element.
    """

    id = IntegerOrRawField
    position = IntegerOrRawField
    statement = StatementField


class Constants(AstPredicate):
    """Predicate representing an element of a tuple of constant terms.

    id: Identifier of the tuple.
    position: Integer representing position of the element the tuple, ordered by <.
    element: Predicate identifying the element.
    """

    id = IntegerOrRawField
    position = IntegerOrRawField
    constant = Function.unary.Field


class Program(AstPredicate):
    """Predicate representing a subprogram statement.

    id: identifier of the program statement.
    name: The name of the subprogram, a string.
    parameters: The parameters of the subprogram, a tuple of constants.
    statements: The tuple of statements comprising the subprogram.
    """

    id = IntegerOrRawField
    name = StringField
    parameters = Constants.unary.Field
    statements = Statements.unary.Field


# note that clingo's parser actually allows arbitrary constant as the external_type
# argument of External, but any other value than true or false results in the external
# statement having no effect


class External(AstPredicate):
    """Predicate representing an external statement.

    id: Identifier of the external statement.
    atom: The external atom.
    body: The tuple of body literals the external statement is conditioned on.
    external_type: The default value of the external statement.
                   May be the constant 'true' or 'false'.
    """

    id = IntegerOrRawField
    atom = SymbolicAtom.unary.Field
    body = BodyLiterals.unary.Field
    external_type = BoolField


class Edge(AstPredicate):
    """Predicate representing an edge statement. Answer sets where the
    directed edges obtained from edge statements form a cyclic graph
    are discarded.

    id: Identifier of the edge statement.
    u: The term forming first element of the directed edge.
    v: The term forming the second element of the directed edge.
    body: The body on which the directed edge is conditioned."""

    id = IntegerOrRawField
    node_u = TermField
    node_v = TermField
    body = BodyLiterals.unary.Field


HeuristicModifierField = refine_field(
    ConstantField,
    ["sign", "level", "true", "false", "init", "factor"],
    name="HeuristicModifierField",
)


class Heuristic(AstPredicate):
    """Predicate representing a heuristic statement.

    id: Identifier of the heuristic statement.
    atom: The symbolic atom to which the heuristic should be applied when
          making a decision on it's truth value.
    body: The body atoms on which the application of the heuristic
          is conditioned.
    bias: The bias we associate with the atom.
    priority: The priority of the heuristic (higher overrides lower).
    modifier: The heuristic modifier to be applied."""

    id = IntegerOrRawField
    atom = SymbolicAtom.unary.Field
    body = BodyLiterals.unary.Field
    bias = TermField
    priority = TermField
    modifier = TermField


class ProjectAtom(AstPredicate):
    """Predicate representing a project atom statement.

    id: Identifier of the project atom statement.
    atom: The atom to be projected onto.
    body: The body literals on which the projection is conditioned."""

    id = IntegerOrRawField
    atom = SymbolicAtom.unary.Field
    body = BodyLiterals.unary.Field


class ProjectSignature(AstPredicate):
    """Predicate representing a project signature statement.

    id: Identifier of the project signature statement.
    name: Name of the predicate to be projected.
    arity: Arity of the predicate to be projected.
    positive: True if predicate is positive, false if strongly negated."""

    id = IntegerOrRawField
    name = ConstantField
    arity = IntegerField
    positive = BoolField


class TheoryOperatorDefinitions(AstPredicate):
    """Predicate representing an element of a tuple of theory
    operator definitions.

    id: Identifier of the tuple of theory operator definitions.
    position: Integer representing position of the element the tuple, ordered by <.
    name: The operator to be defined, in string form.
    priority: The precedence of the operator, determining implicit parentheses.
    operator type: The type of the operator."""

    id = IntegerOrRawField
    position = IntegerOrRawField
    name = StringField
    priority = IntegerField
    operator_type = TheoryOperatorTypeField


class TheoryTermDefinitions(AstPredicate):
    """Predicate representing an element of a tuple of theory term definitions.

    id: Identifier of the tuple of theory term definitions.
    position: Integer representing position of the element the tuple, ordered by <.
    name: Name of the theory term to be defined.
    operators: The theory operators defined over the theory term."""

    id = IntegerOrRawField
    position = IntegerOrRawField
    name = ConstantField
    operators = TheoryOperatorDefinitions.unary.Field


class TheoryGuardDefinition(AstPredicate):
    """Predicate representing a theory guard definition.

    id: Identifier of the theory guard definition.
    operators: The possible operators usable to form the guard.
    term: The theory term usable to form the guard."""

    id = IntegerOrRawField
    operators = TheoryOperators.unary.Field
    term = ConstantField


class TheoryAtomDefinitions(AstPredicate):
    """Predicate representing an element of a tuple of theory atom definitions.

    id: Identifier of the tuple of theory atom definitions.
    position: Integer representing position of the element the tuple, ordered by <.
    atom_type: The type of the theory atom, determining the possible places it
               may occur in a logic program.
    name: The name of the theory atom to be defined
    arity: The arity of the theory atom to be defined
    term: The theory term type that may appear as an element of the theory atom.
    guard: The definition of the theory guard which may serve as the theory
           atom's guard.
    """

    id = IntegerOrRawField
    position = IntegerOrRawField
    atom_type = TheoryAtomTypeField
    name = ConstantField
    arity = IntegerField
    term = ConstantField
    guard = TheoryGuardDefinition.unary.Field


class TheoryDefinition(AstPredicate):
    """Predicate representing a theory definition statement.

    id: The identifier of the theory statement.
    name: Name of the theory to be defined.
    terms: A tuple of theory term definitions.
    atoms: A tuple of theory atom definitions."""

    id = IntegerOrRawField
    name = ConstantField
    terms = TheoryTermDefinitions.unary.Field
    atoms = TheoryAtomDefinitions.unary.Field


class Comment(AstPredicate):
    """Predicate representing a comment statement.

    id: The identifier of the comment statement.
    value: The string value the comment comprises of.
    comment_type: The type of the comment, "block" or "line"."""

    id = IntegerOrRawField
    value = StringField
    comment_type = CommentTypeField


StatementField.fields.extend(
    [
        External.unary.Field,
        Edge.unary.Field,
        Heuristic.unary.Field,
        ProjectAtom.unary.Field,
        ProjectSignature.unary.Field,
        TheoryDefinition.unary.Field,
        Comment.unary.Field,
    ]
)


AstPreds = (
    String,
    Number,
    Variable,
    UnaryOperation,
    BinaryOperation,
    Interval,
    Terms,
    Function,
    ExternalFunction,
    Pool,
    TheoryTerms,
    TheorySequence,
    TheoryFunction,
    TheoryOperators,
    TheoryUnparsedTermElements,
    TheoryUnparsedTerm,
    Guard,
    Guards,
    Comparison,
    BooleanConstant,
    SymbolicAtom,
    Literal,
    Literals,
    ConditionalLiteral,
    AggregateElements,
    Aggregate,
    TheoryAtomElements,
    TheoryGuard,
    TheoryAtom,
    BodyAggregateElements,
    BodyAggregate,
    BodyLiteral,
    BodyLiterals,
    HeadAggregateElements,
    HeadAggregate,
    ConditionalLiterals,
    Disjunction,
    Rule,
    Definition,
    ShowSignature,
    Defined,
    ShowTerm,
    Minimize,
    Script,
    Statements,
    Constants,
    Program,
    External,
    Edge,
    Heuristic,
    ProjectAtom,
    ProjectSignature,
    TheoryOperatorDefinitions,
    TheoryTermDefinitions,
    TheoryGuardDefinition,
    TheoryAtomDefinitions,
    TheoryDefinition,
    Comment,
)


AstIdentifierTermField = combine_fields(
    [pred.unary.Field for pred in AstPreds], name="AstIdentifierTermField"
)


class Position(ComplexTerm):
    """Complex field representing a position in a text file."""

    filename = StringField
    line = IntegerField
    column = IntegerField


class Location(Predicate):
    """Predicate linking an AST identifier to the range in a text
    file from where it was reified."""

    id_term = AstIdentifierTermField
    begin = Position.Field
    end = Position.Field


class Child(Predicate):
    """Predicate linking a parent AST predicate's identifier to it's
    child predicate's identifier."""

    parent = AstIdentifierTermField
    child = AstIdentifierTermField


AstPreds = AstPreds + (Location, Child)

AstPred = Union[
    String,
    Number,
    Variable,
    UnaryOperation,
    BinaryOperation,
    Interval,
    Terms,
    Function,
    ExternalFunction,
    Pool,
    TheoryTerms,
    TheorySequence,
    TheoryFunction,
    TheoryOperators,
    TheoryUnparsedTermElements,
    TheoryUnparsedTerm,
    Guard,
    Guards,
    Comparison,
    BooleanConstant,
    SymbolicAtom,
    Literal,
    Literals,
    ConditionalLiteral,
    AggregateElements,
    Aggregate,
    TheoryAtomElements,
    TheoryGuard,
    TheoryAtom,
    BodyAggregateElements,
    BodyAggregate,
    BodyLiteral,
    BodyLiterals,
    HeadAggregateElements,
    HeadAggregate,
    ConditionalLiterals,
    Disjunction,
    Rule,
    Definition,
    ShowSignature,
    Defined,
    ShowTerm,
    Minimize,
    Script,
    Statements,
    Constants,
    Program,
    External,
    Edge,
    Heuristic,
    ProjectAtom,
    ProjectSignature,
    TheoryOperatorDefinitions,
    TheoryTermDefinitions,
    TheoryGuardDefinition,
    TheoryAtomDefinitions,
    Comment,
    TheoryDefinition,
    Location,
    Child,
]


SubprogramStatements = (
    Rule,
    Definition,
    ShowSignature,
    Defined,
    ShowTerm,
    Minimize,
    Script,
    External,
    Edge,
    Heuristic,
    ProjectAtom,
    ProjectSignature,
    TheoryDefinition,
    Comment,
)

FlattenedTuples = (
    TheoryUnparsedTermElements,
    TheoryAtomElements,
    BodyAggregateElements,
    HeadAggregateElements,
    TheoryOperatorDefinitions,
    TheoryTermDefinitions,
    TheoryAtomDefinitions,
)

pred_names = tuple(predicate.meta.name for predicate in AstPreds)
composed_pred_names = tuple(predicate.meta.name + "_" for predicate in AstPreds)

name2arity2pred = {pred.meta.name: {pred.meta.arity: pred} for pred in AstPreds}


class Final(Predicate):
    """Wrapper predicate to distinguish output AST facts of a transformation."""

    ast = combine_fields([pred.Field for pred in AstPreds])
