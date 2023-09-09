# pylint: disable=too-many-lines
"""Definitions of AST elements as clorm predicates."""
import enum
import inspect
from itertools import count
from types import new_class
from typing import List, Sequence, Type, TypeVar, Union

from clorm import (
    BaseField,
    ComplexTerm,
    ConstantField,
    IntegerField,
    Predicate,
    RawField,
    StringField,
    combine_fields,
    define_enum_field,
    refine_field,
)
from clorm.orm.core import _PredicateMeta

id_count = count()

# by default we use integer identifiers, but allow arbitrary symbols as well
# for flexibility when these are user generated
Identifier_Field = combine_fields([IntegerField, RawField], name="IdentifierField")


class LazilyCombinedField(BaseField):  # nocoverage
    # pylint: disable=no-self-argument
    "A field defined by lazily combining multiple existing field definitions."
    fields: List[BaseField] = []

    # cannot use a staticmethod decorator as clorm raises error if cltopy/pytocl
    # is not callable, and staticmethods are not callable for python < 3.10
    def cltopy(v):
        pass

    def pytocl(v):
        pass


def combine_fields_lazily(
    fields: Sequence[BaseField], *, name: str = ""
) -> Type[LazilyCombinedField]:
    """Factory function that returns a field sub-class that combines
    other fields lazily.

    Essentially the same as the combine_fields defined in the clorm
    package, but allows us to add additional fields after the initial
    combination of fields by appending to the fields attribute of the
    combined field.

    """
    subclass_name = name if name else "AnonymousCombinedBaseField"

    # Must combine at least two fields otherwise it doesn't make sense
    for f in fields:
        if not inspect.isclass(f) or not issubclass(f, BaseField):
            raise TypeError("{f} is not BaseField or a sub-class.")

    fields = list(fields)

    def _pytocl(value):
        for f in fields:
            try:
                return f.pytocl(value)
            except (TypeError, ValueError, AttributeError):
                pass
        raise TypeError(f"No combined pytocl() match for value {value}.")

    def _cltopy(symbol):
        for f in fields:
            try:
                return f.cltopy(symbol)
            except (TypeError, ValueError):
                pass
        raise TypeError(
            (
                f"Object '{symbol}' ({type(symbol)}) failed to unify "
                f"with {subclass_name}."
            )
        )

    return type(
        subclass_name,
        (BaseField,),
        {"fields": fields, "pytocl": _pytocl, "cltopy": _cltopy},
    )


# Enum field definitions


class UnaryOperator(str, enum.Enum):
    "String enum of clingo's unary operators."
    Absolute = "||"  # For taking the absolute value.
    Minus = "-"  # For unary minus and classical negation.
    Negation = "~"  # For bitwise negation


UnaryOperatorField = define_enum_field(
    parent_field=StringField, enum_class=UnaryOperator, name="UnaryOperatorField"
)


class BinaryOperator(str, enum.Enum):
    "String enum of clingo's binary operators."
    And = "&"  # bitwise and
    Division = "/"  # arithmetic division
    Minus = "-"  # arithmetic subtraction
    Modulo = "%"  # arithmetic modulo
    Multiplication = "*"  # arithmetic multiplication
    Or = "?"  # bitwise or
    Plus = "+"  # arithmetic addition
    Power = "**"  # arithmetic exponentiation
    XOr = "^"  # bitwise exclusive or


BinaryOperatorField = define_enum_field(
    parent_field=StringField, enum_class=BinaryOperator, name="BinaryOperatorField"
)


class ComparisonOperator(str, enum.Enum):
    """
    String enumeration of clingo's comparison operators.
    """

    Equal = "="
    GreaterEqual = ">="
    GreaterThan = ">"
    LessEqual = "<="
    LessThan = "<"
    NotEqual = "!="


ComparisonOperatorField = define_enum_field(
    parent_field=StringField,
    enum_class=ComparisonOperator,
    name="ComparisonOperatorField",
)


class TheorySequenceType(str, enum.Enum):
    """String enum of theory sequence types."""

    List = "[]"
    """
    For sequences enclosed in brackets.
    """
    Set = "{}"
    """
    For sequences enclosed in braces.
    """
    Tuple = "()"
    """
    For sequences enclosed in parenthesis.
    """


TheorySequenceTypeField = define_enum_field(
    parent_field=StringField,
    enum_class=TheorySequenceType,
    name="TheorySequenceTypeField",
)


class Sign(str, enum.Enum):
    """String enum of possible sign of a literal."""

    DoubleNegation = "not not"
    """
    For double negated literals (with prefix `not not`)
    """
    Negation = "not"
    """
    For negative literals (with prefix `not`).
    """
    NoSign = "pos"
    """
    For positive literals.
    """


SignField = define_enum_field(
    parent_field=StringField, enum_class=Sign, name="SignField"
)


class AggregateFunction(str, enum.Enum):
    "String enum of clingo's aggregate functions."
    Count = "#count"
    Max = "#max"
    Min = "#min"
    Sum = "#sum"
    SumPlus = "#sum+"


AggregateFunctionField = define_enum_field(
    parent_field=StringField,
    enum_class=AggregateFunction,
    name="AggregateFunctionField",
)


class TheoryOperatorType(str, enum.Enum):
    "String enum of clingo's theory definition types"
    BinaryLeft = "binary, left"
    BinaryRight = "binary, right"
    Unary = "unary"


TheoryOperatorTypeField = define_enum_field(
    parent_field=StringField,
    enum_class=TheoryOperatorType,
    name="TheoryOperatorTypeField",
)


class TheoryAtomType(str, enum.Enum):
    "String enum of clingo's theory atom types."
    Any = "any"
    Body = "body"
    Directive = "directive"
    Head = "head"


TheoryAtomTypeField = define_enum_field(
    parent_field=StringField, enum_class=TheoryAtomType, name="TheoryAtomTypeField"
)


A = TypeVar("A", bound=enum.Enum)
B = TypeVar("B", bound=enum.Enum)


def convert_enum(enum_member: A, other_enum: Type[B]) -> B:
    """Given an enum_member, convert it to the other_enum member of
    the same name.
    """
    # enum_type = type(enum_member)
    # cast to enum - needed as enum members stored in a clingo AST object
    # gets cast to it's raw value for some reason
    # enum_member = enum_type(enum_member)
    try:
        return other_enum[enum_member.name]
    except KeyError as exc:  # nocoverage
        msg = (
            f"Enum {other_enum} has no corresponding member "
            f"with name {enum_member.name}"
        )
        raise ValueError(msg) from exc


# Metaclass shenanigans to dynamically create unary versions of AST predicates,
# which are used to identify child AST facts


# pylint: disable=too-few-public-methods
class id_terms:
    """A namespace to store dynamically generated identifier predicate
    definitions.

    For each defined AstPredicate P, a predicate definition is
    generated in this namespace with the same name, and arity of 1,
    the single argument of which is an identifier I. This unary
    predicate is used as a term in further predicate definitions to
    identify a child fact (representing a child node) of type P with
    identifier I.

    """


class _AstPredicateMeta(_PredicateMeta):
    def __new__(mcs, cls_name, bases, namespace, **kwargs):
        if cls_name != "AstP":

            def id_body(ns):
                ns.update({"id": Identifier_Field(default=lambda: next(id_count))})

            id_term = new_class(cls_name, (ComplexTerm,), kwds=None, exec_body=id_body)
            setattr(id_terms, cls_name, id_term)
        return super().__new__(mcs, cls_name, bases, namespace, **kwargs)


class AstPredicate(Predicate, metaclass=_AstPredicateMeta):
    """A predicate representing an AST node."""


# Terms


class String(AstPredicate):
    """Predicate representing a string term.

    id: Identifier of the string term.
    value: Value of string term, a string term itself.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    value = StringField


class Number(AstPredicate):
    """Predicate representing an integer term.

    id: Identifier of the integer term.
    value: Value of integer term, an integer term itself.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    value = IntegerField


class Variable(AstPredicate):
    """Predicate representing a variable term.

    id: Identifier of variable term.
    value: Value of variable term, a string term.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    name = StringField


TermField = combine_fields_lazily(
    [id_terms.String.Field, id_terms.Number.Field, id_terms.Variable.Field],
    name="TermField",
)


class Unary_Operation(AstPredicate):
    """Predicate representing a unary operation term.

    id: Identifier of the unary operation.
    operator: A clingo unary operator, in string form.
    argument: The term argument the unary operator is applied to.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    operator = UnaryOperatorField
    argument = TermField


class Binary_Operation(AstPredicate):
    """Predicate representing a binary operation term.

    id: Identifier of the binary operation.
    operator: A clingo binary operator, in string form.
    left: Predicate identifying the term that is the left operand of the operation.
    right: Predicate identifying the term that is the right operand of the operation.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    operator = BinaryOperatorField
    left = TermField
    right = TermField


class Interval(AstPredicate):
    """Predicate representing an interval term.

    id: Identifier of the interval.
    left: Left bound of the interval.
    right: Right bound of the interval.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    left = TermField
    right = TermField


class Terms(AstPredicate):
    """Predicate representing an element of a tuple of terms.

    id: Identifier of the tuple.
    position: Integer representing position of the element the tuple, ordered by <.
    element: Term identifying the element.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    position = IntegerField
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

    id = Identifier_Field(default=lambda: next(id_count))
    name = ConstantField
    arguments = id_terms.Terms.Field


class Pool(AstPredicate):
    """Predicate representing a pool of terms.

    id: Identifier of the pool.
    arguments: Terms forming the pool.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    arguments = id_terms.Terms.Field


TermField.fields.extend(
    [
        id_terms.Unary_Operation.Field,
        id_terms.Binary_Operation.Field,
        id_terms.Interval.Field,
        id_terms.Function.Field,
        id_terms.Pool.Field,
    ]
)


TheoryTermField = combine_fields_lazily(
    [
        id_terms.String.Field,
        id_terms.Number.Field,
        id_terms.Function.Field,
        id_terms.Variable.Field,
    ],
    name="TheoryTermField",
)


class Theory_Terms(AstPredicate):
    """Predicate representing an element of a tuple of theory terms.

    id: Identifier of the tuple of theory terms.
    position: Integer representing position of the element the tuple, ordered by <.
    element: Term identifying the element.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    position = IntegerField
    theory_term = TheoryTermField


class Theory_Sequence(AstPredicate):
    """Predicate representing a sequence of theory terms.

    id: The identifier of the theory sequence.
    sequence_type: The type of the theory sequence.
    terms: The tuple of terms forming the sequence.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    sequence_type = TheorySequenceTypeField
    terms = id_terms.Theory_Terms.Field


class Theory_Function(AstPredicate):
    """Predicate representing a theory function term.

    id: The identifier of the theory function.
    name: The name of the theory function.
    terms: The tuple of theory terms forming the arguments of
           the theory function.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    name = ConstantField
    arguments = id_terms.Theory_Terms.Field


class Theory_Operators(AstPredicate):
    """Predicate representing an element of tuple of theory operators.

    id: The identifier of the tuple of theory operators.
    position: Integer representing position of the element the tuple, ordered by <.
    operator: A theory operator, represented as a string.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    position = IntegerField
    operator = StringField


class Theory_Unparsed_Term_Elements(AstPredicate):
    """Predicate representing an element of an unparsed theory term.

    id: Identifier of the tuple of elements.
    position: Integer representing position of the element
              of the theory tuple, ordered by <.
    operators: A tuple of theory operators.
    term: The theory term.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    position = IntegerField
    operators = id_terms.Theory_Operators.Field
    term = TheoryTermField


class Theory_Unparsed_Term(AstPredicate):
    """Predicate representing an unparsed theory term.
    An unparsed theory term consists of a tuple, each element of which
    consists of a tuple of theory operators and a theory term. This
    predicate represents an element of an unparsed theory term.

    id: The identifier of the unparsed theory term.
    elements: The tuple of aforementioned elements
              forming the unparsed theory term.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    elements = id_terms.Theory_Unparsed_Term_Elements.Field


TheoryTermField.fields.extend(
    [
        id_terms.Theory_Sequence.Field,
        id_terms.Theory_Function.Field,
        id_terms.Theory_Unparsed_Term.Field,
    ]
)


# Literals


class Guard(AstPredicate):
    """Predicate representing a guard for a comparison or aggregate atom.

    id: identifier of the guard.

    """

    id = Identifier_Field(default=lambda: next(id_count))
    comparison = ComparisonOperatorField
    term = TermField


class Guards(AstPredicate):
    """Predicate representing a tuple of guards for a comparison or aggregate atom.

    id: Identifier of the guard.
    position: Integer representing position of the element the tuple, ordered by <.
    element: Term identifying the element.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    position = IntegerField
    guard = id_terms.Guard.Field


class Comparison(AstPredicate):
    """Predicate representing a comparison atom.

    id: Identifier of the comparison.
    term: Leftmost term of the comparison atom.
    guards: tuple of guard predicates
    """

    id = Identifier_Field(default=lambda: next(id_count))
    term = TermField
    guards = id_terms.Guards.Field


BoolField = refine_field(ConstantField, ["true", "false"], name="BoolField")


class Boolean_Constant(AstPredicate):
    """Predicate representing a boolean constant, true or false.

    id: Identifier of the boolean constant
    value: Boolean value of the constant, represented by the constant
    terms 'true' and 'false'.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    value = BoolField


class Symbolic_Atom(AstPredicate):
    """Predicate representing a symbolic atom.

    id: Identifier of the atom.
    symbol: The function symbol constituting the atom.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    symbol = id_terms.Function.Field


AtomField = combine_fields_lazily(
    [
        id_terms.Symbolic_Atom.Field,
        id_terms.Comparison.Field,
        id_terms.Boolean_Constant.Field,
    ],
    name="AtomField",
)


class Literal(AstPredicate):
    """Predicate representing a literal.

    id: Identifier of the literal.
    sig: Sign of the literal, in string form. Possible values are
         "pos" "not" and "not not".
    atom: The atom constituting the literal.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    sig = SignField
    atom = AtomField


class Literals(AstPredicate):
    """Predicate representing an element of a tuple of literals.

    id: Identifier of the tuple.
    position: Integer representing position of the element the tuple, ordered by <.
    element: Term identifying the element.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    position = IntegerField
    literal = id_terms.Literal.Field


class Conditional_Literal(AstPredicate):
    """Predicate representing a conditional literal.

    id: Identifier of the conditional literal.
    literal: the literal in the head of the conditional literal
    condition: the tuple of literals forming the condition
    """

    id = Identifier_Field(default=lambda: next(id_count))
    literal = id_terms.Literal.Field
    condition = id_terms.Literals.Field


class Agg_Elements(AstPredicate):
    """Predicate representing an element of an implicit count aggregate.

    id: Identifier of the tuple.
    position: Integer representing position of the element the tuple, ordered by <.
    element: The conditional literal constituting the element of the aggregate.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    position = IntegerField
    element = id_terms.Conditional_Literal.Field


class Aggregate(AstPredicate):
    """Predicate representing an simple (count) aggregate atom with
    conditional literal elements.

    id: Identifier of the count aggregate.
    left_guard: The (optional) left guard of the aggregate, represented
                as a guard tuple of 0 or id_terms. element.
    elements: The tuple of conditional literals forming the elements of
              the aggregate.
    right_guard: The (optional) right guard of the aggregate, represented
                 as a guard tuple of 0 or id_terms. element.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    left_guard = id_terms.Guard.Field
    elements = id_terms.Agg_Elements.Field
    right_guard = id_terms.Guard.Field


class Theory_Atom_Elements(AstPredicate):
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

    id = Identifier_Field(default=lambda: next(id_count))
    position = IntegerField
    terms = id_terms.Theory_Terms.Field
    condition = id_terms.Literals.Field


class Theory_Guard(AstPredicate):
    """Predicate representing a theory guard.

    id: The identifier of the theory guard.
    operator_name: The name of the binary theory operator applied to
                   the aggregated elements of the theory atom and the theory term.
    term: The theory term to which the theory operator is applied.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    operator_name = StringField
    term = TheoryTermField


class Theory_Atom(AstPredicate):
    """Predicate representing a theory atom.

    id: Identifier of the theory atom.
    atom: The atom forming the symbolic part of the theory atom.
    elements: The tuple of elements aggregated in the theory atom.
    guard: The (optional) theory guard applied the aggregated elements
           of the theory atom.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    atom = id_terms.Symbolic_Atom.Field
    elements = id_terms.Theory_Atom_Elements.Field
    guard = id_terms.Theory_Guard.Field


class Body_Agg_Elements(AstPredicate):
    """Predicate representing an element of a body aggregate.

    id: Identifier of the tuple of body aggregate elements.
    position: Integer representing position of the element in the tuple, ordered by <.
    terms: The tuple of terms to which the aggregate function will be applied.
    condition: The tuple of literals forming a conjunction on which the
               tuple of terms is conditioned.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    position = IntegerField
    terms = id_terms.Terms.Field
    condition = id_terms.Literals.Field


class Body_Aggregate(AstPredicate):
    """Predicate representing an aggregate atom occurring in a body.

    id: Identifier of the body aggregate.
    left_guard: The (optional) left guard of the aggregate, represented
                as a guard tuple of 0 or id_terms. elements.
    function: The aggregate function applied to the terms of the aggregate
              remaining after evaluation of the conditions.
    elements: The elements of the aggregate,
    right_guard: The (optional) right guard of the aggregate, represented
                 as a guard tuple of 0 or id_terms. elements.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    left_guard = id_terms.Guard.Field
    function = AggregateFunctionField
    elements = id_terms.Body_Agg_Elements.Field
    right_guard = id_terms.Guard.Field


BodyAtomField = combine_fields_lazily(
    AtomField.fields
    + [  # noqa: W503
        id_terms.Aggregate.Field,
        id_terms.Body_Aggregate.Field,
        id_terms.Theory_Atom.Field,
    ],
    name="BodyAtomField",
)


class Body_Literal(AstPredicate):
    """Predicate representing a literal occurring in the body of a
    rule.

    id: Identifier of the literal.
    sig: Sign of the literal, in string form. Possible values are
         "pos" "not" and "not not".
    atom: The atom constituting the literal.

    """

    id = Identifier_Field(default=lambda: next(id_count))
    sig = SignField
    atom = BodyAtomField


BodyLiteralField = combine_fields_lazily(
    [id_terms.Body_Literal.Field, id_terms.Conditional_Literal.Field],
    name="BodyLiteralField",
)


class Body_Literals(AstPredicate):
    """Predicate representing an element of a tuple of literals
    occurring in the body of a statement.

    id: Identifier of the tuple.
    position: Integer representing position of the element the tuple, ordered by <.
    element:  Term identifying the element.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    position = IntegerField
    body_literal = BodyLiteralField


class Head_Agg_Elements(AstPredicate):
    """Predicate representing an element of a head aggregate.

    id: Identifier of the tuple of head aggregate elements.
    position: Integer representing position of the element the tuple, ordered by <.
    terms: The tuple of terms to which the aggregate function will be applied.
    condition: The conditional literal who's condition determines if
               the tuple of terms are added to the aggregate's set. When the condition
               holds, the head of the conditional literal is also derived.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    position = IntegerField
    terms = id_terms.Terms.Field
    condition = id_terms.Conditional_Literal.Field


class Head_Aggregate(AstPredicate):
    """Predicate representing an aggregate atom occuring in a head.

    id: Identifier of the head aggregate.
    left_guard: The (optional) left guard of the aggregate, represented
                as a guard tuple of 0 or id_terms. elements.
    function: The aggregate function applied to the term elements of the aggregate
              remaining after evaluation of their conditions.
    elements: The elements of the aggregate,
    right_guard: The (optional) right guard of the aggregate, represented
                 as a guard tuple of 0 or id_terms. elements.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    left_guard = id_terms.Guard.Field
    function = AggregateFunctionField
    elements = id_terms.Head_Agg_Elements.Field
    right_guard = id_terms.Guard.Field


class Conditional_Literals(AstPredicate):
    """Predicate representing an element of a tuple of conditional literals.

    id: Identifier of the tuple of conditional literals.
    position: Integer representing position of the element the tuple, ordered by <.
    conditional_literal: Term identifying the conditional literal element.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    position = IntegerField
    conditional_literal = id_terms.Conditional_Literal.Field


class Disjunction(AstPredicate):
    """Predicate representing a disjunction of (conditional) literals.

    id: Identifier of the disjunction.
    elements: The elements of the disjunction, a tuple of conditional literals.
             A literal in a disjunction is represented as a conditional literal
             with an empty condition.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    elements = id_terms.Conditional_Literals.Field


HeadField = combine_fields(
    [
        id_terms.Literal.Field,
        id_terms.Aggregate.Field,
        id_terms.Head_Aggregate.Field,
        id_terms.Disjunction.Field,
        id_terms.Theory_Atom.Field,
    ],
    name="HeadField",
)


class Rule(AstPredicate):

    """Predicate representing a rule statement.

    id: Identifier of the rule.
    head: The head of the rule.
    body: The body of the rule, a tuple of literals.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    head = HeadField
    body = id_terms.Body_Literals.Field


StatementField = combine_fields_lazily([id_terms.Rule.Field], name="StatementField")


class Statements(AstPredicate):
    """Predicate representing an element of a tuple of statements.

    id: Identifier of the tuple.
    position: Integer representing position of the element the tuple, ordered by <.
    element: Predicate identifying the element.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    position = IntegerField
    statement = StatementField


class Constants(AstPredicate):
    """Predicate representing an element of a tuple of constant terms.

    id: Identifier of the tuple.
    position: Integer representing position of the element the tuple, ordered by <.
    element: Predicate identifying the element.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    position = IntegerField
    constant = id_terms.Function.Field


class Program(AstPredicate):
    """Predicate representing a subprogram statement.

    name: The name of the subprogram, a string.
    parameters: The parameters of the subprogram, a tuple of constants.
    statements: The tuple of statements comprising the subprogram.
    """

    name = StringField
    parameters = id_terms.Constants.Field
    statements = id_terms.Statements.Field


# note that clingo's parser actually allows arbitrary constant as the external_type
# argument of External, but any other value than true or false results in the external
# statement having no effect


class External(AstPredicate):
    """Predicate representing an external statement.

    id: Identifier of the external statement.
    atom: The external atom.
    body: The tuple of literals the external statement is conditioned on.
    external_type: The default value of the external statement.
                   May be the constant 'true' or 'false'.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    atom = id_terms.Symbolic_Atom.Field
    body = id_terms.Body_Literals.Field
    external_type = BoolField


class Theory_Operator_Definitions(AstPredicate):
    id = Identifier_Field(default=lambda: next(id_count))
    position = IntegerField
    name = StringField
    priority = IntegerField
    operator_type = TheoryOperatorTypeField


class Theory_Term_Definitions(AstPredicate):
    id = Identifier_Field(default=lambda: next(id_count))
    position = IntegerField
    name = StringField
    operators = id_terms.Theory_Operator_Definitions.Field


class Theory_Guard_Definition(AstPredicate):
    id = Identifier_Field(default=lambda: next(id_count))
    operators = id_terms.Theory_Operators.Field
    term = StringField


class Theory_Atom_Definitions(AstPredicate):
    id = Identifier_Field(default=lambda: next(id_count))
    position = IntegerField
    atom_type = TheoryAtomTypeField
    name = StringField
    arity = IntegerField
    term = StringField
    guard = id_terms.Theory_Guard_Definition.Field


class Theory_Definition(AstPredicate):
    id = Identifier_Field(default=lambda: next(id_count))
    name = StringField
    terms = id_terms.Theory_Term_Definitions.Field
    atoms = id_terms.Theory_Atom_Definitions.Field


StatementField.fields.extend(
    [id_terms.External.Field, id_terms.Theory_Definition.Field]
)


AstPred = Union[
    String,
    Number,
    Variable,
    Unary_Operation,
    Binary_Operation,
    Interval,
    Terms,
    Function,
    Pool,
    Theory_Terms,
    Theory_Sequence,
    Theory_Function,
    Theory_Operators,
    Theory_Unparsed_Term_Elements,
    Theory_Unparsed_Term,
    Guard,
    Guards,
    Comparison,
    Boolean_Constant,
    Symbolic_Atom,
    Literal,
    Literals,
    Conditional_Literal,
    Agg_Elements,
    Aggregate,
    Theory_Atom_Elements,
    Theory_Guard,
    Theory_Atom,
    Body_Agg_Elements,
    Body_Aggregate,
    Body_Literal,
    Body_Literals,
    Head_Agg_Elements,
    Head_Aggregate,
    Conditional_Literals,
    Disjunction,
    Rule,
    Statements,
    Constants,
    Program,
    External,
    Theory_Operator_Definitions,
    Theory_Term_Definitions,
    Theory_Guard_Definition,
    Theory_Atom_Definitions,
    Theory_Definition,
]

AstPreds = [
    String,
    Number,
    Variable,
    Unary_Operation,
    Binary_Operation,
    Interval,
    Terms,
    Function,
    Pool,
    Theory_Terms,
    Theory_Sequence,
    Theory_Function,
    Theory_Operators,
    Theory_Unparsed_Term_Elements,
    Theory_Unparsed_Term,
    Guard,
    Guards,
    Comparison,
    Boolean_Constant,
    Symbolic_Atom,
    Literal,
    Literals,
    Conditional_Literal,
    Agg_Elements,
    Aggregate,
    Theory_Atom_Elements,
    Theory_Guard,
    Theory_Atom,
    Body_Agg_Elements,
    Body_Aggregate,
    Body_Literal,
    Body_Literals,
    Head_Agg_Elements,
    Head_Aggregate,
    Conditional_Literals,
    Disjunction,
    Rule,
    Statements,
    Constants,
    Program,
    External,
    Theory_Operator_Definitions,
    Theory_Term_Definitions,
    Theory_Guard_Definition,
    Theory_Atom_Definitions,
    Theory_Definition,
]

# Predicates for AST transformation


class Final(Predicate):
    """Wrapper predicate to distinguish output AST facts of a transformation."""

    ast = combine_fields([fact.Field for fact in AstPreds])
