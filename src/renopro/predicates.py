"""Definitions of AST elements as clorm predicates."""
import enum
import inspect
from itertools import count
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
    fields: Sequence[Type[BaseField]], *, name: str = ""
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


# Terms


class String(Predicate):
    """Predicate representing a string term.

    id: Identifier of the string term.
    value: Value of string term, a string term itself.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    value = StringField


class String1(ComplexTerm, name="string"):
    "Term identifying a child string fact."
    id = Identifier_Field(default=lambda: next(id_count))


class Number(Predicate):
    """Predicate representing an integer term.

    id: Identifier of the integer term.
    value: Value of integer term, an integer term itself.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    value = IntegerField


class Number1(ComplexTerm, name="number"):
    "Term identifying a child number fact."
    id = Identifier_Field(default=lambda: next(id_count))


class Variable(Predicate):
    """Predicate representing a variable term.

    id: Identifier of variable term.
    value: Value of variable term, a string term.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    name = StringField


class Variable1(ComplexTerm, name="variable"):
    "Term identifying a child variable fact."
    id = Identifier_Field(default=lambda: next(id_count))


TermField = combine_fields_lazily(
    [String1.Field, Number1.Field, Variable1.Field], name="TermField"
)


class Unary_Operation(Predicate):
    """Predicate representing a unary operation term.

    id: Identifier of the unary operation.
    operator: A clingo unary operator, in string form.
    argument: The term argument the unary operator is applied to.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    operator = UnaryOperatorField
    argument = TermField


class Unary_Operation1(ComplexTerm, name="unary_operation"):
    "Term identifying a child unary operation fact."
    id = Identifier_Field(default=lambda: next(id_count))


class Binary_Operation(Predicate):
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


class Binary_Operation1(ComplexTerm, name="binary_operation"):
    "Term identifying a child binary operation fact."
    id = Identifier_Field(default=lambda: next(id_count))


class Interval(Predicate):
    """Predicate representing an interval term.

    id: Identifier of the interval.
    left: Left bound of the interval.
    right: Right bound of the interval.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    left = TermField
    right = TermField


class Interval1(ComplexTerm, name="interval"):
    "Term identifying a child interval fact."
    id = Identifier_Field(default=lambda: next(id_count))


class Terms(Predicate):
    """Predicate representing an element of a tuple of terms.

    id: Identifier of the tuple.
    position: Integer representing position of the element the tuple, ordered by <.
    element: Term identifying the element.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    position = IntegerField
    element = TermField


class Terms1(ComplexTerm, name="terms"):
    "Term identifying a child tuple of terms."
    id = Identifier_Field(default=lambda: next(id_count))


class Function(Predicate):
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
    arguments = Terms1.Field


class Function1(ComplexTerm, name="function"):
    "Term identifying a child function fact."
    id = Identifier_Field(default=lambda: next(id_count))


class Pool(Predicate):
    """Predicate representing a pool of terms.

    id: Identifier of the pool.
    arguments: Terms forming the pool.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    arguments = Terms1.Field


class Pool1(ComplexTerm, name="pool"):
    "Term identifying a child pool fact."
    id = Identifier_Field(default=lambda: next(id_count))


TermField.fields.extend(
    [
        Unary_Operation1.Field,
        Binary_Operation1.Field,
        Interval1.Field,
        Function1.Field,
        Pool1.Field,
    ]
)


# Literals


class Guard(Predicate):
    """Predicate representing a guard for a comparison or aggregate atom.

    id: identifier of the

    """

    id = Identifier_Field(default=lambda: next(id_count))
    comparison = ComparisonOperatorField
    term = TermField


class Guard1(Predicate, name="guard"):
    "Term identifying a child guard fact."
    id = Identifier_Field(default=lambda: next(id_count))


class Guards(Predicate):
    """Predicate representing a tuple of guards for a comparison or aggregate atom.

    id: Identifier of the guard.
    position: Integer representing position of the element the tuple, ordered by <.
    element: Term identifying the element.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    position = IntegerField
    element = Guard1.Field


class Guards1(ComplexTerm, name="guards"):
    "Term identifying a child guard tuple fact."
    id = Identifier_Field(default=lambda: next(id_count))


class Comparison(Predicate):
    """Predicate representing a comparison atom.

    id: Identifier of the comparison.
    term: Leftmost term of the comparison atom.
    guards: tuple of guard predicates
    """

    id = Identifier_Field(default=lambda: next(id_count))
    term = TermField
    guards = Guards1.Field


class Comparison1(ComplexTerm, name="comparison"):
    "Term identifying a child comparison fact"
    id = Identifier_Field(default=lambda: next(id_count))


BoolField = refine_field(ConstantField, ["true", "false"], name="BoolField")


class Boolean_Constant(Predicate):
    """Predicate representing a boolean constant, true or false.

    id: Identifier of the boolean constant
    value: Boolean value of the constant, represented by the constant
    terms 'true' and 'false'.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    value = BoolField


class Boolean_Constant1(ComplexTerm, name="boolean_constant"):
    "Term identifying a child boolean_constant fact."
    id = Identifier_Field(default=lambda: next(id_count))


class Symbolic_Atom(Predicate):
    """Predicate representing a symbolic atom.

    id: Identifier of the atom.
    symbol: The function symbol constituting the atom.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    symbol = Function1.Field


class Symbolic_Atom1(ComplexTerm, name="symbolic_atom"):
    "Term identifying a child symbolic atom fact"
    id = Identifier_Field(default=lambda: next(id_count))


AtomField = combine_fields_lazily(
    [Symbolic_Atom1.Field, Comparison1.Field, Boolean_Constant1.Field], name="AtomField"
)


class Literal(Predicate):
    """Predicate representing a literal.

    id: Identifier of the literal.
    sig: Sign of the literal, in string form. Possible values are
         "pos" "not" and "not not".
    atom: The atom constituting the literal.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    sig = SignField
    atom = AtomField


class Literal1(ComplexTerm, name="literal"):
    "Term identifying a child literal fact."
    id = Identifier_Field(default=lambda: next(id_count))


class Literals(Predicate):
    """Predicate representing an element of a tuple of (conditional) literals.

    id: Identifier of the tuple.
    position: Integer representing position of the element the tuple, ordered by <.
    element: Term identifying the element.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    position = IntegerField  # should we keep track of position?
    element = Literal1.Field


class Literals1(ComplexTerm, name="literals"):
    "Term identifying a child literal tuple."
    id = Identifier_Field(default=lambda: next(id_count))


class Conditional_Literal(Predicate):
    """Predicate representing a conditional literal.

    id: Identifier of the conditional literal.
    literal: the literal in the head of the conditional literal
    condition: the tuple of literals forming the condition
    """

    id = Identifier_Field(default=lambda: next(id_count))
    literal = Literal1.Field
    condition = Literals1.Field


class Conditional_Literal1(ComplexTerm, name="conditional_literal"):
    "Term identifying a child conditional literal."
    id = Identifier_Field(default=lambda: next(id_count))


class Agg_Elements(Predicate, name="agg_elements"):
    """Predicate representing an element of an implicit count aggregate.

    id: Identifier of the tuple.
    position: Integer representing position of the element the tuple, ordered by <.
    element: The conditional literal constituting the element of the aggregate.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    position = IntegerField
    element = Conditional_Literal1.Field


class Agg_Elements1(ComplexTerm, name="agg_elements"):
    "Term identifying a child tuple elements of a count aggregate."
    id = Identifier_Field(default=lambda: next(id_count))


class Aggregate(Predicate):
    """Predicate representing an (implicit) count aggregate atom with
    conditional literal elements.

    id: Identifier of the count aggregate.
    left_guard: The (optional) left guard of the aggregate, represented
                as a guard tuple of 0 or 1 element.
    elements: The tuple of conditional literals forming the elements of
              the aggregate.
    right_guard: The (optional) right guard of the aggregate, represented
                 as a guard tuple of 0 or 1 element.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    left_guard = Guard1.Field
    elements = Agg_Elements1.Field
    right_guard = Guard1.Field


class Aggregate1(ComplexTerm, name="aggregate"):
    "Term identifying a child count aggregate."
    id = Identifier_Field(default=lambda: next(id_count))


class Body_Agg_Elements(Predicate, name="body_agg_elements"):
    """Predicate representing an element of a body aggregate.

    id: Identifier of the tuple of body aggregate elements.
    position: Integer representing position of the element the tuple, ordered by <.
    terms: The tuple of terms to which the aggregate function will be applied.
    condition: The tuple of literals on which the tuple of terms is conditioned.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    position = IntegerField
    terms = Terms1.Field
    condition = Literals1.Field


class Body_Agg_Elements1(ComplexTerm, name="body_agg_elements"):
    "Term identifying the child tuple of elements in a body aggregate."
    id = Identifier_Field(default=lambda: next(id_count))


class Body_Aggregate(Predicate):
    """Predicate representing an aggregate atom occurring in a body.

    id: Identifier of the body aggregate.
    left_guard: The (optional) left guard of the aggregate, represented
                as a guard tuple of 0 or 1 elements.
    function: The aggregate function applied to the terms of the aggregate
              remaining after evaluation of the conditions.
    elements: The elements of the aggregate,
    right_guard: The (optional) right guard of the aggregate, represented
                 as a guard tuple of 0 or 1 elements.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    left_guard = Guard1.Field
    function = AggregateFunctionField
    elements = Body_Agg_Elements1.Field
    right_guard = Guard1.Field


class Body_Aggregate1(Predicate, name="body_aggregate"):
    "Term identifying a child body aggregate"
    id = Identifier_Field(default=lambda: next(id_count))


ExtendedBodyAtomField = combine_fields(
    [Aggregate1.Field, Body_Aggregate1.Field], name="ExtendedBodyAtomField"
)


# could not get inheritance to work for predicate definitions.
# should ask Dave about how to set this up. Ideally Body_Literal
# would just inherit from Literal and override the atom field.


class Extended_Body_Literal(Predicate, name="literal"):
    """Predicate representing a literal occurring in the body of a
    rule, the atom of which is a (body) aggregate or theory atom.

    id: Identifier of the literal.
    sig: Sign of the literal, in string form. Possible values are
         "pos" "not" and "not not".
    atom: The aggregate or theory atom constituting the literal.

    """

    id = Identifier_Field(default=lambda: next(id_count))
    sig = SignField
    atom = ExtendedBodyAtomField


class Extended_Body_Literal1(Literal, name="literal"):
    "Term identifying a child body aggregate literal."
    id = Identifier_Field(default=lambda: next(id_count))


BodyLiteralField = combine_fields(
    [Literal1.Field, Conditional_Literal1.Field, Extended_Body_Literal1.Field],
    name="BodyLiteralField",
)


class Body_Literals(Predicate):
    """Predicate representing an element of a tuple of literals
    occurring in the body of a statement.

    id: Identifier of the tuple.
    position: Integer representing position of the element the tuple, ordered by <.
    element:  Term identifying the element.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    position = IntegerField
    element = BodyLiteralField


class Body_Literals1(ComplexTerm, name="body_literals"):
    "Term identifying a child tuple of body literals."
    id = Identifier_Field(default=lambda: next(id_count))


class Head_Agg_Elements(Predicate, name="head_agg_elements"):
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
    terms = Terms1.Field
    condition = Conditional_Literal1.Field


class Head_Agg_Elements1(ComplexTerm, name="head_agg_elements"):
    "Term identifying the child tuple of elements in a head aggregate."
    id = Identifier_Field(default=lambda: next(id_count))


class Head_Aggregate(Predicate):
    """Predicate representing an aggregate atom occuring in a head.

    id: Identifier of the head aggregate.
    left_guard: The (optional) left guard of the aggregate, represented
                as a guard tuple of 0 or 1 elements.
    function: The aggregate function applied to the term elements of the aggregate
              remaining after evaluation of their conditions.
    elements: The elements of the aggregate,
    right_guard: The (optional) right guard of the aggregate, represented
                 as a guard tuple of 0 or 1 elements.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    left_guard = Guard1.Field
    function = AggregateFunctionField
    elements = Head_Agg_Elements1.Field
    right_guard = Guard1.Field


class Head_Aggregate1(Predicate, name="head_aggregate"):
    "Term identifying a child body aggregate"
    id = Identifier_Field(default=lambda: next(id_count))


class Disjunction(Predicate):
    """Predicate representing a disjunction of (conditional) literals.

    id: Identifier of the disjunction.
    elements: The conditional literals constituting the disjunction.
              Literals occurring in the disjunction are represented
              as a conditional literal with an empty condition.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    position = IntegerField
    element = Conditional_Literal1.Field


class Disjunction1(ComplexTerm, name="disjunction"):
    "Term representing a child disjunction."
    id = Identifier_Field(default=lambda: next(id_count))


HeadField = combine_fields(
    [Literal1.Field, Aggregate1.Field, Head_Aggregate1.Field, Disjunction1.Field],
    name="HeadField",
)


class Rule(Predicate):

    """Predicate representing a rule statement.

    id: Identifier of the rule.
    head: The head of the rule.
    body: The body of the rule, a tuple of literals.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    head = HeadField
    body = Body_Literals1.Field


class Rule1(ComplexTerm, name="rule"):
    "Term identifying a child rule fact."
    id = Identifier_Field(default=lambda: next(id_count))


# note that clingo's parser actually allows arbitrary constant as the external_type
# argument of External, but any other value than true or false results in the external
# statement having no effect


class External(Predicate):
    """Predicate representing an external statement.

    id: Identifier of the external statement.
    atom: The external atom.
    body: The tuple of literals the external statement is conditioned on.
    external_type: The default value of the external statement.
                   May be the constant 'true' or 'false'.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    atom = Symbolic_Atom1.Field
    body = Body_Literals1.Field
    external_type = BoolField


class External1(ComplexTerm, name="external"):
    "Term identifying a child external fact."
    id = Identifier_Field(default=lambda: next(id_count))


StatementField = combine_fields([Rule1.Field, External1.Field], name="StatementField")


class Statements(Predicate):
    """Predicate representing an element of a tuple of statements.

    id: Identifier of the tuple.
    position: Integer representing position of the element the tuple, ordered by <.
    element: Predicate identifying the element.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    position = IntegerField
    element = StatementField


class Statements1(ComplexTerm, name="statements"):
    "Term identifying a child statement tuple"
    id = Identifier_Field(default=lambda: next(id_count))


class Constants(Predicate):
    """Predicate representing an element of a tuple of constant terms.

    id: Identifier of the tuple.
    position: Integer representing position of the element the tuple, ordered by <.
    element: Predicate identifying the element.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    position = IntegerField
    element = Function1.Field


class Constants1(ComplexTerm, name="constants"):
    "Term identifying a child constant tuple."
    id = Identifier_Field(default=lambda: next(id_count))


class Program(Predicate):
    """Predicate representing a subprogram statement.

    name: The name of the subprogram, a string.
    parameters: The parameters of the subprogram, a tuple of constants.
    statements: The tuple of statements comprising the subprogram.
    """

    name = StringField
    parameters = Constants1.Field
    statements = Statements1.Field


AstPredicate = Union[
    String,
    Number,
    Variable,
    Unary_Operation,
    Binary_Operation,
    Interval,
    Terms,
    Function,
    Pool,
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
    Body_Agg_Elements,
    Body_Aggregate,
    Extended_Body_Literal,
    Body_Literals,
    Head_Agg_Elements,
    Head_Aggregate,
    Disjunction,
    Rule,
    Statements,
    Constants,
    Program,
    External,
]

AstPredicates = [
    String,
    Number,
    Variable,
    Unary_Operation,
    Binary_Operation,
    Interval,
    Terms,
    Function,
    Pool,
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
    Body_Agg_Elements,
    Body_Aggregate,
    Extended_Body_Literal,
    Body_Literals,
    Head_Agg_Elements,
    Head_Aggregate,
    Disjunction,
    Rule,
    Statements,
    Constants,
    Program,
    External,
]

# Predicates for AST transformation


class Final(Predicate):
    """Wrapper predicate to distinguish output AST facts of a transformation."""

    ast = combine_fields([fact.Field for fact in AstPredicates])
