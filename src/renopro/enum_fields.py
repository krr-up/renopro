"Definitions of enum fields to be used in AST predicates."
import enum
import inspect
from types import new_class
from typing import Type, TypeVar

from clorm import BaseField, ConstantField, StringField


def define_enum_field(
    parent_field: Type[BaseField], enum_class: Type[enum.Enum], *, name: str = ""
) -> Type[BaseField]:  # nocoverage
    """Factory function that returns a BaseField sub-class for an
    Enum. Essentially the same as the one defined in clorm, but stores
    the enum that defines the field under attribute 'enum' for later
    use.

    Enums are part of the standard library since Python 3.4. This method
    provides an alternative to using refine_field() to provide a restricted set
    of allowable values.

    Example:
       .. code-block:: python

          class IO(str,Enum):
              IN="in"
              OUT="out"

          # A field that unifies against ASP constants "in" and "out"
          IOField = define_enum_field(ConstantField,IO)

    Positional argument:

       field_class: the field that is being sub-classed

       enum_class: the Enum class

    Optional keyword-only arguments:

       name: name for new class (default: anonymously generated).

    """
    subclass_name = name if name else parent_field.__name__ + "_Restriction"
    if not inspect.isclass(parent_field) or not issubclass(parent_field, BaseField):
        raise TypeError(f"{parent_field} is not a subclass of BaseField")

    if not inspect.isclass(enum_class) or not issubclass(enum_class, enum.Enum):
        raise TypeError(f"{enum_class} is not a subclass of enum.Enum")

    values = set(i.value for i in enum_class)

    def _pytocl(py):
        val = py.value
        if val not in values:
            raise ValueError(
                f"'{val}' is not a valid value of enum class '{enum_class.__name__}'"
            )
        return val

    def body(ns):
        ns.update({"pytocl": _pytocl, "cltopy": enum_class, "enum": enum_class})

    return new_class(subclass_name, (parent_field,), {}, body)


A = TypeVar("A", bound=enum.Enum)
B = TypeVar("B", bound=enum.Enum)


def convert_enum(enum_member: A, other_enum: Type[B]) -> B:
    """Given an enum_member, convert it to the other_enum member of
    the same name.
    """
    try:
        return other_enum[enum_member.name]
    except KeyError as exc:  # nocoverage
        msg = (
            f"Enum {other_enum} has no corresponding member "
            f"with name {enum_member.name}"
        )
        raise ValueError(msg) from exc


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
    BinaryLeft = "binary_left"
    BinaryRight = "binary_right"
    Unary = "unary"


TheoryOperatorTypeField = define_enum_field(
    parent_field=ConstantField,
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
    parent_field=ConstantField, enum_class=TheoryAtomType, name="TheoryAtomTypeField"
)
