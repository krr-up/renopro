"Definitions of enum fields to be used in AST predicates."
import enum

from clorm import ConstantField, StringField

from renopro.utils.clorm_utils import define_enum_field


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
