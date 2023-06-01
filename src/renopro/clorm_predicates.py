"""Definitions of AST elements as clorm predicates."""
from typing import Union, Iterable
import enum

from clorm import (IntegerField, Predicate, StringField, RawField,
                   BaseField, combine_fields, define_enum_field)
from clingo import ast

# by default we use integer identifiers, but allow arbitrary symbols as well
# for flexibility when these are user generated
Identifier = combine_fields([IntegerField, RawField])

term_fields = list()


class Term(BaseField):

    def pytocl(v):
        global term_fields
        for f in term_fields:
            try:
                return f.pytocl(v)
            except (TypeError, ValueError, AttributeError):
                pass
        raise TypeError("No combined pytocl() match for value {}".format(v))

    def cltopy(r):
        global term_fields
        for f in term_fields:
            try:
                return f.cltopy(r)
            except (TypeError, ValueError):
                pass
        raise TypeError(
            f"Object '{r}' ({type(r)}) failed to unify with Term"
        )


class String(Predicate):
    value = StringField


class Number(Predicate):
    value = IntegerField


class Variable(Predicate):
    name = StringField


class Term_Tuple_Id(Predicate):
    identifier = Identifier

    class Meta:
        name = "term_tuple"


class Function(Predicate):
    name = StringField
    arguments = Term_Tuple_Id.Field


term_fields.extend([String.Field, Number.Field, Variable.Field,
                    Function.Field])


# Term = combine_fields([String.Field, Integer.Field, Variable.Field,
#                        Function.Field])


class BinaryOperator(str, enum.Enum):
    And = "&"  # bitwise and
    Division = "/"  # arithmetic division
    Minus = "-"  # arithmetic subtraction
    Modulo = "%"  # arithmetic modulo
    Multiplication = "*"  # arithmetic multiplication
    Or = "?"  # bitwise or
    Plus = "+"  # arithmetic addition
    Power = "**"  # arithmetic exponentiation
    XOr = "^"  # bitwise exclusive or


class Binary_Operation(Predicate):
    operator = define_enum_field(parent_field=StringField,
                                 enum_class=BinaryOperator,
                                 name="OperatorField")
    left = Term
    right = Term


term_fields.append(Binary_Operation.Field)


class Term_Tuple(Predicate):
    identifier = Identifier  # identifier of collection
    position = IntegerField  # 0 indexed position of element in collection
    element = Term


class Literal(Predicate):
    sign_ = IntegerField
    atom = Function.Field


class Literal_Tuple_Id(Predicate):
    identifier = Identifier

    class Meta:
        name = "literal_tuple"


class Literal_Tuple(Predicate):
    identifier = Identifier
    position = IntegerField  # should we keep track of position?
    element = Literal.Field


class Rule(Predicate):
    head = Literal.Field
    body = Literal_Tuple_Id.Field


AST_Predicate = Union[
    String,
    Number,
    Variable,
    Term_Tuple_Id,
    Function,
    Term_Tuple,
    Literal,
    Literal_Tuple_Id,
    Literal_Tuple,
    Rule
]

AST_Predicates = [
    String,
    Number,
    Variable,
    Term_Tuple_Id,
    Function,
    Term_Tuple,
    Literal,
    Literal_Tuple_Id,
    Literal_Tuple,
    Rule
]


AST_Fact = Union[
    Term_Tuple,
    Literal_Tuple,
    Rule
]

AST_Facts = [
    Term_Tuple,
    Literal_Tuple,
    Rule
]

# Predicates for AST transformation


class Final(Predicate):
    ast = RawField
