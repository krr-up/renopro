# Definitions of AST elements as clorm predicates
# Note that the full input language of gringo is not represented.
from typing import Union

from clorm import IntegerField, Predicate, StringField, combine_fields


class String(Predicate):
    value = StringField


class Integer(Predicate):
    value = IntegerField


class Variable(Predicate):
    name = StringField


class Term_Tuple_Id(Predicate):
    identifier = IntegerField

    class Meta:
        name = "term_tuple"


class Function(Predicate):
    name = StringField
    arguments = Term_Tuple_Id.Field


Term = combine_fields([String.Field, Integer.Field, Variable.Field, Function.Field])


class Term_Tuple(Predicate):
    identifier = IntegerField  # identifier of collection
    position = IntegerField  # 0 indexed position of element in collection
    element = Term


class Literal(Predicate):
    sign_ = IntegerField
    atom = Function.Field


class Literal_Tuple_Id(Predicate):
    identifier = IntegerField

    class Meta:
        name = "literal_tuple"


class Literal_Tuple(Predicate):
    identifier = IntegerField
    position = IntegerField  # should we keep track of position?
    element = Literal.Field


class Rule(Predicate):
    head = Literal.Field
    body = Literal_Tuple_Id.Field


AST_Predicate = Union[
    String,
    Integer,
    Variable,
    Term_Tuple_Id,
    Function,
    Term_Tuple,
    Literal,
    Literal_Tuple_Id,
    Literal_Tuple,
    Rule,
]
