"""Definitions of AST elements as clorm predicates."""
from typing import Union, Sequence, Type
import enum
import inspect
from itertools import count

from clorm import (IntegerField, Predicate, StringField, RawField,
                   BaseField, combine_fields, define_enum_field,
                   ConstantField)
from clingo import ast

id_count = count()
next_id = lambda: next(id_count)

# by default we use integer identifiers, but allow arbitrary symbols as well
# for flexibility when these are user generated
Identifier = combine_fields([IntegerField, RawField])

id_pred2ast_pred = dict()


def make_id_predicate(ast_pred):
    """Utility function to make a Predicate subclass with single
    identifier field.

    """
    id_pred_name = ast_pred.__name__ + "1"
    id_pred = type(id_pred_name, (Predicate,),
                   {"id": Identifier(default=next_id),
                    "Meta": type("Meta", tuple(), {"name": ast_pred.meta.name})})
    id_pred2ast_pred.update({id_pred: ast_pred})
    return id_pred


def combine_fields_lazily(fields: Sequence[Type[BaseField]], *, name:
                          str = "") -> Type[BaseField]:
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
            raise TypeError("{} is not BaseField or a sub-class".format(f))
    if len(fields) < 2:
        raise TypeError("Must specify at least two fields to combine")

    fields = list(fields)

    def _pytocl(v):
        for f in fields:
            try:
                return f.pytocl(v)
            except (TypeError, ValueError, AttributeError):
                pass
        raise TypeError("No combined pytocl() match for value {}".format(v))

    def _cltopy(r):
        for f in fields:
            try:
                return f.cltopy(r)
            except (TypeError, ValueError):
                pass
        raise TypeError(
            "Object '{}' ({}) failed to unify with {}".format(r, type(r), subclass_name)
        )

    return type(subclass_name, (BaseField,), {"fields": fields,
                                              "pytocl": _pytocl, "cltopy": _cltopy})


class String(Predicate):
    id = Identifier
    value = StringField


String1 = make_id_predicate(String)


class Number(Predicate):
    id = Identifier
    value = IntegerField


Number1 = make_id_predicate(Number)


class Variable(Predicate):
    id = Identifier
    name = StringField


Variable1 = make_id_predicate(Variable)


class Constant(Predicate):
    id = Identifier
    name = ConstantField


Constant1 = make_id_predicate(Constant)


Term = combine_fields_lazily([String1.Field, Number1.Field,
                              Variable1.Field, Constant1.Field], name="Term")


class Term_Tuple(Predicate):
    id = Identifier  # identifier of collection
    position = IntegerField  # 0 indexed position of element in collection
    element = Term


Term_Tuple1 = make_id_predicate(Term_Tuple)


class Function(Predicate):
    id = Identifier
    name = ConstantField
    arguments = Term_Tuple1.Field


Function1 = make_id_predicate(Function)


Term.fields.append(Function1.Field)


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


binary_operator_cl2ast = {BinaryOperator[op.name]: ast.BinaryOperator[op.name]
                          for op in ast.BinaryOperator}
binary_operator_ast2cl = {v: k for k, v in binary_operator_cl2ast.items()}


class Binary_Operation(Predicate):
    id = Identifier
    operator = define_enum_field(parent_field=StringField,
                                 enum_class=BinaryOperator,
                                 name="OperatorField")
    left = Term
    right = Term


Binary_Operation1 = make_id_predicate(Binary_Operation)


Term.fields.append(Binary_Operation1.Field)

# an atom may be a predicate (Function) or propositional constant (Constant)
Atom = combine_fields([Function1.Field, Constant1.Field], name="Atom")


class Sign(str, enum.Enum):
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


sign_cl2ast = {Sign[op.name]: ast.Sign[op.name] for op in ast.Sign}
sign_ast2cl = {v: k for k, v in sign_cl2ast.items()}


class Literal(Predicate):
    id = Identifier
    sig = define_enum_field(parent_field=StringField,
                            enum_class=Sign,
                            name="SignField")
    atom = Atom


Literal1 = make_id_predicate(Literal)


class Literal_Tuple(Predicate):
    id = Identifier
    position = IntegerField  # should we keep track of position?
    element = Literal1.Field


Literal_Tuple1 = make_id_predicate(Literal_Tuple)


class Rule(Predicate):
    id = Identifier
    head = Literal1.Field
    body = Literal_Tuple1.Field


Rule1 = make_id_predicate(Rule)

# will be changed to combine_fileds once more statements are implemented
Statement = Rule1.Field


class Statement_Tuple(Predicate):
    id = Identifier
    position = IntegerField
    element = Statement


Statement_Tuple1 = make_id_predicate(Statement_Tuple)


class Constant_Tuple(Predicate):
    id = Identifier
    position = IntegerField
    element = Constant1.Field


Constant_Tuple1 = make_id_predicate(Constant_Tuple)


class Program(Predicate):
    name = StringField
    # restrict to tuple of strings as that's what clingo allows.
    parameters = Constant_Tuple1.Field
    statements = Statement_Tuple1.Field


AST_Predicate = Union[
    String,
    String1,
    Number,
    Number1,
    Variable,
    Variable1,
    Constant,
    Constant1,
    Term_Tuple,
    Term_Tuple1,
    Function,
    Function1,
    Binary_Operation,
    Binary_Operation1,
    Literal,
    Literal1,
    Literal_Tuple,
    Literal_Tuple1,
    Rule,
    Rule1,
    Statement_Tuple,
    Statement_Tuple1,
    Constant_Tuple,
    Constant_Tuple1,
    Program
]

AST_Predicates = [
    String,
    String1,
    Number,
    Number1,
    Variable,
    Variable1,
    Constant,
    Constant1,
    Term_Tuple,
    Term_Tuple1,
    Function,
    Function1,
    Binary_Operation,
    Binary_Operation1,
    Literal,
    Literal1,
    Literal_Tuple,
    Literal_Tuple1,
    Rule,
    Rule1,
    Statement_Tuple,
    Statement_Tuple1,
    Constant_Tuple,
    Constant_Tuple1,
    Program
]

AST_Fact = Union[
    String,
    Number,
    Variable,
    Constant,
    Function,
    Term_Tuple,
    Binary_Operation,
    Literal,
    Literal_Tuple,
    Rule,
    Statement_Tuple,
    Constant_Tuple,
    Program
]

AST_Facts = [
    String,
    Number,
    Variable,
    Constant,
    Function,
    Term_Tuple,
    Binary_Operation,
    Literal,
    Literal_Tuple,
    Rule,
    Statement_Tuple,
    Constant_Tuple,
    Program
]

# Predicates for AST transformation


class Final(Predicate):
    ast = combine_fields([fact.Field for fact in AST_Facts])
