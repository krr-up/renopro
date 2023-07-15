"""Definitions of AST elements as clorm predicates."""
import enum
import inspect
from itertools import count
from typing import Sequence, Type, Union

from clingo import ast
from clorm import (
    BaseField,
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
Identifier_Field = combine_fields([IntegerField, RawField])

id_pred2ast_pred = dict()


def make_id_predicate(ast_pred):
    """Utility function to make a Predicate subclass with single
    identifier field.

    """
    id_pred_name = ast_pred.__name__ + "1"
    id_pred = type(
        id_pred_name,
        (Predicate,),
        {
            "id": Identifier_Field(default=lambda: next(id_count)),
            "Meta": type("Meta", tuple(), {"name": ast_pred.meta.name}),
        },
    )
    id_pred2ast_pred.update({id_pred: ast_pred})
    return id_pred


def combine_fields_lazily(
    fields: Sequence[Type[BaseField]], *, name: str = ""
) -> Type[BaseField]:
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

    def _pytocl(v):
        for f in fields:
            try:
                return f.pytocl(v)
            except (TypeError, ValueError, AttributeError):
                pass
        raise TypeError(f"No combined pytocl() match for value {v}.")

    def _cltopy(r):
        for f in fields:
            try:
                return f.cltopy(r)
            except (TypeError, ValueError):
                pass
        raise TypeError(
            (f"Object '{r}' ({type(r)}) failed to unify " f"with {subclass_name}.")
        )

    return type(
        subclass_name,
        (BaseField,),
        {"fields": fields, "pytocl": _pytocl, "cltopy": _cltopy},
    )


class String(Predicate):
    id = Identifier_Field
    value = StringField


String1 = make_id_predicate(String)


class Number(Predicate):
    id = Identifier_Field
    value = IntegerField


Number1 = make_id_predicate(Number)


class Variable(Predicate):
    id = Identifier_Field
    name = StringField


Variable1 = make_id_predicate(Variable)


Term_Field = combine_fields_lazily(
    [String1.Field, Number1.Field, Variable1.Field], name="Term"
)


class Term_Tuple(Predicate):
    id = Identifier_Field  # identifier of collection
    position = IntegerField  # 0 indexed position of element in collection
    element = Term_Field


Term_Tuple1 = make_id_predicate(Term_Tuple)


class Function(Predicate):
    """Note: we represent constants as a Function with an empty term
    tuple (i.e. no term_tuple fact with a matching identifier"""

    id = Identifier_Field
    name = ConstantField
    arguments = Term_Tuple1.Field


Function1 = make_id_predicate(Function)


Term_Field.fields.append(Function1.Field)


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


binary_operator_cl2ast = {
    BinaryOperator[op.name]: ast.BinaryOperator[op.name] for op in ast.BinaryOperator
}
binary_operator_ast2cl = {v: k for k, v in binary_operator_cl2ast.items()}


class Binary_Operation(Predicate):
    id = Identifier_Field
    operator = define_enum_field(
        parent_field=StringField, enum_class=BinaryOperator, name="OperatorField"
    )
    left = Term_Field
    right = Term_Field


Binary_Operation1 = make_id_predicate(Binary_Operation)


Term_Field.fields.append(Binary_Operation1.Field)


class Atom(Predicate):
    id = Identifier_Field
    symbol = Function1.Field


Atom1 = make_id_predicate(Atom)


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
    id = Identifier_Field
    sig = define_enum_field(parent_field=StringField, enum_class=Sign, name="SignField")
    atom = Atom1.Field


Literal1 = make_id_predicate(Literal)


class Literal_Tuple(Predicate):
    id = Identifier_Field
    position = IntegerField  # should we keep track of position?
    element = Literal1.Field


Literal_Tuple1 = make_id_predicate(Literal_Tuple)


class Rule(Predicate):
    id = Identifier_Field
    head = Literal1.Field
    body = Literal_Tuple1.Field


Rule1 = make_id_predicate(Rule)


# note that clingo's parser actually allows arbitrary constant as the external_type
# argument of External, but any other value than true or false results in the external
# statement having no effect
ExternalTypeField = refine_field(
    ConstantField, ["true", "false"], name="ExternalTypeField"
)


class External(Predicate):
    id = Identifier_Field
    atom = Atom1.Field
    body = Literal_Tuple1.Field
    external_type = ExternalTypeField


External1 = make_id_predicate(External)


Statement = combine_fields([Rule1.Field, External1.Field])


class Statement_Tuple(Predicate):
    id = Identifier_Field
    position = IntegerField
    element = Statement


Statement_Tuple1 = make_id_predicate(Statement_Tuple)


class Constant_Tuple(Predicate):
    id = Identifier_Field
    position = IntegerField
    element = Function1.Field


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
    Term_Tuple,
    Term_Tuple1,
    Function,
    Function1,
    Binary_Operation,
    Binary_Operation1,
    Atom,
    Atom1,
    Literal,
    Literal1,
    Literal_Tuple,
    Literal_Tuple1,
    Rule,
    Rule1,
    External,
    External1,
    Statement_Tuple,
    Statement_Tuple1,
    Constant_Tuple,
    Constant_Tuple1,
    Program,
]

AST_Predicates = [
    String,
    String1,
    Number,
    Number1,
    Variable,
    Variable1,
    Term_Tuple,
    Term_Tuple1,
    Function,
    Function1,
    Binary_Operation,
    Binary_Operation1,
    Atom,
    Atom1,
    Literal,
    Literal1,
    Literal_Tuple,
    Literal_Tuple1,
    Rule,
    Rule1,
    External,
    External1,
    Statement_Tuple,
    Statement_Tuple1,
    Constant_Tuple,
    Constant_Tuple1,
    Program,
]

AST_Fact = Union[
    String,
    Number,
    Variable,
    Function,
    Term_Tuple,
    Binary_Operation,
    Atom,
    Literal,
    Literal_Tuple,
    Rule,
    External,
    Statement_Tuple,
    Constant_Tuple,
    Program,
]

AST_Facts = [
    String,
    Number,
    Variable,
    Function,
    Term_Tuple,
    Binary_Operation,
    Atom,
    Literal,
    Literal_Tuple,
    Rule,
    External,
    Statement_Tuple,
    Constant_Tuple,
    Program,
]

# Predicates for AST transformation


class Final(Predicate):
    ast = combine_fields([fact.Field for fact in AST_Facts])
