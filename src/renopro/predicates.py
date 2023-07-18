"""Definitions of AST elements as clorm predicates."""
import enum
import inspect
from itertools import count
from typing import List, Sequence, Type, Union

from clingo import ast
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
Identifier_Field = combine_fields([IntegerField, RawField])


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


class String(Predicate):
    """Predicate representing a string term.

    id: Identifier of the string term.
    value: Value of string term, a string term itself.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    value = StringField


class String1(ComplexTerm):
    "Term identifying a child string predicate."
    id = Identifier_Field(default=lambda: next(id_count))

    class Meta:
        # pylint: disable=too-few-public-methods, missing-class-docstring
        name = "string"


class Number(Predicate):
    """Predicate representing an integer term.

    id: Identifier of the integer term.
    value: Value of integer term, an integer term itself."""

    id = Identifier_Field(default=lambda: next(id_count))
    value = IntegerField


class Number1(ComplexTerm):
    "Term identifying a child number predicate."
    id = Identifier_Field(default=lambda: next(id_count))

    class Meta:
        # pylint: disable=too-few-public-methods, missing-class-docstring
        name = "number"


class Variable(Predicate):
    """Predicate representing a variable term.

    id: Identifier of variable term.
    value: Value of variable term, a string term.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    name = StringField


class Variable1(ComplexTerm):
    "Term identifying a child variable predicate."
    id = Identifier_Field(default=lambda: next(id_count))

    class Meta:
        # pylint: disable=too-few-public-methods, missing-class-docstring
        name = "variable"


Term_Field = combine_fields_lazily(
    [String1.Field, Number1.Field, Variable1.Field], name="Term"
)


class Term_Tuple(Predicate):
    """Predicate representing an element of a tuple of terms.

    id: Identifier of the tuple.
    position: Position of the element the tuple, given by < relation over integers.
    element: Predicate identifying the element.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    position = IntegerField
    element = Term_Field


class Term_Tuple1(ComplexTerm):
    "Term identifying a child term tuple predicate."
    id = Identifier_Field(default=lambda: next(id_count))

    class Meta:
        # pylint: disable=too-few-public-methods, missing-class-docstring
        name = "term_tuple"


class Function(Predicate):
    """Predicate representing a function symbol with term arguments.
    Note that we represent function terms and constant terms as well as symbolic
    atoms and propositional constants via this predicate.

    id: Identifier of the function.
    name: Symbolic name of the function, a constant term.
    arguments: Term tuple predicate identifying the function's arguments.
               If there are no elements of the term tuple with a matching
               identifier, the function has no arguments and is thus a constant.

    """

    id = Identifier_Field(default=lambda: next(id_count))
    name = ConstantField
    arguments = Term_Tuple1.Field


class Function1(ComplexTerm):
    "Term identifying a child function predicate."
    id = Identifier_Field(default=lambda: next(id_count))

    class Meta:
        # pylint: disable=too-few-public-methods, missing-class-docstring
        name = "function"


Term_Field.fields.append(Function1.Field)


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


binary_operator_cl2ast = {
    BinaryOperator[op.name]: ast.BinaryOperator[op.name] for op in ast.BinaryOperator
}
binary_operator_ast2cl = {v: k for k, v in binary_operator_cl2ast.items()}


class Binary_Operation(Predicate):
    """Predicate representing a binary operation term.

    id: Identifier of the binary operation.
    operator: A clingo binary operator, in string form.
    left: Predicate identifying the term that is the left operand of the operation.
    right: Predicate identifying the term that is the right operand of the operation.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    operator = define_enum_field(
        parent_field=StringField, enum_class=BinaryOperator, name="OperatorField"
    )
    left = Term_Field
    right = Term_Field


class Binary_Operation1(ComplexTerm):
    "Term identifying a child binary operation predicate."
    id = Identifier_Field(default=lambda: next(id_count))

    class Meta:
        # pylint: disable=too-few-public-methods, missing-class-docstring
        name = "binary_operation"


Term_Field.fields.append(Binary_Operation1.Field)


class Interval(Predicate):
    """Predicate representing an interval term.

    id: Identifier of the interval.
    left: Left bound of the interval.
    right: Right bound of the interval.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    left = Term_Field
    right = Term_Field


class Interval1(ComplexTerm):
    "Term identifying a child interval predicate."
    id = Identifier_Field(default=lambda: next(id_count))

    class Meta:
        # pylint: disable=too-few-public-methods, missing-class-docstring
        name = "interval"


Term_Field.fields.append(Interval1.Field)


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


comp_operator_cl2ast = {
    ComparisonOperator[op.name]: ast.ComparisonOperator[op.name]
    for op in ast.ComparisonOperator
}
comp_operator_ast2cl = {v: k for k, v in comp_operator_cl2ast.items()}


class Guard_Tuple(Predicate):
    """Predicate representing a tuple of guards for a comparison or aggregate atom.

    id: Identifier of the guard.
    position: Position of the element the tuple, given by < relation over integers.
    comparison: The clingo comparison operator, in string form.
    term: The term serving as the guard.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    position = IntegerField
    comparison = define_enum_field(
        parent_field=StringField, enum_class=ComparisonOperator, name="ComparisonField"
    )
    term = Term_Field


class Guard_Tuple1(ComplexTerm):
    "Term identifying a child guard tuple predicate."
    id = Identifier_Field(default=lambda: next(id_count))

    class Meta:
        # pylint: disable=too-few-public-methods, missing-class-docstring
        name = "guard_tuple"


class Comparison(Predicate):
    """Predicate representing a comparison atom.

    id: Identifier of the comparison.
    term: Leftmost term of the comparison atom.
    guards: tuple of guard predicates
    """

    id = Identifier_Field(default=lambda: next(id_count))
    term = Term_Field
    guards = Guard_Tuple1.Field


class Comparison1(ComplexTerm):
    "Term identifying a child comparison predicate"
    id = Identifier_Field(default=lambda: next(id_count))

    class Meta:
        # pylint: disable=too-few-public-methods, missing-class-docstring
        name = "comparison"


class Symbolic_Atom(Predicate):
    """Predicate representing a symbolic atom.

    id: Identifier of the atom.
    symbol: The function symbol constituting the atom.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    symbol = Function1.Field


class Symbolic_Atom1(ComplexTerm):
    "Term identifying a child symbolic atom predicate"
    id = Identifier_Field(default=lambda: next(id_count))

    class Meta:
        # pylint: disable=too-few-public-methods, missing-class-docstring
        name = "symbolic_atom"


AtomField = combine_fields_lazily(
    [Symbolic_Atom1.Field, Comparison1.Field], name="AtomField"
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


sign_cl2ast = {Sign[op.name]: ast.Sign[op.name] for op in ast.Sign}
sign_ast2cl = {v: k for k, v in sign_cl2ast.items()}


class Literal(Predicate):
    """Predicate representing a literal.

    id: Identifier of the literal.
    sig: Sign of the literal, in string form. Possible values are
         "pos" "not" and "not not".
    atom: The atom constituting the literal.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    sig = define_enum_field(parent_field=StringField, enum_class=Sign, name="SignField")
    atom = AtomField


class Literal1(ComplexTerm):
    "Term identifying a child literal predicate."
    id = Identifier_Field(default=lambda: next(id_count))

    class Meta:
        # pylint: disable=too-few-public-methods, missing-class-docstring
        name = "literal"


class Literal_Tuple(Predicate):
    """Predicate representing an element of a tuple of literals.

    id: Identifier of the tuple.
    position: Position of the element the tuple, given by < relation over integers.
    element: Predicate identifying the element.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    position = IntegerField  # should we keep track of position?
    element = Literal1.Field


class Literal_Tuple1(ComplexTerm):
    "Term identifying a child literal tuple."
    id = Identifier_Field(default=lambda: next(id_count))

    class Meta:
        # pylint: disable=too-few-public-methods, missing-class-docstring
        name = "literal_tuple"


class Rule(Predicate):
    """Predicate representing a rule statement.

    id: Identifier of the rule.
    head: The head of the rule.
    body: The body of the rule, a tuple of literals.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    head = Literal1.Field
    body = Literal_Tuple1.Field


class Rule1(ComplexTerm):
    "Term identifying a child rule predicate."
    id = Identifier_Field(default=lambda: next(id_count))

    class Meta:
        # pylint: disable=too-few-public-methods, missing-class-docstring
        name = "rule"


# note that clingo's parser actually allows arbitrary constant as the external_type
# argument of External, but any other value than true or false results in the external
# statement having no effect
ExternalTypeField = refine_field(
    ConstantField, ["true", "false"], name="ExternalTypeField"
)


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
    body = Literal_Tuple1.Field
    external_type = ExternalTypeField


class External1(ComplexTerm):
    "Term identifying a child external predicate."
    id = Identifier_Field(default=lambda: next(id_count))

    class Meta:
        # pylint: disable=too-few-public-methods, missing-class-docstring
        name = "external"


Statement = combine_fields([Rule1.Field, External1.Field])


class Statement_Tuple(Predicate):
    """Predicate representing an element of a tuple of statements.

    id: Identifier of the tuple.
    position: Position of the element the tuple, given by < relation over integers.
    element: Predicate identifying the element.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    position = IntegerField
    element = Statement


class Statement_Tuple1(ComplexTerm):
    "Term identifying a child statement tuple"
    id = Identifier_Field(default=lambda: next(id_count))

    class Meta:
        # pylint: disable=too-few-public-methods, missing-class-docstring
        name = "statement_tuple"


class Constant_Tuple(Predicate):
    """Predicate representing an element of a tuple of constant terms.

    id: Identifier of the tuple.
    position: Position of the element the tuple, given by < relation over integers.
    element: Predicate identifying the element.
    """

    id = Identifier_Field(default=lambda: next(id_count))
    position = IntegerField
    element = Function1.Field


class Constant_Tuple1(ComplexTerm):
    "Term identifying a child constant tuple."
    id = Identifier_Field(default=lambda: next(id_count))

    class Meta:
        # pylint: disable=too-few-public-methods, missing-class-docstring
        name = "constant_tuple"


class Program(Predicate):
    """Predicate representing a subprogram statement.

    name: The name of the subprogram, a string.
    parameters: The parameters of the subprogram, a tuple of constants.
    statements: The tuple of statements comprising the subprogram."""

    name = StringField
    parameters = Constant_Tuple1.Field
    statements = Statement_Tuple1.Field


AstPredicate = Union[
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
    Interval,
    Interval1,
    Guard_Tuple,
    Guard_Tuple1,
    Comparison,
    Comparison1,
    Symbolic_Atom,
    Symbolic_Atom1,
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
    Interval,
    Interval1,
    Guard_Tuple,
    Guard_Tuple1,
    Comparison,
    Comparison1,
    Symbolic_Atom,
    Symbolic_Atom1,
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

AstFact = Union[
    String,
    Number,
    Variable,
    Function,
    Term_Tuple,
    Binary_Operation,
    Interval,
    Guard_Tuple,
    Comparison,
    Symbolic_Atom,
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
    Interval,
    Guard_Tuple,
    Comparison,
    Symbolic_Atom,
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
    """Wrapper predicate to distinguish output AST facts of a transformation."""

    ast = combine_fields([fact.Field for fact in AST_Facts])
