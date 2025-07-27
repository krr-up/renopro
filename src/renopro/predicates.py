# pylint: disable=too-many-lines,abstract-method
"""Definitions of AST elements as clorm predicates."""
import inspect
import re
from itertools import count
from types import new_class
from typing import Any, Sequence, Type, Union, TypeAlias

from clingo import ast, Symbol
from clorm import (
    BaseField,
    ComplexTerm,
    ConstantField,
    ConstantStr,
    IntegerField,
    Predicate,
    StringField,
    refine_field,
    define_nested_list_field,
    HeadList,
    field,
)

from renopro.enum_fields import (
    define_enum_field,
    AggregateFunctionField,
    BinaryOperatorField,
    CommentTypeField,
    ComparisonOperatorField,
    SignField,
    TheoryAtomTypeField,
    TheoryOperatorTypeField,
    TheorySequenceTypeField,
    UnaryOperatorField,
    AggregateFunction,
    BinaryOperator,
    CommentType,
    ComparisonOperator,
    Sign,
    TheoryAtomType,
    TheoryOperatorType,
    TheorySequenceType,
    UnaryOperator,
    Bool,
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


Term: TypeAlias = (
    "String | Number | Variable| Unary_Operation | Binary_Operation | Interval | Function | External_Function | Pool"
)


class String(ComplexTerm):
    """ComplexTerm representing a string term.

    string: Value of string term, a string term itself.
    """

    string: str


class Number(ComplexTerm):
    """ComplexTerm representing an integer term.

    number: Value of integer term, an integer term itself.
    """

    number: int


class Variable(ComplexTerm):
    """ComplexTerm representing a variable term.

    value: Value of variable term, a string term.
    """

    name: str


# We need to use this lazily combined field for terms, as clorm does not
# currently support circular forward references in type annotations. In
# the future (python 3.14) it would be nice to add support using PEP 649 749

TermField: Term = combine_fields(
    [String.Field, Number.Field, Variable.Field], name="TermField"
)


class Unary_Operation(ComplexTerm):
    """ComplexTerm representing a unary operation term.

    operator: A clingo unary operator, in string form.
    argument: The term argument the unary operator is applied to.
    """

    operator_type: UnaryOperator
    argument: Term = TermField


class Binary_Operation(ComplexTerm):
    """ComplexTerm representing a binary operation term.

    operator: A clingo binary operator, in string form.
    left: ComplexTerm identifying the term that is the left operand of the operation.
    right: ComplexTerm identifying the term that is the right operand of the operation.
    """

    operator_type: BinaryOperator
    left: Term = TermField
    right: Term = TermField


class Interval(ComplexTerm):
    """ComplexTerm representing an interval term.

    left: Left bound of the interval.
    right: Right bound of the interval.
    """

    left: Term = TermField
    right: Term = TermField


TermList = HeadList[Term]

TermListField: TermList = define_nested_list_field(TermField, name="TermListField")


class Function(ComplexTerm):
    """ComplexTerm representing a function symbol with term arguments.
    Note that we represent function terms and constant terms as well as symbolic
    atoms and propositional constants via this predicate.

    name: Symbolic name of the function, a constant term.
    arguments: Term  identifying the function's arguments.
               If there are no elements of the term tuple with a matching
               identifier, the function has no arguments and is thus a constant.

    """

    arity: int
    name: str
    arguments: TermList = TermListField


class External_Function(ComplexTerm):
    """ComplexTerm representing an external function written in a
    scripting language to be evaluated during grounding.

    name: Symbolic name of the function, a constant term.
    arguments: Term  identifying the function's arguments.
    """

    arity: int
    name: str
    arguments: TermList = TermListField


class Pool(ComplexTerm):
    """ComplexTerm representing a pool of terms.

    arguments: Terms forming the pool.
    """

    arguments: TermList = TermListField


TermField.fields.extend(
    [
        Unary_Operation.Field,
        Binary_Operation.Field,
        Interval.Field,
        Function.Field,
        External_Function.Field,
        Pool.Field,
    ]
)

TheoryTerm: TypeAlias = (
    "String | Number | Function | Variable | Theory_Sequence | Theory_Function | Theory_Unparsed_Term"
)

TheoryTermField: TheoryTerm = combine_fields(
    [String.Field, Number.Field, Function.Field, Variable.Field], name="TheoryTermField"
)

TheoryTermList = HeadList[TheoryTerm]

TheoryTermListField: TheoryTermList = define_nested_list_field(
    TheoryTermField, name="TheoryTermListField"
)


class Theory_Sequence(ComplexTerm):
    """ComplexTerm representing a sequence of theory terms.

    length: The length of the theory sequence.
    sequence_type: The type of the theory sequence.
    terms: The list of terms forming the sequence.
    """

    length: int
    sequence_type: TheorySequenceType
    terms: TheoryTermList = TheoryTermListField


class Theory_Function(ComplexTerm):
    """ComplexTerm representing a theory function term.

    arity: The arity of the theory function.
    name: The name of the theory function.
    terms: The list of theory terms forming the arguments of
           the theory function.
    """

    arity: int
    name: str
    arguments: TheoryTermList = TheoryTermListField


class Theory_Operator(ComplexTerm):
    """Complex Term representing a theory operator."""

    operator: str


TheoryOperatorList = HeadList[Theory_Operator]


class Theory_Unparsed_Term_Element(ComplexTerm):
    """ComplexTerm representing an element of a list of unparsed
    theory term elements. Each element consists of a list of theory
    operators and a theory term.

    length: The length of the theory operators.
    operators: A list of theory operators.
    term: The theory term that (one of the) operand of the operator(s).

    """

    length: int
    operators: TheoryOperatorList
    term: TheoryTerm = TheoryTermField


class Theory_Unparsed_Term(ComplexTerm):
    """ComplexTerm representing an unparsed theory term.  An unparsed
    theory term consists of a list of unparsed theory term elements.

    elements: The list of aforementioned elements
              forming the unparsed theory term.

    """

    elements: HeadList[Theory_Unparsed_Term_Element]


TheoryTermField.fields.extend(
    [Theory_Sequence.Field, Theory_Function.Field, Theory_Unparsed_Term.Field]
)


# Literals


class Guard(ComplexTerm):
    """ComplexTerm representing a guard for a comparison or aggregate atom."""

    comparison: ComparisonOperator
    term: Term = TermField


class Comparison(ComplexTerm):
    """ComplexTerm representing a comparison atom.

    term: Leftmost term of the comparison atom.
    guards: list of guard predicates
    """

    term: Term = TermField
    guards: HeadList[Guard]


class Boolean_Constant(ComplexTerm):
    """ComplexTerm representing a boolean constant, true or false.

    value: Boolean value of the constant, represented by the constant
    terms 'true' and 'false'.
    """

    value: Bool


# For some reason, item(fox;beans) parses differently than
# item((fox;beans)). The latter gives a symbolic atom with a function
# argument with name item and the pool as the argument. The former
# gives a symbolic atom with a pool argument, the arguments of which
# are item(fox) and item(beans). Therefore, a symbolic atom can also
# have a pool argument.


SymbolicAtomSymbol = Function | Pool | Unary_Operation


class Symbolic_Atom(ComplexTerm):
    """ComplexTerm representing a symbolic atom.

    symbol: The symbol constituting the atom.
    """

    symbol: SymbolicAtomSymbol


Atom = Symbolic_Atom | Comparison | Boolean_Constant


class Literal(ComplexTerm):
    """ComplexTerm representing a literal.

    sign_: Sign of the literal, in string form. Possible values are
         "pos" "not" and "not not".
    atom: The atom constituting the literal.
    """

    sign_: Sign
    atom: Atom


LiteralList = HeadList[Literal]


class Conditional_Literal(ComplexTerm):
    """ComplexTerm representing a conditional literal.

    literal: the literal in the head of the conditional literal
    condition: the list of literals forming the condition
    """

    literal: Literal
    condition: LiteralList


ConditionalLiteralList = HeadList[Conditional_Literal]


class Nil(ComplexTerm, is_tuple=True):
    """Complex term representing lack of value.
    In Lisp style, we represent this as an empty tuple."""

    pass


GuardOrNil = Guard | Nil


class Aggregate(ComplexTerm):
    """ComplexTerm representing an simple (count) aggregate atom with
    conditional literal elements.

    left_guard: The (optional) left guard of the aggregate.
    elements: The list of conditional literals forming the elements of
              the aggregate.
    right_guard: The (optional) right guard of the aggregate.
    """

    left_guard: GuardOrNil
    elements: ConditionalLiteralList
    right_guard: GuardOrNil


class Theory_Atom_Element(ComplexTerm):
    """ComplexTerm representing an element of a list forming a theory atom's
    aggregate-like component.

    terms: The list of theory terms which form an element to be conditionally
           aggregated in the theory condition.
    atom: The list of literals forming a conjunction on which the
               list of theory terms is conditioned.
    """

    terms: TheoryTermList = TheoryTermListField
    condition: LiteralList


class Theory_Guard(ComplexTerm):
    """ComplexTerm representing a theory guard.

    operator_name: The name of the binary theory operator applied to
                   the aggregated elements of the theory atom and the theory term.
    term: The theory term to which the theory operator is applied.
    """

    operator_name: str
    term: TheoryTerm = TheoryTermField


TheoryAtomElementList = HeadList[Theory_Atom_Element]

TheoryGuardOrNil = Theory_Guard | Nil


class Theory_Atom(ComplexTerm):
    """ComplexTerm representing a theory atom.

    term: The forming the symbolic part of the theory atom.
    elements: The list of elements aggregated in the theory atom.
    guard: The (optional) theory guard applied the aggregated elements
           of the theory atom.
    """

    term: Function
    elements: TheoryAtomElementList
    guard: Theory_Guard


class Body_Aggregate_Element(ComplexTerm):
    """ComplexTerm representing an element of a body aggregate.

    terms: The list of terms to which the aggregate function will be applied.
    condition: The list of literals forming a conjunction on which the
               list of terms is conditioned.
    """

    terms: TermList = TermListField
    condition: LiteralList


class Body_Aggregate(ComplexTerm):
    """ComplexTerm representing an aggregate atom occurring in a body.

    left_guard: The (optional) left guard of the aggregate..
    function: The aggregate function applied to the terms of the aggregate
              remaining after evaluation of the conditions.
    elements: The elements of the aggregate,
    right_guard: The (optional) right guard of the aggregate.
    """

    left_guard: GuardOrNil
    function: AggregateFunction
    elements: HeadList[Body_Aggregate_Element]
    right_guard: GuardOrNil


BodyAtom = Atom | Aggregate | Body_Aggregate | Theory_Atom


class Body_Literal(ComplexTerm):
    """ComplexTerm representing a literal occurring in the body of a
    rule.

    sig: Sign of the literal, in string form. Possible values are
         "pos" "not" and "not not".
    atom: The atom constituting the literal.

    """

    sign_: Sign
    atom: BodyAtom


BodyLiteralOrConditionalLiteral = Body_Literal | Conditional_Literal


class Head_Aggregate_Element(ComplexTerm):
    """ComplexTerm representing an element of a head aggregate.

    terms: The list of terms to which the aggregate function will be applied.
    condition: The conditional literal who's condition determines if
               the list of terms are added to the aggregate's set. When the condition
               holds, the head of the conditional literal is also derived.
    """

    terms: TermList = TermListField
    condition: Conditional_Literal


class Head_Aggregate(ComplexTerm):
    """ComplexTerm representing an aggregate atom occuring in a head.

    left_guard: The (optional) left guard of the aggregate.
    function: The aggregate function applied to the term elements of the aggregate
              remaining after evaluation of their conditions.
    elements: The elements of the aggregate,
    right_guard: The (optional) right guard of the aggregate.
    """

    left_guard: GuardOrNil
    function: AggregateFunction
    elements: HeadList[Head_Aggregate_Element]
    right_guard: GuardOrNil


LiteralOrConditionalLiteral = Literal | Conditional_Literal


class Disjunction(ComplexTerm):
    """ComplexTerm representing a disjunction of (conditional) literals.

    elements: The elements of the disjunction, a list of conditional literals
    or regular literals.
    """

    elements: LiteralOrConditionalLiteral


Head = Literal | Aggregate | Head_Aggregate | Disjunction | Theory_Atom

Body = HeadList[BodyLiteralOrConditionalLiteral]


# Statements


class Rule(ComplexTerm):
    """ComplexTerm representing a rule statement.

    head: The head of the rule.
    body: The body of the rule, a list of body literals.
    """

    head: Head
    body: Body


class Definition(ComplexTerm):
    """ComplexTerm representing a definition statement (defining a constant).

    name: Name of the constant defined.
    value: Default value of the constant.
    is_default: true if the statement gives the default value of the
                constant, false if it's overriding it."""

    name: ConstantStr
    # the term in reality must not contain variables, pools or
    # intervals otherwise the parser will complain (the constructor
    # function does not care). Should we encode this in the clorm repr?
    value: Term = TermField
    is_default: Bool


class Show_Signature(ComplexTerm):
    """ComplexTerm representing a show statement given by a predicate signature.

    name: Name of the predicate.
    arity: Arity of the predicate.
    positive: true if predicate is positive, false if strongly negated."""

    name: str
    arity: int
    positive: Bool


class Defined(ComplexTerm):
    """ComplexTerm representing a defined statement.

    name: Name of the predicate to be marked as defined.
    arity: Arity of the predicate.
    positive: true if predicate is positive, false if strongly negated."""

    name: str
    arity: int
    positive: Bool


class Show_Term(ComplexTerm):
    """ComplexTerm representing a show term statement.

    term: The term to be shown.
    body: The body literals the term is conditioned on."""

    term: Term = TermField
    body: Body


class Minimize(ComplexTerm):
    """ComplexTerm representing a minimize statement.

    weight: The weight associated with the list of terms.
    priority: The priority assigned to the weight.
    terms: The list of terms, which contribute their associated weight
           to the minimization. Similar to aggregates, a tuple (across all
           minimize statements) can only contribute a weight once (as in a set).
    body: The body literals on which the list of terms are conditioned."""

    weight: Term = TermField
    priority: Term = TermField
    terms: TermList = TermListField
    body: Body


class Script(ComplexTerm):
    """ComplexTerm representing a script statement.

    name: Name of the embedded script.
    code: The code of the script."""

    name: str
    code: str


# note that clingo's parser actually allows arbitrary constant as the external_type
# argument of External, but any other value than true or false results in the external
# statement having no effect


class External(ComplexTerm):
    """ComplexTerm representing an external statement.

    atom: The external atom.
    body: The list of body literals the external statement is conditioned on.
    external_type: The default value of the external statement.
                   May be the constant 'true' or 'false'.
    """

    atom: Symbolic_Atom
    body: Body
    external_type: Bool


class Edge(ComplexTerm):
    """ComplexTerm representing an edge statement. Answer sets where the
    directed edges obtained from edge statements form a cyclic graph
    are discarded.

    u: The term forming first element of the directed edge.
    v: The term forming the second element of the directed edge.
    body: The body on which the directed edge is conditioned."""

    node_u: Term = TermField
    node_v: Term = TermField
    body: Body


class Heuristic(ComplexTerm):
    """ComplexTerm representing a heuristic statement.

    atom: The symbolic atom to which the heuristic should be applied when
          making a decision on it's truth value.
    body: The list of body atoms on which the application of the heuristic
          is conditioned.
    bias: The bias we associate with the atom.
    priority: The priority of the heuristic (higher overrides lower).
    modifier: The heuristic modifier to be applied."""

    atom: Symbolic_Atom
    body: Body
    bias: Term = TermField
    priority: Term = TermField
    modifier: Term = TermField


class Project_Atom(ComplexTerm):
    """ComplexTerm representing a project atom statement.

    atom: The atom to be projected onto.
    body: The list of body literals on which the projection is conditioned."""

    atom: Symbolic_Atom
    body: Body


class Project_Signature(ComplexTerm):
    """ComplexTerm representing a project signature statement.

    name: Name of the predicate to be projected.
    arity: Arity of the predicate to be projected.
    positive: True if predicate is positive, false if strongly negated."""

    name: str
    arity: int
    positive: Bool


class Theory_Operator_Definition(ComplexTerm):
    """ComplexTerm representing a theory operator definition.

    name: The operator to be defined, in string form.
    priority: The precedence of the operator, determining implicit parentheses.
    operator type: The type of the operator."""

    name: str
    priority: int
    operator_type: TheoryOperatorType


class Theory_Term_Definition(ComplexTerm):
    """ComplexTerm representing a theory term definition.

    name: Name of the theory term to be defined.
    operators: The theory operators defined over the theory term."""

    name: str
    operators: HeadList[Theory_Operator_Definition]


class Theory_Guard_Definition(ComplexTerm):
    """ComplexTerm representing a theory guard definition.

    operators: The possible operators usable to form the guard.
    term: The theory term usable to form the guard."""

    operators: TheoryOperatorList
    term: str


class Theory_Atom_Definition(ComplexTerm):
    """ComplexTerm representing a theory atom definition.

    atom_type: The type of the theory atom, determining the possible places it
               may occur in a logic program.
    name: The name of the theory atom to be defined
    arity: The arity of the theory atom to be defined
    term: The theory term type that may appear as an element of the theory atom.
    guard: The definition of the theory guard which may serve as the theory
           atom's guard.
    """

    atom_type: TheoryAtomType
    name: str
    arity: int
    term: str
    guard: Theory_Guard_Definition | Nil


class Theory_Definition(ComplexTerm):
    """ComplexTerm representing a theory definition statement.

    name: Name of the theory to be defined.
    terms: A list of theory term definitions.
    atoms: A list of theory atom definitions."""

    name: ConstantStr
    terms: HeadList[Theory_Term_Definition]
    atoms: HeadList[Theory_Atom_Definition]


class Comment(ComplexTerm):
    """ComplexTerm representing a comment statement.

    value: The string value the comment comprises of.
    comment_type: The type of the comment, "block" or "line"."""

    value: str
    comment_type: CommentType


Statement = Union[
    Rule,
    Definition,
    Show_Signature,
    Defined,
    Show_Term,
    Minimize,
    Script,
    External,
    Edge,
    Heuristic,
    Project_Atom,
    Project_Signature,
    Theory_Definition,
    Comment,
]

Statements = (
    Rule,
    Definition,
    Show_Signature,
    Defined,
    Show_Term,
    Minimize,
    Script,
    External,
    Edge,
    Heuristic,
    Project_Atom,
    Project_Signature,
    Theory_Definition,
    Comment,
)


class Program(ComplexTerm):
    """ComplexTerm representing a subprogram statement.

    name: The name of the subprogram, a string.
    parameters: The parameters of the subprogram, a list of constants.
    statements: The list of statements comprising the subprogram.
    """

    name: str
    parameters: HeadList[Function]
    statements: HeadList[Statement]


ASTs = (
    String,
    Number,
    Variable,
    Unary_Operation,
    Binary_Operation,
    Interval,
    Function,
    External_Function,
    Pool,
    Theory_Sequence,
    Theory_Function,
    Theory_Operator,
    Theory_Unparsed_Term_Element,
    Theory_Unparsed_Term,
    Guard,
    Comparison,
    Boolean_Constant,
    Symbolic_Atom,
    Literal,
    Conditional_Literal,
    Aggregate,
    Theory_Atom_Element,
    Theory_Guard,
    Theory_Atom,
    Body_Aggregate_Element,
    Body_Aggregate,
    Body_Literal,
    Head_Aggregate_Element,
    Head_Aggregate,
    Disjunction,
    Rule,
    Definition,
    Show_Signature,
    Defined,
    Show_Term,
    Minimize,
    Script,
    External,
    Edge,
    Heuristic,
    Project_Atom,
    Project_Signature,
    Theory_Operator_Definition,
    Theory_Guard_Definition,
    Theory_Atom_Definition,
    Theory_Definition,
    Comment,
    Program,
)

AST = Union[
    String,
    Number,
    Variable,
    Unary_Operation,
    Binary_Operation,
    Interval,
    Function,
    External_Function,
    Pool,
    Theory_Sequence,
    Theory_Function,
    Theory_Operator,
    Theory_Unparsed_Term_Element,
    Theory_Unparsed_Term,
    Guard,
    Comparison,
    Boolean_Constant,
    Symbolic_Atom,
    Literal,
    Conditional_Literal,
    Aggregate,
    Theory_Atom_Element,
    Theory_Guard,
    Theory_Atom,
    Body_Aggregate_Element,
    Body_Aggregate,
    Body_Literal,
    Head_Aggregate_Element,
    Head_Aggregate,
    Disjunction,
    Rule,
    Definition,
    Show_Signature,
    Defined,
    Show_Term,
    Minimize,
    Script,
    External,
    Edge,
    Heuristic,
    Project_Atom,
    Project_Signature,
    Theory_Operator_Definition,
    Theory_Guard_Definition,
    Theory_Atom_Definition,
    Theory_Definition,
    Comment,
    Program,
]


class Position(ComplexTerm):
    """Complex field representing a position in a text file."""

    filename: str
    line: int
    column: int


class Location(Predicate):
    """Predicate linking an AST to the range in a text file / stdin
    from where it was reified.

    """

    ast: AST
    begin: Position
    end: Position


class Node(Predicate):
    """Predicate containing an AST term."""

    ast: AST


class Child(Predicate):
    """Predicate linking a parent AST term and it's location to
    (one of) it's child AST terms."""

    parent: Location
    child: Location


ASTFacts = [Node, Child, Location]

ASTFact = Node | Child | Location


class Transformed(Predicate):
    """Wrapper predicate to distinguish output AST facts of a transformation."""

    ast: ASTFact


name2arity2pred = {pred.meta.name: {pred.meta.arity: pred} for pred in ASTs}

camel2snake = re.compile(
    r"""
        (?<=[a-z])      # preceded by lowercase
        (?=[A-Z])       # followed by uppercase
        |               #   OR
        (?<=[A-Z])       # preceded by lowercase
        (?=[A-Z][a-z])  # followed by uppercase, then lowercase
    """,
    re.X,
)
snake2camel = re.compile("_")

clingo2clorm_name = {v.name: camel2snake.sub("_", v.name) for v in ast.ASTType}
clorm2clingo_name = {p.__name__: snake2camel.sub("", p.__name__) for p in ASTs}
