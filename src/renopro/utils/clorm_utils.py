"""Clorm related utility functions."""
import enum
import inspect
import re
from contextlib import AbstractContextManager
from types import TracebackType, new_class
from typing import Any, Sequence, Type, TypeVar, cast

from clingo import Symbol
from clorm import BaseField, UnifierNoMatchError
from thefuzz import process  # type: ignore

from renopro import predicates as preds


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

    def _pytocl(py: enum.Enum) -> Any:
        val = py.value
        if val not in values:
            raise ValueError(
                f"'{val}' is not a valid value of enum class '{enum_class.__name__}'"
            )
        return val

    def body(ns: dict[str, Any]) -> None:
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


class ChildQueryError(Exception):
    """Exception raised when a required child fact of an AST fact
    cannot be found.

    """


class ChildrenQueryError(Exception):
    """Exception raised when the expected number child facts of an AST
    fact cannot be found.

    """


class TryUnify(AbstractContextManager):  # type: ignore
    """Context manager to try some operation that requires unification
    of some set of ast facts. Enhance error message if unification fails.
    """

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is UnifierNoMatchError:
            self.handle_unify_error(cast(UnifierNoMatchError, exc_value))

    @staticmethod
    def handle_unify_error(error: UnifierNoMatchError) -> None:
        """Enhance UnifierNoMatchError with some more
        useful error messages to help debug the reason unification failed.

        """
        unmatched = error.symbol
        name2arity2pred = {
            pred.meta.name: {pred.meta.arity: pred} for pred in preds.AstPreds
        }
        candidate = name2arity2pred.get(unmatched.name, {}).get(
            len(unmatched.arguments)
        )
        if candidate is None:
            fuzzy_name = process.extractOne(unmatched.name, name2arity2pred.keys())[0]
            signatures = [
                f"{fuzzy_name}/{arity}." for arity in name2arity2pred[fuzzy_name]
            ]
            msg = f"""No AST fact of matching signature found for symbol
            '{unmatched}'.
            Similar AST fact signatures are:
            """ + "\n".join(
                signatures
            )
            raise UnifierNoMatchError(
                inspect.cleandoc(msg), unmatched, error.predicates
            ) from None  # type: ignore
        for idx, arg in enumerate(unmatched.arguments):
            # This is very hacky. Should ask Dave for a better
            # solution, if there is one.
            arg_field = candidate[idx]._field  # pylint: disable=protected-access
            arg_field_str = re.sub(r"\(.*?\)", "", str(arg_field))
            try:
                arg_field.cltopy(arg)
            except (TypeError, ValueError):
                msg = f"""Cannot unify symbol
                '{unmatched}'
                to only candidate AST fact of matching signature
                {candidate.meta.name}/{candidate.meta.arity}
                due to failure to unify symbol's argument
                '{arg}'
                against the corresponding field
                '{arg_field_str}'."""
                raise UnifierNoMatchError(
                    inspect.cleandoc(msg), unmatched, (candidate,)
                ) from None  # type: ignore
        raise RuntimeError("Code should be unreachable")  # nocoverage
