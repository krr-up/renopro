# nocoverage
from pathlib import Path
from typing import Any

import renopro.predicates as preds


def is_child_field(field: Any) -> bool:
    "Check if field identifies a child predicate."
    # Only complex or combined fields which are not IntegerOrRawField
    # identify child predicates
    return field.complex is not None or (
        hasattr(field, "fields") and not field is preds.IntegerOrRawField
    )


def generate_replace() -> None:
    "Generate replacement rules from predicates."
    program = (
        "% replace(A,B) replaces a child predicate identifier A\n"
        "% with child predicate identifier B in each AST fact where A\n"
        "% occurs as a term.\n\n"
    )
    for predicate in preds.AstPreds:
        if predicate is preds.Location or predicate is preds.Child:
            continue
        replacements = []
        name = predicate.meta.name
        arity = predicate.meta.arity
        for idx, key in enumerate(predicate.meta.keys()):
            if key == "id":
                continue
            field = getattr(predicate, key).meta.field
            # we only care about replacing and combined
            # fields - these are the terms that identify a child
            # predicate
            if is_child_field(field):
                replacements.append(idx)
        for idx in replacements:
            old_args = ",".join(
                ["X" + str(i) if i != idx else "A" for i in range(arity)]
            )
            new_args = ",".join(
                ["X" + str(i) if i != idx else "B" for i in range(arity)]
            )
            program += (
                f"ast(add({name}({new_args}));"
                f"delete({name}({old_args})))\n :- "
                f"ast(_replace_id(A,B)), ast(fact({name}({old_args}));add({name}({old_args}))).\n\n"
            )
    program += "ast(add(child(X0,B));delete(child(X0,A)))\n  :- ast(_replace_id(A,B)), ast(fact(child(X0,A));add(child(X0,A)))."
    Path("src", "renopro", "asp", "replace_id.lp").write_text(program, encoding="utf-8")


def generate_add_child() -> None:
    """Generate rules to create child relations for all facts added
    via ast add, and rules to replace identifiers via
    ast replace."""
    add_child_program = "% Add child relations for facts added via ast add.\n\n"
    for predicate in preds.AstPreds:
        if predicate is preds.Location or predicate is preds.Child:
            continue
        child_arg_indices = []
        name = predicate.meta.name
        arity = predicate.meta.arity
        for idx, key in enumerate(predicate.meta.keys()):
            if key == "id":
                continue
            field = getattr(predicate, key).meta.field
            if is_child_field(field):
                child_arg_indices.append(idx)
        for idx in child_arg_indices:
            add_child_args = ",".join(
                ["X" + str(i) if i != idx else "Child" for i in range(arity)]
            )
            add_child_program += (
                f"ast(add(child({name}(X0),Child)))\n  :- "
                f"ast(add({name}({add_child_args}))).\n\n"
            )
        Path("src", "renopro", "asp", "add-children.lp").write_text(
            add_child_program, encoding="utf-8"
        )


def generate_ast_fact2id() -> None:
    "Generate rules mapping AST facts to their identifiers"
    program = "% map ast facts to their identifiers.\n\n"
    for predicate in preds.AstPreds:
        if predicate is preds.Location:
            location = "location(Id,Begin,End)"
            program += f"ast_fact2id({location},Id) :- ast(fact({location});add({location});delete({location})).\n"
            continue
        if predicate is preds.Child:
            continue
        name = predicate.meta.name
        arity = predicate.meta.arity
        args = ",".join(["X" + str(i) for i in range(arity)])
        fact = f"{name}({args})"
        identifier = f"{name}(X0)"
        program += f"ast_fact2id({fact},{identifier})\n  :- ast(fact({fact});add({fact});delete({fact})).\n\n"
    path = Path("src", "renopro", "asp", "ast_fact2id.lp")
    path.write_text(program, encoding="utf-8")


def generate_ast() -> None:
    "Generate rules to tag AST facts."
    program = "% Rules to tag AST facts.\n\n"
    for predicate in preds.AstPreds:
        name = predicate.meta.name
        arity = predicate.meta.arity
        args = ",".join(["X" + str(i) for i in range(arity)])
        fact = f"{name}({args})"
        rule = f"ast(fact({fact})) :- {fact}.\n"
        program += rule
    Path("src", "renopro", "asp", "ast.lp").write_text(program, encoding="utf-8")


def generate_defined() -> None:
    "Generate defined statements for AST facts."
    program = "% Defined statements for AST facts.\n\n"
    for predicate in preds.AstPreds:
        name = predicate.meta.name
        arity = predicate.meta.arity
        statement = f"#defined {name}/{arity}.\n"
        program += statement
    Path("src", "renopro", "asp", "defined.lp").write_text(program, encoding="utf-8")


if __name__ == "__main__":
    generate_replace()
    generate_add_child()
    generate_ast()
    generate_defined()
    generate_ast_fact2id()
