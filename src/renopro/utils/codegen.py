# nocoverage
from pathlib import Path
from typing import Any, List
import json

import renopro.predicates as preds


pred_name2child_idx: dict[str, List[int]] = {}


def build_pred_name2child_idx() -> None:
    for predicate in preds.AstPreds:
        idx_list: List[int] = []
        for idx, key in enumerate(predicate.meta.keys()):
            if key == "id":
                continue
            field = getattr(predicate, key).meta.field
            # these are the fields that contain a child identifier
            if field.complex is not None or (
                hasattr(field, "fields") and field is not preds.IntegerOrRawField
            ):
                idx_list.append(idx)
        pred_name2child_idx[predicate.meta.name] = idx_list


def generate_replace() -> None:
    "Generate replacement rules from predicates."
    program = (
        "% replace(A,B) replaces a child predicate identifier A\n"
        "% with child predicate identifier B in each AST fact where A\n"
        "% occurs as a term.\n\n#program always.\n"
    )
    for predicate in preds.AstPreds:
        if predicate in [preds.Location, preds.Child]:
            continue
        name = predicate.meta.name
        arity = predicate.meta.arity
        for idx in pred_name2child_idx[predicate.meta.name]:
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
    add_child_program = (
        "% Add child relations for facts added via ast add.\n\n#program always.\n"
    )
    for predicate in preds.AstPreds:
        if predicate is preds.Location or predicate is preds.Child:
            continue
        name = predicate.meta.name
        arity = predicate.meta.arity
        for idx in pred_name2child_idx[predicate.meta.name]:
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


def generate_wrap_ast() -> None:
    "Generate rules to tag AST facts."
    program = "% Rules to tag AST facts.\n\n#program always.\n"
    for predicate in preds.AstPreds:
        name = predicate.meta.name
        arity = predicate.meta.arity
        args = ",".join(["X" + str(i) for i in range(arity)])
        fact = f"{name}({args})"
        rule = f"ast(fact({fact})) :- {fact}, not final.\n"
        program += rule
    Path("src", "renopro", "asp", "wrap_ast.lp").write_text(program, encoding="utf-8")


def generate_unwrap_ast()  -> None:
    program = "% Rules to unwrap tagged AST facts for next step of transformation.\n\n#program always.\n"
    for predicate in preds.AstPreds:
        name = predicate.meta.name
        arity = predicate.meta.arity
        args = ",".join(["X" + str(i) for i in range(arity)])
        fact = f"{name}({args})"
        rule = f"{fact} :- 'transformed({fact}).\n"
        program += rule
    Path("src", "renopro", "asp", "unwrap_ast.lp").write_text(program, encoding="utf-8")


def generate_defined() -> None:
    "Generate defined statements for AST facts."
    program = "% Defined statements for AST facts.\n\n"
    for predicate in preds.AstPreds:
        name = predicate.meta.name
        arity = predicate.meta.arity
        statement = f"#defined {name}/{arity}.\n"
        program += statement
    Path("src", "renopro", "asp", "defined.lp").write_text(program, encoding="utf-8")


def dump_json() -> None:
    """Serialize data used by embedded python scripts in clingo to json.

    This allows us to avoid having to re-run the modules within the
    embedded python interpreter, saving us some precious start-up time
    when doing e.g. transformation.
    """
    data_path = Path("src", "renopro", "data")
    (data_path / "pred_names.json").write_text(
        json.dumps(preds.pred_names), encoding="utf-8"
    )
    (data_path / "composed_pred_names.json").write_text(
        json.dumps(preds.composed_pred_names), encoding="utf-8"
    )
    (data_path / "pred_name2child_idx.json").write_text(
        json.dumps(pred_name2child_idx), encoding="utf-8"
    )


if __name__ == "__main__":
    build_pred_name2child_idx()
    generate_replace()
    generate_add_child()
    generate_wrap_ast()
    generate_unwrap_ast()
    generate_defined()
    dump_json()
