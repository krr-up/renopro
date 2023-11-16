# nocoverage
from pathlib import Path

import renopro.predicates as preds


def generate_replace():
    "Generate replacement rules from predicates."
    program = (
        "% replace(A,B) replaces a child predicate identifier A\n"
        "% with child predicate identifier B in each AST fact where A\n"
        "% occurs as a term.\n\n"
    )
    for predicate in preds.AstPreds:
        if predicate is preds.Location:
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
            if field.complex is not None or hasattr(field, "fields"):
                replacements.append(idx)
        for idx in replacements:
            old_args = ",".join(
                ["X" + str(i) if i != idx else "A" for i in range(arity)]
            )
            new_args = ",".join(
                ["X" + str(i) if i != idx else "B" for i in range(arity)]
            )
            program += (
                f"ast_operation(add({name}({new_args}));"
                f"delete({name}({old_args})))\n :- "
                f"ast_operation(replace(A,B)), {name}({old_args}).\n\n"
            )
        Path("src", "renopro", "asp", "replace.lp").write_text(program)


def generate_ast():
    "Generate rules to tag AST facts."
    program = "% Rules to tag AST facts.\n\n"
    for predicate in preds.AstPreds:
        name = predicate.meta.name
        arity = predicate.meta.arity
        args = ",".join(["X" + str(i) for i in range(arity)])
        fact = f"{name}({args})"
        rule = f"ast({fact}) :- {fact}.\n"
        program += rule
    Path("src", "renopro", "asp", "ast.lp").write_text(program)


def generate_defined():
    "Generate defined statements for AST facts."
    program = "% Defined statements for AST facts.\n\n"
    for predicate in preds.AstPreds:
        name = predicate.meta.name
        arity = predicate.meta.arity
        statement = f"#defined {name}/{arity}.\n"
        program += statement
    Path("src", "renopro", "asp", "defined.lp").write_text(program)


def generate_child():
    "Generate rules to define child relation among AST identifiers."
    program = (
        "% child(X,Y) holds if the ast fact with identifier X has a child fact\n"
        '% with identifier Y. Note that identifiers X, Y are "typed";\n'
        "% that is, they are of the form ast_pred(id).\n\n"
    )
    for predicate in preds.AstPreds:
        if predicate is preds.Location:
            continue
        children = []
        name = predicate.meta.name
        arity = predicate.meta.arity
        for idx, key in enumerate(predicate.meta.keys()):
            if key == "id":
                continue
            field = getattr(predicate, key).meta.field
            # we only care about complex terms and combined
            # fields - these are the terms that identify a child
            # predicate
            if field.complex is not None:
                child_term = field.complex
                child_name = child_term.meta.name
                child_arity = child_term.non_unary.meta.arity
                children.append((idx, child_name, child_arity))
            if hasattr(field, "fields"):
                for f in field.fields:
                    child_term = f.complex
                    child_name = child_term.meta.name
                    child_arity = child_term.non_unary.meta.arity
                    children.append((idx, child_name, child_arity))
        ",".join(["_" for i in range(arity - 1)])
        for idx, child_name, child_arity in children:
            parent_idx2arg = {0: "X", idx: "Y"}
            parent_args = [parent_idx2arg.get(i, "_") for i in range(arity)]
            parent_pred = f"{name}({','.join(parent_args)})"
            child_pred = (
                f"{child_name}(Y,{','.join(['_' for _ in range(child_arity - 1)])})"
            )
            child_relation = f"child({name}(X),{child_name}(Y))"
            program += f"{child_relation} :- {parent_pred}, {child_pred}.\n\n"
        Path("src", "renopro", "asp", "child.lp").write_text(program)


if __name__ == "__main__":
    generate_replace()
    generate_ast()
    generate_defined()
    generate_child()
