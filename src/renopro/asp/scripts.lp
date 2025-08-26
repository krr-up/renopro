#script (python)

import json
from pathlib import Path

from clingo.symbol import Function, Number, SymbolType, Symbol, String, List

#data_dir = Path("src", "renopro", "data")
#pred_names = json.loads((data_dir / "pred_names.json").read_text("utf-8"))
#composed_pred_names = json.loads((data_dir / "composed_pred_names.json").read_text("utf-8"))
#pred_name2child_idx = json.loads(
#    Path("src", "renopro", "data", "pred_name2child_idx.json").read_text("utf-8")
#)

pred_names = ["string", "number", "variable", "unary_operation", "binary_operation", "interval", "terms", "function", "external_function", "pool", "theory_terms", "theory_sequence", "theory_function", "theory_operators", "theory_unparsed_term_elements", "theory_unparsed_term", "guard", "guards", "comparison", "boolean_constant", "symbolic_atom", "literal", "literals", "conditional_literal", "aggregate_elements", "aggregate", "theory_atom_elements", "theory_guard", "theory_atom", "body_aggregate_elements", "body_aggregate", "body_literal", "body_literals", "head_aggregate_elements", "head_aggregate", "conditional_literals", "disjunction", "rule", "definition", "show_signature", "defined", "show_term", "minimize", "script", "statements", "constants", "program", "external", "edge", "heuristic", "project_atom", "project_signature", "theory_operator_definitions", "theory_term_definitions", "theory_guard_definition", "theory_atom_definitions", "theory_definition", "comment", "location", "child"]

composed_pred_names = ["string_", "number_", "variable_", "unary_operation_", "binary_operation_", "interval_", "terms_", "function_", "external_function_", "pool_", "theory_terms_", "theory_sequence_", "theory_function_", "theory_operators_", "theory_unparsed_term_elements_", "theory_unparsed_term_", "guard_", "guards_", "comparison_", "boolean_constant_", "symbolic_atom_", "literal_", "literals_", "conditional_literal_", "aggregate_elements_", "aggregate_", "theory_atom_elements_", "theory_guard_", "theory_atom_", "body_aggregate_elements_", "body_aggregate_", "body_literal_", "body_literals_", "head_aggregate_elements_", "head_aggregate_", "conditional_literals_", "disjunction_", "rule_", "definition_", "show_signature_", "defined_", "show_term_", "minimize_", "script_", "statements_", "constants_", "program_", "external_", "edge_", "heuristic_", "project_atom_", "project_signature_", "theory_operator_definitions_", "theory_term_definitions_", "theory_guard_definition_", "theory_atom_definitions_", "theory_definition_", "comment_", "location_", "child_"]

pred_name2child_idx = {"string": [], "number": [], "variable": [], "unary_operation": [2], "binary_operation": [2, 3], "interval": [1, 2], "terms": [2], "function": [2], "external_function": [2], "pool": [1], "theory_terms": [2], "theory_sequence": [2], "theory_function": [2], "theory_operators": [], "theory_unparsed_term_elements": [2, 3], "theory_unparsed_term": [1], "guard": [2], "guards": [2], "comparison": [1, 2], "boolean_constant": [], "symbolic_atom": [1], "literal": [2], "literals": [2], "conditional_literal": [1, 2], "aggregate_elements": [2], "aggregate": [1, 2, 3], "theory_atom_elements": [2, 3], "theory_guard": [2], "theory_atom": [1, 2, 3], "body_aggregate_elements": [2, 3], "body_aggregate": [1, 3, 4], "body_literal": [2], "body_literals": [2], "head_aggregate_elements": [2, 3], "head_aggregate": [1, 3, 4], "conditional_literals": [2], "disjunction": [1], "rule": [1, 2], "definition": [2], "show_signature": [], "defined": [], "show_term": [1, 2], "minimize": [1, 2, 3, 4], "script": [], "statements": [2], "constants": [2], "program": [2, 3], "external": [1, 2], "edge": [1, 2, 3], "heuristic": [1, 2, 3, 4, 5], "project_atom": [1, 2], "project_signature": [], "theory_operator_definitions": [], "theory_term_definitions": [3], "theory_guard_definition": [1], "theory_atom_definitions": [6], "theory_definition": [2, 3], "comment": [], "location": [0, 1, 2], "child": [0, 1]}


def _decompose(f: Symbol, top_id: Symbol, asts: List[Symbol], i: int, top_level=False, override_id=None):
    current_id = override_id if override_id is not None else i
    fresh_id = Function("_fresh", [top_id, Number(current_id)]) if not top_level else top_id.arguments[0]
    i = i + 1
    current_id = i
    ast_args = [fresh_id] if not top_level else []
    for arg in f.arguments:
        if arg.type is SymbolType.Function and arg.name in composed_pred_names:
            id_term, i = _decompose(arg, top_id, asts, i)
            ast_args.append(id_term)
        elif arg.type is SymbolType.Function and arg.name == "":
            j = i
            for element in arg.arguments:
                id_term, i = _decompose(element, top_id, asts, i, override_id=j)
            ast_args.append(id_term)
        elif arg.type is SymbolType.Function and arg.name in pred_names and len(arg.arguments) == 1:
            ast_args.append(arg)
        else:
            ast_args.append(arg)
    asts.append(Function(f.name.rstrip("_"), ast_args))
    return Function(f.name.rstrip("_"), [fresh_id]), i

def decompose(f: Symbol):
    asts = []
    _decompose(f, Function(f.name, f.arguments[:1]), asts, 0, top_level=True)
    return asts

def ast_fact2id(fact: Symbol) -> Symbol:
    return Function(fact.name, fact.arguments[:1])

def ast_fact2children_ids(fact):
    try:
        return [fact.arguments[idx] for idx in pred_name2child_idx[fact.name]]
    except KeyError:
        return String("invalid_signature")

def loc2str(location: Symbol) -> Symbol:
    pairs = []
    for pair in zip(location.arguments[1].arguments, location.arguments[2].arguments):
        pairs.append(
            str(pair[0]).strip('"') if pair[0] == pair[1] else str(pair[0]).strip('"') + "-" + str(pair[1]).strip('"')
        )
    return String(":".join(pairs).strip('"'))

#end.
