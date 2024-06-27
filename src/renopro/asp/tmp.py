
import json
from pathlib import Path

from clingo.symbol import Function, Number, SymbolType, Symbol, String, List

data_dir = Path("src", "renopro", "data")
pred_names = json.loads((data_dir / "pred_names.json").read_text("utf-8"))
composed_pred_names = json.loads((data_dir / "composed_pred_names.json").read_text("utf-8"))
pred_name2child_idx = json.loads(
    Path("src", "renopro", "data", "pred_name2child_idx.json").read_text("utf-8")
)


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

def ast_fact2children(fact):
    try:
        return [fact.arguments[idx] for idx in pred_name2child_idx[fact.name]]
    except KeyError:
        return String("invalid_signature")

def location2str(location: Symbol) -> Symbol:
    pairs = []
    for pair in zip(location.arguments[1].arguments, location.arguments[2].arguments):
        pairs.append(
            str(pair[0]) if pair[0] == pair[1] else str(pair[0]) + "-" + str(pair[1])
        )
    return String(":".join(pairs))
