#include "ast.lp".
#include "defined.lp".
#include "replace.lp".

#script (python)

from clingo.symbol import Function, Number, SymbolType, Symbol, List

from renopro.predicates import pred_names, composed_pred_names


def _decompose(f: Symbol, top_id: Symbol, asts: List[Symbol], i: int, top_level=False, override_id=None):
    current_id = override_id if override_id is not None else i
    fresh_id = Function("_fresh", [top_id, Number(current_id)]) if not top_level else top_id.arguments[0]
    i = i + 1
    current_id = i
    ast_args = [fresh_id] if not top_level else []
    for arg in f.arguments:
        if arg.type is SymbolType.Function and arg.name in composed_pred_names:
            id_term, i = _decompose(arg, top_id, asts, i)
            asts.append(Function("child", [Function(f.name.rstrip("_"), [fresh_id]), id_term]))
            ast_args.append(id_term)
        elif arg.type is SymbolType.Function and arg.name == "":
            j = i
            for element in arg.arguments:
                id_term, i = _decompose(element, top_id, asts, i, override_id=j)
                asts.append(Function("child", [Function(f.name.rstrip("_"), [fresh_id]), id_term]))
            ast_args.append(id_term)
        elif arg.type is SymbolType.Function and arg.name in pred_names and len(arg.arguments) == 1:
            asts.append(Function("child", [Function(f.name.rstrip("_"), [fresh_id]), arg]))
            ast_args.append(arg)
        else:
            ast_args.append(arg)
    asts.append(Function(f.name.rstrip("_"), ast_args))
    return Function(f.name.rstrip("_"), [fresh_id]), i

def decompose(f: Symbol):
    asts = []
    _decompose(f, Function(f.name, f.arguments[:1]), asts, 0, top_level=True)
    return asts

 #end.

% delete stops inertia from old ast
final(A) :- ast(A), not ast_operation(delete(A)).
% and even overrides adding of new ast
final(A) :- ast_operation(add(A)), not ast_operation(delete(A)).

#defined ast_operation/1.

#show final/1.
