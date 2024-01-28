#include "ast.lp".
#include "defined.lp".
#include "replace.lp".
#include "decompose.lp".
#include "add-children.lp".

% delete stops inertia from old ast
transformed(A) :- ast(A), not ast_operation(delete(A)).
% and even overrides adding of new ast
transformed(A) :- ast_operation(add(A)), not ast_operation(delete(A)).

#include "transformed_ast_fact2id.lp".
#include "reachable.lp".

#defined ast_operation/1.

#show final/1.
