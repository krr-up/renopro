#include "ast.lp".
#include "defined.lp".

final(A) :- ast(A), not ast_operation(delete(A)), not ast_operation(replace(A,_)).
final(R) :- ast_operation(replace(_,R)).
final(N) :- ast_operation(add(N)).

#defined ast_operation/1.

#show final/1.
