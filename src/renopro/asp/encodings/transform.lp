#include "ast.lp".

final(A) :- ast(A), not ast_operation(delete(A)), not ast_operation(replace(A,_)).
final(R) :- ast_operation(replace(_,R)).
final(N) :- ast_operation(add(N)).

#show final/1.
