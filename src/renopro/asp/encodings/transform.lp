% tagging of input ast elements
ast(rule(H,B)) :- rule(H,B).
ast(literal_tuple(I,P,E)) :- literal_tuple(I,P,E).
ast(term_tuple(I,P,E)) :- term_tuple(I,P,E).

final(A) :- ast(A), not delete(A), not replace(A,_).
final(R) :- replace(_,R).
final(N) :- new(N).

#show final/1.
