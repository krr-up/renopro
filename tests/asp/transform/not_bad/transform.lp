% ast transformation:

% transforms every rule with single symbolic head
% literal with name "good", adding a negated symbolic literal to the
% body with name "bad", and same arguments as the head literal

% examples:
% good(X) :- person(X).
% ->
%	good(X) :- person(X), not bad(X).
%
% good(dog(X,"spotty")) :- cute(dog(X,"spotty")).
% ->
% good(dog(X,"spotty")) :- cute(dog(X,"spotty")), not bad(dog(X,"spotty")).

% auxiliary predicate to get maximal index within a tuple of rule body literals
#program step(not_bad).
#const not_bad=0.

ast(replace(R,
					  rule(H,(body_literal("not",(symbolic_atom(function("bad",Terms)))),B))
))
  :- node(R), R=rule(H,B) H=literal("",symbolic_atom(function(Arity,"good",Terms))).
