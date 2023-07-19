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
max_lit_index(LT,Idx)
:- Idx = #max{ P : body_literal_tuple(LT,P,_); -1 }, rule(_,_,body_literal_tuple(LT)).


% the transformation itself
ast_operation(
add((body_literal_tuple(LT,N+1,literal(new_id(LT,0)));
		literal(new_id(LT,0),"not",symbolic_atom(new_id(LT,1)));
		symbolic_atom(new_id(LT,1),function(new_id(LT,2)));
		function(new_id(LT,2),bad,Fargs))))
:- rule(_,literal(L),body_literal_tuple(LT)),
	 literal(L,"pos",symbolic_atom(A)),
	 symbolic_atom(A,function(F)),
	 function(F,good,Fargs),
	 max_lit_index(LT,N).
