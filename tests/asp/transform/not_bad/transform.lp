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
max_lit_index(LT,Idx)
:- Idx = #max{ P : body_literals(LT,P,_); -1 }, rule(_,_,body_literals(LT)).


%% the transformation itself
ast(
		add((body_literals(LT,N+1,body_literal(new_id(LT,0)));
				 body_literal(new_id(LT,0),"not",symbolic_atom(new_id(LT,1)));
				 symbolic_atom(new_id(LT,1),function(new_id(LT,2)));
				 function(new_id(LT,2),bad,Fargs))))
:- rule(_,literal(L),body_literals(LT)),
	 literal(L,"pos",symbolic_atom(A)),
	 symbolic_atom(A,function(F)),
	 function(F,good,Fargs),
	 max_lit_index(LT,N).
