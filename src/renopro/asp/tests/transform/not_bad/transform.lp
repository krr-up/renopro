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

% auxiliary predicate to get maximal index within a tuple
max_index(I,N) :- N = #max{ P : literal_tuple(I,P,_); -1 }, literal_tuple(I,_,_).


% the transformation itself
add((literal_tuple(LT,N+1,literal(new_id(LT,0))));
		literal(new_id(LT,0),"not",function(new_id(LT,1)));
		function(new_id(LT,1),bad,Fargs))
:- rule(_,literal(L),literal_tuple(LT)),
	 literal(L,"pos",function(F)),
	 function(F,good,Fargs),
	 max_index(LT,N).
