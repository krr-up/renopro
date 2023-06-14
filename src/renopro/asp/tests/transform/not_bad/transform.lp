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
add_not_bad(LitTuple,N+1,Fargs)
:- rule(_,literal(Lit),literal_tuple(LitTuple)),
	 literal(Lit,"pos",function(Func)),
	 function(Func,good,Fargs),
	 max_index(LitTuple,N).


% effect of add_not_bad
add(literal_tuple(LitTuple,Pos,literal(new_id(LitTuple,0))))
:- add_not_bad(LitTuple,Pos,Fargs).

add(literal(new_id(LitTuple,0),"not",function(new_id(LitTuple,1))))
:- add_not_bad(LitTuple,Pos,Fargs).

add(function(new_id(LitTuple,1),bad,Fargs))
:- add_not_bad(LitTuple,Pos,Fargs).
