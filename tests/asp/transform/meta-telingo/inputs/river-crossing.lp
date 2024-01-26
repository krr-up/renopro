% clingo encoding.lp --output=reify | clingo - meta-telingo.lp -c horizon=6 0
% clingo encoding.lp --output=reify | clingo - meta-telingo.lp -c horizon=7 0

%#program always.

item(fox;beans;goose).
route(river_bank,far_bank). route(far_bank,river_bank).
eats(fox,goose). eats(goose,beans).

%#program initial.

% Initial, everything is at the river bank
at(farmer,river_bank) :- initial.
at(X,river_bank) :- item(X), initial.

%#program dynamic.

% Choose moving exactly one item
0 {move(X):item(X)} 1 :- not initial.

% The farmer is always moving
move(farmer) :- not initial.

% Effect axiom for moving
at(X,B) :- prev(at(X,A)), move(X), route(A,B), not initial.

% Items not moved stay where they are
at(X,A) :- prev(at(X,A)), not move(X), not initial.

% The farmer cannot move item X if it is at the opposite bank
:- move(X), item(X), prev(at(farmer,A)), not prev(at(X,A)), not initial.

%#program always.

% Neither farmer nor items can be at both banks
:- at(X,A), at(X,B), A<B.

% If an item eats anther items, we cannot leave them alone
:- eats(X,Y), at(X,A), at(Y,A), not at(farmer,A).

%#program final.

% Finally, everything has to be at the far bank
:- at(X,river_bank), final.

% Domain predicates

loc(L) :- route(L,_).
animal(farmer).
animal(X) :- item(X).


#show show(at(X,A)) : animal(X), loc(A).
#show show(move(X)) : animal(X).

#external initial.
#external final.
#external move(X) : animal(X).
#external prev(at(X,A)) : animal(X), loc(A).
