a(1) :- initially.
next(a(X)) :- prev(prev(a(X))).
a(1) :- finally.
#external prev(prev(a(X))) : a(X).
#external a(X) : next(a(X)).

c :- not(a(X)).
c :- while(false,a(X)).
#external not(a(X)) : a(X).
#external while(false,a(X)).

#external until(a(X),b(X)) : a(X),b(X).



a(f(X)) :- prev(a(X)).

% this has infinite grounding though with the program above
#external prev(a(X)) : a(X).

% argument restricted programs.
% 
