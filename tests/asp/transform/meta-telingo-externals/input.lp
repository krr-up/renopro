#program initially.
a(1).

#program always.
next(a(X)) :- prev(prev(a(X))).
b :- not(a(X)).

#program finally
a(1).

modal(next,1,1).
modal(prev,1,1).

modal(until,2,1).
modal(until,2,2).
