#program initial.
a.
#program dynamic.
l :- m.
#program final.
:- n.
#program always.
e :- prev(prev(b)), until(c,prev(d)).
a(X) :- until(b(X),next(c(X+Y))) : next(d(Y)); e(X).
f(X) :- #sum{1:prev(h(Y))} > 3, i(X).
j(X) :- next(wprev(k(X))).

