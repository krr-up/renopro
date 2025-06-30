#external initial. [false]
#external weak_next(d(Y)) : e(X). [false]
#external prev(h(Y)) : h(Y); i(X). [false]
#external c : a. [false]
#external d : a. [false]
#external b(X) : e(X). [false]
#external c((X+Y)) : e(X). [false]
#external j(X) : h(Y); i(X). [false]
#external k(X) : j(X). [false]
#program base.
#program base.
a :- initial; b.
until(c,prev(d)) :- a.
release(b(X),next(c((X+Y)))): weak_next(d(Y)) :- e(X).
3 < #max { 1: next(j(X)): prev(h(Y)) } :- i(X).
prev(weak_prev(k(X))) :- j(X).
