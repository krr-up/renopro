#external c : a. [false]
#external b(X) : e(X). [false]
#external j(X) : i(X); h(Y). [false]
#external k(X) : j(X). [false]
#external c((X+Y)) : e(X). [false]
#external d : a. [false]
#external initial. [false]
#external wnext(d(Y)) : e(X). [false]
#external prev(h(Y)) : i(X); h(Y). [false]
#program base.
a :- initial; b.
until(c,prev(d)) :- a.
release(b(X),next(c((X+Y)))): wnext(d(Y)) :- e(X).
3 < #max { 1: next(j(X)): prev(h(Y)) } :- i(X).
prev(wprev(k(X))) :- j(X).
#program base.
