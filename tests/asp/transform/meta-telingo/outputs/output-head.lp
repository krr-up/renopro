#program base.
a :- initial; b.
until(c,prev(d)) :- a.
release(b(X),next(c((X+Y)))): wnext(d(Y)) :- e(X).
3 < #max { 1: next(j(X)): prev(h(Y)) } :- i(X).
prev(wprev(k(X))) :- j(X).
#program base.
#external prev(h(Y)) : h(Y); i(X). [false]
#external wnext(d(Y)) : e(X). [false]
#external initial. [false]
#external k(X) : j(X). [false]
#external d : a. [false]
#external c((X+Y)) : e(X). [false]
#external j(X) : h(Y); i(X). [false]
#external b(X) : e(X). [false]
#external c : a. [false]
