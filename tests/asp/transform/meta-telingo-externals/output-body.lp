#external initially. [false]
#external finally : n. [false]
#external initially : m. [false]
#external prev(prev(b)) : d; b. [false]
#external until(c,prev(d)) : d; b. [false]
#external until(b(X),next(c((X+Y)))) : e(X); d(Y). [false]
#external next(d(Y)) : e(X); d(Y). [false]
#external prev(h(Y)) : i(X); h(Y). [false]
#external next(wprev(k(X))). [false]
#program base.
a :- initially.
#program base.
l :- not initially; m.
#program base.
e :- prev(prev(b)); until(c,prev(d)).
a(X) :- until(b(X),next(c((X+Y)))): next(d(Y)); e(X).
f(X) :- 3 < #sum { 1: prev(h(Y)) }; i(X).
j(X) :- next(wprev(k(X))).
#program base.
#false :- finally; n.
#program base.
