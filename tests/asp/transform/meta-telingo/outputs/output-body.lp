#external initial. [false]
#external final. [false]
#external initial. [false]
#external prev(prev(b)). [false]
#external until(c,prev(d)). [false]
#external until(b(X),next(c((X+Y)))) : e(X); d(Y). [false]
#external next(d(Y)) : e(X); d(Y). [false]
#external prev(h(Y)) : i(X); h(Y). [false]
#external next(wprev(k(X))). [false]
#program base.
a :- initial.
#program base.
l :- not initial; m.
#program base.
e :- prev(prev(b)); until(c,prev(d)).
a(X) :- until(b(X),next(c((X+Y)))): next(d(Y)); e(X).
f(X) :- 3 < #sum { 1: prev(h(Y)) }; i(X).
j(X) :- next(wprev(k(X))).
#program base.
#false :- final; n.
#program base.
