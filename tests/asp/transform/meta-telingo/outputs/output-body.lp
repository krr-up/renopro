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
#external prev(h(Y)) : h(Y); i(X). [false]
#external next(d(Y)) : d(Y); e(X). [false]
#external until(b(X),next(c((X+Y)))) : d(Y); e(X). [false]
#external initial. [false]
#external initial. [false]
#external prev(prev(b)). [false]
#external until(c,prev(d)). [false]
#external next(wprev(k(X))). [false]
#external final. [false]
