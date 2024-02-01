until(a(X),b(Y)); not next(c(Y+X)) :- prev(d(X)), eventually(e(Y)).
f(X) :- wprev(g(X)).
h(X); i(Y) :- since(j(X),k(Y)).
h(X); i(Y) :- trigger(j(X),k(Y)).
prev(l(X)) :- m(X), not n(X), not wprev(o(X)).

