good(X) :- person(X); not bad(X).
good(dog(X,"spotty")) :- cute(dog(X,"spotty")); not bad(dog(X,"spotty")).
