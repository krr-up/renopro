#program base.
a(T) :- a((T-1)); time(T).
b((T+1)) :- b(T); time(T).
time((0..1)).
initially(0).
finally(1).
