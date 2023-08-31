#program base.
a(T) :- initially(T); time(T).
a(T) :- a((T-1)); time(T).
b((T+1)) :- b(T); time(T).
time(0).
time(1).
initially(0).
