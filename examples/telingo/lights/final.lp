#program base.
push((T+1)) :- initially(T); time(T).
#false :- red(T); green(T); time(T).
red(T) :- not green(T); time(T).
green((T+2)) :- push(T); time(T).
red(T) :- green((T-1)); green((T-2)); time(T).
time((0..6)).
initially(0).
finally(6).
