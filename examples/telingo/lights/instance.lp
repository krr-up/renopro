next(push) :- initially.

:- red, green.
red :- not green.
next(next(green)):- push.
red :- prev(green), prev(prev(green)).