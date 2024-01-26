next(push) :- initial.

:- red, green.
red :- not green.
next(next(green)):- push.
red :- prev(green), prev(prev(green)).
