% reified fact representation of program:
% #program base.
% a.

program("base",constant_tuple(0),statement_tuple(1)).
statement_tuple(1,0,rule(2)).
rule(2,literal(3),literal_tuple(5)).
% facts representing head literal a.
literal(3,"pos",constant(4)).
constant(4,a).
