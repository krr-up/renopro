% reified fact representation of program:
% #program base.
% a :- b; not c.

program("base",constant_tuple(0),statement_tuple(1)).
statement_tuple(1,0,rule(2)).
rule(2,literal(3),literal_tuple(5)).
% facts representing head literal a.
literal(3,"pos",constant(4)).
constant(4,a).
% facts representing body literal b.
literal_tuple(5,0,literal(6)).
literal(6,"pos",constant(7)).
constant(7,b).
% facts representing body literal not c.
literal_tuple(5,1,literal(8)).
literal(8,"not",constant(9)).
constant(9,c).
