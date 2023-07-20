% reified fact representation of program:
% #program base.
% #false. #true.

program("base",constant_tuple(0),statement_tuple(1)).
statement_tuple(1,0,rule(2)).
rule(2,literal(3),body_literal_tuple(5)).
% facts representing head literal a.
literal(3,"pos",boolean_constant(4)).
boolean_constant(4,false).
statement_tuple(1,1,rule(6)).
rule(6,literal(7),body_literal_tuple(9)).
literal(7,"pos",boolean_constant(8)).
boolean_constant(8,true).
