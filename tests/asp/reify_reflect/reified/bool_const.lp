% reified fact representation of program:
% #program base.
% #false. #true.

program(0,"base",constants(1),statements(2)).
statements(2,0,rule(3)).
rule(3,literal(4),body_literals(6)).
% facts representing head literal a.
literal(4,"pos",boolean_constant(5)).
boolean_constant(5,false).
statements(2,1,rule(7)).
rule(7,literal(8),body_literals(10)).
literal(8,"pos",boolean_constant(9)).
boolean_constant(9,true).
