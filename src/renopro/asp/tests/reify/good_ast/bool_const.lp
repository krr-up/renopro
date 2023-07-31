% reified fact representation of program:
% #program base.
% #false. #true.

program("base",constants(0),statements(1)).
statements(1,0,rule(2)).
rule(2,literal(3),body_literals(5)).
% facts representing head literal a.
literal(3,"pos",boolean_constant(4)).
boolean_constant(4,false).
statements(1,1,rule(6)).
rule(6,literal(7),body_literals(9)).
literal(7,"pos",boolean_constant(8)).
boolean_constant(8,true).
