% reified fact representation of program:
% #program base.
% :- a.

program("base",constants(0),statements(1)).
statements(1,0,rule(2)).
rule(2,literal(3),body_literals(5)).
literal(3,"pos",boolean_constant(4)).
boolean_constant(4,false).
body_literals(5,0,body_literal(6)).
body_literal(6,"pos",symbolic_atom(7)).
symbolic_atom(7,function(8)).
function(8,a,terms(9)).
