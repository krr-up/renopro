% reified fact representation of program:
% #program base.
% a(1..3).

program(0,"base",constants(1),statements(2)).
statements(2,0,rule(3)).
rule(3,literal(4),body_literals(11)).
literal(4,"pos",symbolic_atom(5)).
symbolic_atom(5,function(6)).
function(6,a,terms(7)).
terms(7,0,interval(8)).
interval(8,number(9),number(10)).
number(9,1).
number(10,3).
