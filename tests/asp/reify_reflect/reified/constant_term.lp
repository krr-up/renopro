% reified fact representation of program:
% #program base.
% good(human).

program(0,"base",constants(1),statements(2)).
statements(2,0,rule(3)).
rule(3,literal(4),body_literals(10)).
% facts representing head literal yummy("carrot").
literal(4,"pos",symbolic_atom(5)).
symbolic_atom(5,function(6)).
function(6,good,terms(7)).
terms(7,0,function(8)).
function(8,human,terms(9)).
