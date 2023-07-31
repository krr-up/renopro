% reified fact representation of program:
% #program base.
% good(human).

program("base",constants(0),statements(1)).
statements(1,0,rule(2)).
rule(2,literal(3),body_literals(9)).
% facts representing head literal yummy("carrot").
literal(3,"pos",symbolic_atom(4)).
symbolic_atom(4,function(5)).
function(5,good,terms(6)).
terms(6,0,function(7)).
function(7,human,terms(8)).
