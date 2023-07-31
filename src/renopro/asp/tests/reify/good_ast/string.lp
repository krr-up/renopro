% reified fact representation of program:
% #program base.
% yummy("carrot").

program("base",constants(0),statements(1)).
statements(1,0,rule(2)).
rule(2,literal(3),body_literals(8)).
% facts representing head literal yummy("carrot").
literal(3,"pos",symbolic_atom(4)).
symbolic_atom(4,function(5)).
function(5,yummy,terms(6)).
terms(6,0,string(7)).
string(7,"carrot").
