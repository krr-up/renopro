% reified fact representation of program:
% #program base.
% pool((1;a)).

program("base",constants(0),statements(1)).
statements(1,0,rule(2)).
rule(2,literal(3),body_literals(12)).
% facts representing head literal pool(1;a).
literal(3,"pos",symbolic_atom(4)).
symbolic_atom(4,function(5)).
function(5,pool,terms(6)).
terms(6,0,pool(7)).
pool(7,terms(8)).
terms(8,0,number(9)).
number(9,1).
terms(8,1,function(10)).
function(10,a,terms(11)).
