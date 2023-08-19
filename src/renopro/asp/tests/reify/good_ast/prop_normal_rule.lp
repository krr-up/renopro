% reified fact representation of program:
% #program base.
% a :- b; not c.

program("base",constants(0),statements(1)).
statements(1,0,rule(2)).
rule(2,literal(3),body_literals(7)).
% facts representing head literal a.
literal(3,"pos",symbolic_atom(4)).
symbolic_atom(4,function(5)).
function(5,a,terms(6)).
% facts representing body literal b.
body_literals(7,0,body_literal(8)).
body_literal(8,"pos",symbolic_atom(9)).
symbolic_atom(9,function(10)).
function(10,b,terms(11)).
% facts representing body literal not c.
body_literals(7,1,body_literal(12)).
body_literal(12,"not",symbolic_atom(13)).
symbolic_atom(13,function(14)).
function(14,c,terms(15)).
