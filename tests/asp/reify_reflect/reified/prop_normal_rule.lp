% reified fact representation of program:
% #program base.
% a :- b; not c.

program(0,"base",constants(1),statements(2)).
statements(2,0,rule(3)).
rule(3,literal(4),body_literals(8)).
% facts representing head literal a.
literal(4,"pos",symbolic_atom(5)).
symbolic_atom(5,function(6)).
function(6,a,terms(7)).
% facts representing body literal b.
body_literals(8,0,body_literal(9)).
body_literal(9,"pos",symbolic_atom(10)).
symbolic_atom(10,function(11)).
function(11,b,terms(12)).
% facts representing body literal not c.
body_literals(8,1,body_literal(13)).
body_literal(13,"not",symbolic_atom(14)).
symbolic_atom(14,function(15)).
function(15,c,terms(16)).
