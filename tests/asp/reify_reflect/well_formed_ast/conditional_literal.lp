% reified fact representation of program:
% #program base.
% a :- b : c, d.

program("base",constants(0),statements(1)).
statements(1,0,rule(2)).
rule(2,literal(3),body_literals(7)).
% facts representing head literal a.
literal(3,"pos",symbolic_atom(4)).
symbolic_atom(4,function(5)).
function(5,a,terms(6)).
% facts representing conditional literal b : c, d.
body_literals(7,0,conditional_literal(8)).
conditional_literal(8,literal(9),literals(13)).
literal(9,"pos",symbolic_atom(10)).
symbolic_atom(10,function(11)).
function(11,b,terms(12)).
literals(13,0,literal(14)).
literal(14,"pos",symbolic_atom(15)).
symbolic_atom(15,function(16)).
function(16,c,terms(17)).
literals(13,1,literal(18)).
literal(18,"pos",symbolic_atom(19)).
symbolic_atom(19,function(20)).
function(20,d,terms(21)).
