% reified fact representation of program:
% #program base.
% a :- b : c, d.

program(0,"base",constants(1),statements(2)).
statements(2,0,rule(3)).
rule(3,literal(4),body_literals(8)).
% facts representing head literal a.
literal(4,"pos",symbolic_atom(5)).
symbolic_atom(5,function(6)).
function(6,a,terms(7)).
% facts representing conditional literal b : c, d.
body_literals(8,0,conditional_literal(9)).
conditional_literal(9,literal(10),literals(14)).
literal(10,"pos",symbolic_atom(11)).
symbolic_atom(11,function(12)).
function(12,b,terms(13)).
literals(14,0,literal(15)).
literal(15,"pos",symbolic_atom(16)).
symbolic_atom(16,function(17)).
function(17,c,terms(18)).
literals(14,1,literal(19)).
literal(19,"pos",symbolic_atom(20)).
symbolic_atom(20,function(21)).
function(21,d,terms(22)).
