% reified fact representation of program:
% #program base.
% rel(2,1) :- rel(1,2).

program("base",constants(0),statements(1)).
statements(1,0,rule(2)).
rule(2,literal(3),body_literals(9)).
% facts representing head literal rel(2,1).
literal(3,"pos",symbolic_atom(4)).
symbolic_atom(4,function(5)).
function(5,rel,terms(6)).
terms(6,0,number(7)).
number(7,2).
terms(6,1,number(8)).
number(8,1).
% facts representing body literal rel(1,2).
body_literals(9,0,body_literal(10)).
body_literal(10,"pos",symbolic_atom(11)).
symbolic_atom(11,function(12)).
function(12,rel,terms(13)).
terms(13,0,number(14)).
number(14,1).
terms(13,1,number(15)).
number(15,2).
