% reified fact representation of program:
% #program base.
% rel(Y,X) :- rel(X,Y).

program("base",constants(0),statements(1)).
statements(1,0,rule(2)).
rule(2,literal(3),body_literals(9)).
% facts representing head literal rel(Y,X).
literal(3,"pos",symbolic_atom(4)).
symbolic_atom(4,function(5)).
function(5,rel,terms(6)).
terms(6,0,variable(7)).
variable(7,"Y").
terms(6,1,variable(8)).
variable(8,"X").
% facts representing body literal rel(X,Y).
body_literals(9,0,literal(10)).
literal(10,"pos",symbolic_atom(11)).
symbolic_atom(11,function(12)).
function(12,rel,terms(13)).
terms(13,0,variable(14)).
variable(14,"X").
terms(13,1,variable(15)).
variable(15,"Y").
