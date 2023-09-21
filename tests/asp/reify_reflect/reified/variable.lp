% reified fact representation of program:
% #program base.
% rel(Y,X) :- rel(X,Y).

program(0,"base",constants(1),statements(2)).
statements(2,0,rule(3)).
rule(3,literal(4),body_literals(10)).
% facts representing head literal rel(Y,X).
literal(4,"pos",symbolic_atom(5)).
symbolic_atom(5,function(6)).
function(6,rel,terms(7)).
terms(7,0,variable(8)).
variable(8,"Y").
terms(7,1,variable(9)).
variable(9,"X").
% facts representing body literal rel(X,Y).
body_literals(10,0,body_literal(11)).
body_literal(11,"pos",symbolic_atom(12)).
symbolic_atom(12,function(13)).
function(13,rel,terms(14)).
terms(14,0,variable(15)).
variable(15,"X").
terms(14,1,variable(16)).
variable(16,"Y").
