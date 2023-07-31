% reified fact representation of program:
% #program base.
% a(X); b(X) : c(X) :- d(X).

program("base",constants(0),statements(1)).
statements(1,0,rule(2)).
rule(2,disjunction(3),body_literals(21)).
disjunction(3,0,literal(4)).
literal(4,"pos",symbolic_atom(5)).
symbolic_atom(5,function(6)).
function(6,a,terms(7)).
terms(7,0,variable(8)).
variable(8,"X").
disjunction(3,1,conditional_literal(9)).
conditional_literal(9,literal(10),literals(15)).
literal(10,"pos",symbolic_atom(11)).
symbolic_atom(11,function(12)).
function(12,b,terms(13)).
terms(13,0,variable(14)).
variable(14,"X").
literals(15,0,literal(16)).
literal(16,"pos",symbolic_atom(17)).
symbolic_atom(17,function(18)).
function(18,c,terms(19)).
terms(19,0,variable(20)).
variable(20,"X").
body_literals(21,0,literal(22)).
literal(22,"pos",symbolic_atom(23)).
symbolic_atom(23,function(24)).
function(24,d,terms(25)).
terms(25,0,variable(26)).
variable(26,"X").
