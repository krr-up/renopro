% reified fact representation of program:
% #program base.
% 1 <= {a(X): b(X); c} :- d(X).

program("base",constants(0),statements(1)).
statements(1,0,rule(2)).
rule(2,aggregate(3),body_literals(26)).
aggregate(3,guard(4),agg_elements(6),guard(25)).
guard(4,"<=",number(5)).
number(5,1).
agg_elements(6,0,conditional_literal(7)).
conditional_literal(7,literal(8),literals(13)).
literal(8,"pos",symbolic_atom(9)).
symbolic_atom(9,function(10)).
function(10,a,terms(11)).
terms(11,0,variable(12)).
variable(12,"X").
literals(13,0,literal(14)).
literal(14,"pos",symbolic_atom(15)).
symbolic_atom(15,function(16)).
function(16,b,terms(17)).
terms(17,0,variable(18)).
variable(18,"X").
agg_elements(6,1,conditional_literal(19)).
conditional_literal(19,literal(20),literals(24)).
literal(20,"pos",symbolic_atom(21)).
symbolic_atom(21,function(22)).
function(22,c,terms(23)).
body_literals(26,0,literal(27)).
literal(27,"pos",symbolic_atom(28)).
symbolic_atom(28,function(29)).
function(29,d,terms(30)).
terms(30,0,variable(31)).
variable(31,"X").
