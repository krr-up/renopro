% reified fact representation of program:
% #program base.
% 1 <= { a: b; c }.

program(0,"base",constants(1),statements(2)).
statements(2,0,rule(3)).
rule(3,aggregate(4),body_literals(25)).
aggregate(4,guard(5),aggregate_elements(7),guard(24)).
guard(5,"<=",number(6)).
number(6,1).
aggregate_elements(7,0,conditional_literal(8)).
conditional_literal(8,literal(9),literals(13)).
literal(9,"pos",symbolic_atom(10)).
symbolic_atom(10,function(11)).
function(11,a,terms(12)).
literals(13,0,literal(14)).
literal(14,"pos",symbolic_atom(15)).
symbolic_atom(15,function(16)).
function(16,b,terms(17)).
aggregate_elements(7,1,conditional_literal(18)).
conditional_literal(18,literal(19),literals(23)).
literal(19,"pos",symbolic_atom(20)).
symbolic_atom(20,function(21)).
function(21,c,terms(22)).
