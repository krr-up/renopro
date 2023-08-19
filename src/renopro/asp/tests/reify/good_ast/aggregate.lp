% reified fact representation of program:
% #program base.
% 1 <= { a: b; c }.

program("base",constants(0),statements(1)).
statements(1,0,rule(2)).
rule(2,aggregate(3),body_literals(24)).
aggregate(3,guard(4),agg_elements(6),guard(23)).
guard(4,"<=",number(5)).
number(5,1).
agg_elements(6,0,conditional_literal(7)).
conditional_literal(7,literal(8),literals(12)).
literal(8,"pos",symbolic_atom(9)).
symbolic_atom(9,function(10)).
function(10,a,terms(11)).
literals(12,0,literal(13)).
literal(13,"pos",symbolic_atom(14)).
symbolic_atom(14,function(15)).
function(15,b,terms(16)).
agg_elements(6,1,conditional_literal(17)).
conditional_literal(17,literal(18),literals(22)).
literal(18,"pos",symbolic_atom(19)).
symbolic_atom(19,function(20)).
function(20,c,terms(21)).
