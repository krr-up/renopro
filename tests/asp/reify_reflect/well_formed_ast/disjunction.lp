% reified fact representation of program:
% #program base.
% a; b : c, not d.

program("base",constants(0),statements(1)).
statements(1,0,rule(2)).
rule(2,disjunction(3),body_literals(25)).
disjunction(3,conditional_literals(4)).
conditional_literals(4,0,conditional_literal(5)).
conditional_literal(5,literal(6),literals(10)).
literal(6,"pos",symbolic_atom(7)).
symbolic_atom(7,function(8)).
function(8,a,terms(9)).
conditional_literals(4,1,conditional_literal(11)).
conditional_literal(11,literal(12),literals(16)).
literal(12,"pos",symbolic_atom(13)).
symbolic_atom(13,function(14)).
function(14,b,terms(15)).
literals(16,0,literal(17)).
literal(17,"pos",symbolic_atom(18)).
symbolic_atom(18,function(19)).
function(19,c,terms(20)).
literals(16,1,literal(21)).
literal(21,"not",symbolic_atom(22)).
symbolic_atom(22,function(23)).
function(23,d,terms(24)).
