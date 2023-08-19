% reified fact representation of program:
% #program base.
% a; b : c, not d.

program("base",constants(0),statements(1)).
statements(1,0,rule(2)).
rule(2,disjunction(3),body_literals(24)).
disjunction(3,0,conditional_literal(4)).
conditional_literal(4,literal(5),literals(9)).
literal(5,"pos",symbolic_atom(6)).
symbolic_atom(6,function(7)).
function(7,a,terms(8)).
disjunction(3,1,conditional_literal(10)).
conditional_literal(10,literal(11),literals(15)).
literal(11,"pos",symbolic_atom(12)).
symbolic_atom(12,function(13)).
function(13,b,terms(14)).
literals(15,0,literal(16)).
literal(16,"pos",symbolic_atom(17)).
symbolic_atom(17,function(18)).
function(18,c,terms(19)).
literals(15,1,literal(20)).
literal(20,"not",symbolic_atom(21)).
symbolic_atom(21,function(22)).
function(22,d,terms(23)).
