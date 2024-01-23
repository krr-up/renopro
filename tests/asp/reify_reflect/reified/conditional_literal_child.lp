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


child(program(0),constants(1)).
child(program(0),statements(2)).
child(statements(2),rule(3)).
child(rule(3),literal(4)).
child(literal(4),symbolic_atom(5)).
child(symbolic_atom(5),function(6)).
child(function(6),terms(7)).
child(rule(3),body_literals(8)).
child(body_literals(8),conditional_literal(9)).
child(conditional_literal(9),literal(10)).
child(literal(10),symbolic_atom(11)).
child(symbolic_atom(11),function(12)).
child(function(12),terms(13)).
child(conditional_literal(9),literals(14)).
child(literals(14),literal(15)).
child(literal(15),symbolic_atom(16)).
child(symbolic_atom(16),function(17)).
child(function(17),terms(18)).
child(literals(14),literal(19)).
child(literal(19),symbolic_atom(20)).
child(symbolic_atom(20),function(21)).
child(function(21),terms(22)).
