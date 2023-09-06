% reified fact representation of program:
% #program base.
% &a{[1,b]}.

program("base",constants(0),statements(1)).
statements(1,0,rule(2)).
rule(2,theory_atom(3),body_literals(16)).
theory_atom(3,symbolic_atom(4),theory_atom_elements(7),theory_guard(15)).
symbolic_atom(4,function(5)).
function(5,a,terms(6)).
theory_atom_elements(7,0,theory_terms(8),literals(14)).
theory_terms(8,0,theory_sequence(9)).
theory_sequence(9,"[]",theory_terms(10)).
theory_terms(10,0,number(11)).
number(11,1).
theory_terms(10,1,function(12)).
function(12,b,terms(13)).
