% reified fact representation of program:
% #program base.
% &a{} | b.

program("base",constants(0),statements(1)).
statements(1,0,rule(2)).
rule(2,theory_atom(3),body_literals(11)).
theory_atom(3,symbolic_atom(4),theory_atom_elements(7),theory_guard(8)).
symbolic_atom(4,function(5)).
function(5,a,terms(6)).
theory_guard(8,"|",function(9)).
function(9,b,terms(10)).
