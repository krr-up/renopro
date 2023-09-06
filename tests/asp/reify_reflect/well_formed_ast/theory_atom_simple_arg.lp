% reified fact representation of program:
% #program base.
% &a("b").

program("base",constants(0),statements(1)).
statements(1,0,rule(2)).
rule(2,theory_atom(3),body_literals(10)).
theory_atom(3,symbolic_atom(4),theory_atom_elements(8),theory_guard(9)).
symbolic_atom(4,function(5)).
function(5,a,terms(6)).
terms(6,0,string(7)).
string(7,"b").
