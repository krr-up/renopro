% reified fact representation of program:
% #program base.
% &a { +1!-"b" }.

program("base",constants(0),statements(1)).
statements(1,0,rule(2)).
rule(2,theory_atom(3),body_literals(15)).
theory_atom(3,symbolic_atom(4),theory_atom_elements(7),theory_guard(14)).
symbolic_atom(4,function(5)).
function(5,a,terms(6)).
theory_atom_elements(7,0,theory_terms(8),literals(13)).
theory_terms(8,0,theory_unparsed_term(9)).
theory_unparsed_term(9,theory_unparsed_term_elements(10)).
theory_unparsed_term_elements(10,0,theory_operators(11),number(12)).
theory_operators(11,0,"+").
number(12,1).
theory_unparsed_term_elements(10,1,theory_operators(12),string(13)).
theory_operators(12,0,"!").
theory_operators(12,1,"-").
string(13,"b").
