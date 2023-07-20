% reified fact representation of program:
% #program base.
% :- a.

program("base",constant_tuple(0),statement_tuple(1)).
statement_tuple(1,0,rule(2)).
rule(2,literal(3),body_literal_tuple(5)).
literal(3,"pos",boolean_constant(4)).
boolean_constant(4,false).
body_literal_tuple(5,0,literal(6)).
literal(6,"pos",symbolic_atom(7)).
symbolic_atom(7,function(8)).
function(8,a,term_tuple(9)).
