% reified fact representation of program:
% #program base.
% a(1..3).

program("base",constant_tuple(0),statement_tuple(1)).
statement_tuple(1,0,rule(2)).
rule(2,literal(3),body_literal_tuple(10)).
literal(3,"pos",symbolic_atom(4)).
symbolic_atom(4,function(5)).
function(5,a,term_tuple(6)).
term_tuple(6,0,interval(7)).
interval(7,number(8),number(9)).
number(8,1).
number(9,3).
