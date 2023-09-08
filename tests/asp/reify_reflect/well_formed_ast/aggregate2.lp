% reified fact representation of program:
% #program base.
% 1 <= { a } < 3.

program("base",constants(0),statements(1)).
statements(1,0,rule(2)).
rule(2,aggregate(3),body_literals(15)).
aggregate(3,guard(4),agg_elements(6),guard(13)).
guard(4,"<=",number(5)).
number(5,1).
agg_elements(6,0,conditional_literal(7)).
conditional_literal(7,literal(8),literals(12)).
literal(8,"pos",symbolic_atom(9)).
symbolic_atom(9,function(10)).
function(10,a,terms(11)).
guard(13,"<",number(14)).
number(14,3).
