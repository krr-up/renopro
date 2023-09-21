% reified fact representation of program:
% #program base.
% 1 <= { a } < 3.

program(0,"base",constants(1),statements(2)).
statements(2,0,rule(3)).
rule(3,aggregate(4),body_literals(16)).
aggregate(4,guard(5),aggregate_elements(7),guard(14)).
guard(5,"<=",number(6)).
number(6,1).
aggregate_elements(7,0,conditional_literal(8)).
conditional_literal(8,literal(9),literals(13)).
literal(9,"pos",symbolic_atom(10)).
symbolic_atom(10,function(11)).
function(11,a,terms(12)).
guard(14,"<",number(15)).
number(15,3).
