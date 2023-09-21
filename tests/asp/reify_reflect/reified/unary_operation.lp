% reified fact representation of program:
% #program base.
% neg(-1).

program(0,"base",constants(1),statements(2)).
statements(2,0,rule(3)).
rule(3,literal(4),body_literals(10)).
% facts representing head literal neg((-1)).
literal(4,"pos",symbolic_atom(5)).
symbolic_atom(5,function(6)).
function(6,neg,terms(7)).
% facts representing the unary operation (-1).
terms(7,0,unary_operation(8)).
unary_operation(8,"-",number(9)).
number(9,1).
