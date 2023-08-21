% reified fact representation of program:
% #program base.
% neg(-1).

program("base",constants(0),statements(1)).
statements(1,0,rule(2)).
rule(2,literal(3),body_literals(9)).
% facts representing head literal neg((-1)).
literal(3,"pos",symbolic_atom(4)).
symbolic_atom(4,function(5)).
function(5,neg,terms(6)).
% facts representing the unary operation (-1).
terms(6,0,unary_operation(7)).
unary_operation(7,"-",number(8)).
number(8,1).
