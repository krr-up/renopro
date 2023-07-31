% reified fact representation of program:
% #program base.
% equal((1+1),2).

program("base",constants(0),statements(1)).
statements(1,0,rule(2)).
rule(2,literal(3),body_literals(11)).
% facts representing head literal equal((1+1),2).
literal(3,"pos",symbolic_atom(4)).
symbolic_atom(4,function(5)).
function(5,equal,terms(6)).
% facts representing the binary operation (1+1).
terms(6,0,binary_operation(7)).
binary_operation(7,"+",number(8),number(9)).
number(8,1).
number(9,1).
terms(6,1,number(10)).
number(10,2).
