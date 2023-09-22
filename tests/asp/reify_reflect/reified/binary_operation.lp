program(0,"base",constants(1),statements(2)).
statements(2,0,rule(3)).
rule(3,literal(4),body_literals(12)).
% facts representing head literal equal((2+1),2).
literal(4,"pos",symbolic_atom(5)).
symbolic_atom(5,function(6)).
function(6,equal,terms(7)).
% facts representing the binary operation (2+1).
terms(7,0,binary_operation(8)).
binary_operation(8,"+",number(9),number(10)).
number(9,1).
number(10,1).
terms(7,1,number(11)).
number(11,2).
