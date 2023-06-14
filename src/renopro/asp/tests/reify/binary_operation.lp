% reified fact representation of program:
% #program base.
% equal((1+1),2).

program("base",constant_tuple(0),statement_tuple(1)).
statement_tuple(1,0,rule(2)).
rule(2,literal(3),literal_tuple(10)).
% facts representing head literal equal((1+1),2).
literal(3,"pos",function(4)).
function(4,equal,term_tuple(5)).
% facts representing the binary operation (1+1).
term_tuple(5,0,binary_operation(6)).
binary_operation(6,"+",number(7),number(8)).
number(7,1).
number(8,1).
term_tuple(5,1,number(9)).
number(9,2).
