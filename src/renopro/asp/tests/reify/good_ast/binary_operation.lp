% reified fact representation of program:
% #program base.
% equal((1+1),2).

program("base",constant_tuple(0),statement_tuple(1)).
statement_tuple(1,0,rule(2)).
rule(2,literal(3),literal_tuple(11)).
% facts representing head literal equal((1+1),2).
literal(3,"pos",atom(4)).
atom(4,function(5)).
function(5,equal,term_tuple(6)).
% facts representing the binary operation (1+1).
term_tuple(6,0,binary_operation(7)).
binary_operation(7,"+",number(8),number(9)).
number(8,1).
number(9,1).
term_tuple(6,1,number(10)).
number(10,2).
