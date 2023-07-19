% reified fact representation of program:
% #program base.
% good(human).

program("base",constant_tuple(0),statement_tuple(1)).
statement_tuple(1,0,rule(2)).
rule(2,literal(3),body_literal_tuple(9)).
% facts representing head literal yummy("carrot").
literal(3,"pos",symbolic_atom(4)).
symbolic_atom(4,function(5)).
function(5,good,term_tuple(6)).
term_tuple(6,0,function(7)).
function(7,human,term_tuple(8)).
