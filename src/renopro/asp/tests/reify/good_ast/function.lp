% reified fact representation of program:
% #program base.
% next(move(a)).

program("base",constant_tuple(0),statement_tuple(1)).
statement_tuple(1,0,rule(2)).
rule(2,literal(3),body_literal_tuple(11)).
% facts representing head literal rel(2,1).
literal(3,"pos",symbolic_atom(4)).
symbolic_atom(4,function(5)).
function(5,next,term_tuple(6)).
term_tuple(6,0,function(7)).
function(7,move,term_tuple(8)).
term_tuple(8,0,function(9)).
function(9,a,term_tuple(10)).
