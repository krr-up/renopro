% reified fact representation of program:
% #program base.
% next(move(a)).

program("base",constant_tuple(0),statement_tuple(1)).
statement_tuple(1,0,rule(2)).
rule(2,literal(3),literal_tuple(9)).
% facts representing head literal rel(2,1).
literal(3,"pos",function(4)).
function(4,next,term_tuple(5)).
term_tuple(5,0,function(6)).
function(6,move,term_tuple(7)).
term_tuple(7,0,constant(8)).
constant(8,a).
