% reified fact representation of program:
% #program base.
% yummy("carrot").

program("base",constant_tuple(0),statement_tuple(1)).
statement_tuple(1,0,rule(2)).
rule(2,literal(3),literal_tuple(8)).
% facts representing head literal yummy("carrot").
literal(3,"pos",atom(4)).
atom(4,function(5)).
function(5,yummy,term_tuple(6)).
term_tuple(6,0,string(7)).
string(7,"carrot").
