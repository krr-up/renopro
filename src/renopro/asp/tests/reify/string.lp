% reified fact representation of program:
% #program base.
% yummy("carrot").

program("base",constant_tuple(0),statement_tuple(1)).
statement_tuple(1,0,rule(2)).
rule(2,literal(3),literal_tuple(7)).
% facts representing head literal yummy("carrot").
literal(3,"pos",function(4)).
function(4,yummy,term_tuple(5)).
term_tuple(5,0,string(6)).
string(6,"carrot").
