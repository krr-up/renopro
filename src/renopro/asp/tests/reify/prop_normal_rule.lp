% reified fact representation of program:
% #program base.
% a :- b; not c.

program("base",constant_tuple(0),statement_tuple(1)).
statement_tuple(1,0,rule(2)).
rule(2,literal(3),literal_tuple(6)).
% facts representing head literal a.
literal(3,"pos",function(4)).
function(4,a,term_tuple(5)).
% facts representing body literal b.
literal_tuple(6,0,literal(7)).
literal(7,"pos",function(8)).
function(8,b,term_tuple(9)).
% facts representing body literal not c.
literal_tuple(6,1,literal(10)).
literal(10,"not",function(11)).
function(11,c,term_tuple(12)).
