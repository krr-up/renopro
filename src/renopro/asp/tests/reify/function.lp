% reified fact representation of program:
% #program base.
% rel(2,1) :- rel(1,2).

program("base",constant_tuple(0),statement_tuple(1)).
statement_tuple(1,0,rule(2)).
rule(2,literal(3),literal_tuple(8)).
% facts representing head literal rel(2,1).
literal(3,"pos",function(4)).
function(4,rel,term_tuple(5)).
term_tuple(5,0,number(6)).
number(6,2).
term_tuple(5,1,number(7)).
number(7,1).
% facts representing body literal rel(1,2).
literal_tuple(8,0,literal(9)).
literal(9,"pos",function(10)).
function(10,rel,term_tuple(11)).
term_tuple(11,0,number(12)).
number(12,1).
term_tuple(11,1,number(13)).
number(13,2).
