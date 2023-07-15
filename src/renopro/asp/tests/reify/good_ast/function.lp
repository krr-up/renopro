% reified fact representation of program:
% #program base.
% rel(2,1) :- rel(1,2).

program("base",constant_tuple(0),statement_tuple(1)).
statement_tuple(1,0,rule(2)).
rule(2,literal(3),literal_tuple(9)).
% facts representing head literal rel(2,1).
literal(3,"pos",atom(4)).
atom(4,function(5)).
function(5,rel,term_tuple(6)).
term_tuple(6,0,number(7)).
number(7,2).
term_tuple(6,1,number(8)).
number(8,1).
% facts representing body literal rel(1,2).
literal_tuple(9,0,literal(10)).
literal(10,"pos",atom(11)).
atom(11,function(12)).
function(12,rel,term_tuple(13)).
term_tuple(13,0,number(14)).
number(14,1).
term_tuple(13,1,number(15)).
number(15,2).
