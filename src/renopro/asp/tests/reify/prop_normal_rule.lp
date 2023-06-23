% reified fact representation of program:
% #program base.
% a :- b; not c.

program("base",constant_tuple(0),statement_tuple(1)).
statement_tuple(1,0,rule(2)).
rule(2,literal(3),literal_tuple(7)).
% facts representing head literal a.
literal(3,"pos",atom(4)).
atom(4,function(5)).
function(5,a,term_tuple(6)).
% facts representing body literal b.
literal_tuple(7,0,literal(8)).
literal(8,"pos",atom(9)).
atom(9,function(10)).
function(10,b,term_tuple(11)).
% facts representing body literal not c.
literal_tuple(7,1,literal(12)).
literal(12,"not",atom(13)).
atom(13,function(14)).
function(14,c,term_tuple(15)).
