% reified fact representation of program:
% #program base.
% rel(Y,X) :- rel(X,Y).

program("base",constant_tuple(0),statement_tuple(1)).
statement_tuple(1,0,rule(2)).
rule(2,literal(3),literal_tuple(9)).
% facts representing head literal rel(Y,X).
literal(3,"pos",atom(4)).
atom(4,function(5)).
function(5,rel,term_tuple(6)).
term_tuple(6,0,variable(7)).
variable(7,"Y").
term_tuple(6,1,variable(8)).
variable(8,"X").
% facts representing body literal rel(X,Y).
literal_tuple(9,0,literal(10)).
literal(10,"pos",atom(11)).
atom(11,function(12)).
function(12,rel,term_tuple(13)).
term_tuple(13,0,variable(14)).
variable(14,"X").
term_tuple(13,1,variable(15)).
variable(15,"Y").
