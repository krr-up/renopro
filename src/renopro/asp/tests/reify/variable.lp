% reified fact representation of program:
% #program base.
% rel(Y,X) :- rel(X,Y).

program("base",constant_tuple(0),statement_tuple(1)).
statement_tuple(1,0,rule(2)).
rule(2,literal(3),literal_tuple(8)).
% facts representing head literal rel(2,1).
literal(3,"pos",function(4)).
function(4,rel,term_tuple(5)).
term_tuple(5,0,variable(6)).
variable(6,"Y").
term_tuple(5,1,variable(7)).
variable(7,"X").
% facts representing body literal rel(1,2).
literal_tuple(8,0,literal(9)).
literal(9,"pos",function(10)).
function(10,rel,term_tuple(11)).
term_tuple(11,0,variable(12)).
variable(12,"X").
term_tuple(11,1,variable(13)).
variable(13,"Y").
