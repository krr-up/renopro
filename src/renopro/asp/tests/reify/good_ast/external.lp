% reified fact representation of program:
% #program base.
% #external a(X) : c(X); d(e(X)).

program("base",constant_tuple(0),statement_tuple(1)).
statement_tuple(1,0,external(2)).
external(2,symbolic_atom(3),literal_tuple(7),false).
symbolic_atom(3,function(4)).
function(4,a,term_tuple(5)).
term_tuple(5,0,variable(6)).
variable(6,"X").
literal_tuple(7,0,literal(8)).
literal(8,"pos",symbolic_atom(9)).
symbolic_atom(9,function(10)).
function(10,c,term_tuple(11)).
term_tuple(11,0,variable(12)).
variable(12,"X").
literal_tuple(7,1,literal(13)).
literal(13,"pos",symbolic_atom(14)).
symbolic_atom(14,function(15)).
function(15,d,term_tuple(16)).
term_tuple(16,0,function(17)).
function(17,e,term_tuple(18)).
term_tuple(18,0,variable(19)).
variable(19,"X").
