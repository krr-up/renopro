% reified fact representation of program:
% #program base.
% #external a(X) : c(X); d(e(X)).

program("base",constants(0),statements(1)).
statements(1,0,external(2)).
external(2,symbolic_atom(3),body_literals(7),false).
symbolic_atom(3,function(4)).
function(4,a,terms(5)).
terms(5,0,variable(6)).
variable(6,"X").
body_literals(7,0,body_literal(8)).
body_literal(8,"pos",symbolic_atom(9)).
symbolic_atom(9,function(10)).
function(10,c,terms(11)).
terms(11,0,variable(12)).
variable(12,"X").
body_literals(7,1,body_literal(13)).
body_literal(13,"pos",symbolic_atom(14)).
symbolic_atom(14,function(15)).
function(15,d,terms(16)).
terms(16,0,function(17)).
function(17,e,terms(18)).
terms(18,0,variable(19)).
variable(19,"X").
