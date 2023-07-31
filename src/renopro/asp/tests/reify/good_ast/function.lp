% reified fact representation of program:
% #program base.
% next(move(a)).

program("base",constants(0),statements(1)).
statements(1,0,rule(2)).
rule(2,literal(3),body_literals(11)).
% facts representing head literal rel(2,1).
literal(3,"pos",symbolic_atom(4)).
symbolic_atom(4,function(5)).
function(5,next,terms(6)).
terms(6,0,function(7)).
function(7,move,terms(8)).
terms(8,0,function(9)).
function(9,a,terms(10)).
