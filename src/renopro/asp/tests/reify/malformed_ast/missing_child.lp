% Malformed set of ast facts with a missing child fact

program("base",constants(0),statements(1)).
statements(1,0,rule(2)).
rule(2,literal(3),body_literals(7)).
% facts representing head literal a.
literal(3,"pos",symbolic_atom(4)).
symbolic_atom(4,function(5)).  % child fact function(5,_,_) is missing but required
