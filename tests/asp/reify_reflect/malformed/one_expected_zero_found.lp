% Malformed set of ast facts with a missing child fact

program(0,"base",constants(1),statements(2)).
statements(2,0,rule(3)).
rule(3,literal(4),body_literals(8)).
% facts representing head literal a.
literal(4,"pos",symbolic_atom(5)).
symbolic_atom(5,function(6)).  % child fact function(6,_,_) is missing but required
