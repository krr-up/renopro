% Malformed set of ast facts with a missing child fact

program("base",constant_tuple(0),statement_tuple(1)).
statement_tuple(1,0,rule(2)).
rule(2,literal(3),literal_tuple(7)).
% facts representing head literal a.
literal(3,"pos",atom(4)).
atom(4,function(5)).
% should only have one child like function(5,_,_)
function(5,a,term_tuple(6)).
function(5,b,term_tuple(7)).
