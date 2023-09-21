% reified fact representation of program:
% #program base.
% :- #sum+ { 1,2: not a; 3: b } != 4.

program(0,"base",constants(1),statements(2)).
statements(2,0,rule(3)).
rule(3,literal(4),body_literals(6)).
literal(4,"pos",boolean_constant(5)).
boolean_constant(5,false).
body_literals(6,0,body_literal(7)).
body_literal(7,"pos",body_aggregate(8)).
body_aggregate(8,guard(9),"#sum+",body_aggregate_elements(11),guard(27)).
guard(9,"!=",number(10)).
number(10,4).
body_aggregate_elements(11,0,terms(12),literals(15)).
terms(12,0,number(13)).
number(13,1).
terms(12,1,number(14)).
number(14,2).
literals(15,0,literal(16)).
literal(16,"not",symbolic_atom(17)).
symbolic_atom(17,function(18)).
function(18,a,terms(19)).
body_aggregate_elements(11,1,terms(20),literals(22)).
terms(20,0,number(21)).
number(21,3).
literals(22,0,literal(23)).
literal(23,"pos",symbolic_atom(24)).
symbolic_atom(24,function(25)).
function(25,b,terms(26)).
