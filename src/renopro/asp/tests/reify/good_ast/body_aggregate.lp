% reified fact representation of program:
% #program base.
% :- #sum+ { 1,2: not a; 3: b } != 4.

program("base",constants(0),statements(1)).
statements(1,0,rule(2)).
rule(2,literal(3),body_literals(5)).
literal(3,"pos",boolean_constant(4)).
boolean_constant(4,false).
body_literals(5,0,body_literal(6)).
body_literal(6,"pos",body_aggregate(7)).
body_aggregate(7,guard(8),"#sum+",body_agg_elements(10),guard(26)).
guard(8,"!=",number(9)).
number(9,4).
body_agg_elements(10,0,terms(11),literals(14)).
terms(11,0,number(12)).
number(12,1).
terms(11,1,number(13)).
number(13,2).
literals(14,0,literal(15)).
literal(15,"not",symbolic_atom(16)).
symbolic_atom(16,function(17)).
function(17,a,terms(18)).
body_agg_elements(10,1,terms(19),literals(21)).
terms(19,0,number(20)).
number(20,3).
literals(21,0,literal(22)).
literal(22,"pos",symbolic_atom(23)).
symbolic_atom(23,function(24)).
function(24,b,terms(25)).
