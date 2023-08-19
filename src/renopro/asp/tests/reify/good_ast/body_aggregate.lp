% reified fact representation of program:
% #program base.
% :- #sum+ { 1,2: not a; 3: b } != 4.

program("base",constants(0),statements(1)).
statements(1,0,rule(2)).
rule(2,literal(3),body_literals(5)).
literal(3,"pos",boolean_constant(4)).
boolean_constant(4,false).
body_literals(5,0,literal(6)).
literal(6,"pos",body_aggregate(7)).
body_aggregate(7,guard(8),"#sum+",body_agg_elements(10),guard(31)).
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
body_agg_elements(10,1,terms(19),literals(26)).
terms(24,0,number(25)).
number(25,3).
literals(26,0,literal(27)).
literal(27,"pos",symbolic_atom(28)).
symbolic_atom(28,function(29)).
function(29,b,terms(30)).
