% reified fact representation of program:
% #program base.
% #sum { 1,2: a: not b; 3: c } = 4.

program("base",constants(0),statements(1)).
statements(1,0,rule(2)).
rule(2,head_aggregate(3),body_literals(29)).
head_aggregate(3,guard(4),"#sum",head_agg_elements(6),guard(28)).
guard(4,"=",number(5)).
number(5,4).
head_agg_elements(6,0,terms(7),conditional_literal(10)).
terms(7,0,number(8)).
number(8,1).
terms(7,1,number(9)).
number(9,2).
conditional_literal(10,literal(11),literals(15)).
literal(11,"pos",symbolic_atom(12)).
symbolic_atom(12,function(13)).
function(13,a,terms(14)).
literals(15,0,literal(16)).
literal(16,"not",symbolic_atom(17)).
symbolic_atom(17,function(18)).
function(18,b,terms(19)).
head_agg_elements(6,1,terms(20),conditional_literal(22)).
terms(20,0,number(21)).
number(21,3).
conditional_literal(22,literal(23),literals(27)).
literal(23,"pos",symbolic_atom(24)).
symbolic_atom(24,function(25)).
function(25,c,terms(26)).
