% reified fact representation of program:
% #program base.
% 1 < 2 != 3 > 4.

program("base",constants(0),statements(1)).
statements(1,0,rule(2)).
rule(2,literal(3),body_literals(10)).
literal(3,"pos",comparison(4)).
comparison(4,number(5),guards(6)).
number(5,1).
guards(6,0,"<",number(7)).
number(7,2).
guards(6,1,"!=",number(8)).
number(8,3).
guards(6,2,">", number(9)).
number(9,4).
