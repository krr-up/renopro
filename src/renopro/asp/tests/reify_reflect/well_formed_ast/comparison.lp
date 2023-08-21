% reified fact representation of program:
% #program base.
% 1 < 2 != 3 > 4.

program("base",constants(0),statements(1)).
statements(1,0,rule(2)).
rule(2,literal(3),body_literals(13)).
literal(3,"pos",comparison(4)).
comparison(4,number(5),guards(6)).
number(5,1).
guards(6,0,guard(7)).
guard(7,"<",number(8)).
number(8,2).
guards(6,1,guard(9)).
guard(9,"!=",number(10)).
number(10,3).
guards(6,2,guard(11)).
guard(11,">", number(12)).
number(12,4).
