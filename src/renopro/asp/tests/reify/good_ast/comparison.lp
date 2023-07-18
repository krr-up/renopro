% reified fact representation of program:
% #program base.
% 1 < 2 != 3 > 4.

program("base",constant_tuple(0),statement_tuple(1)).
statement_tuple(1,0,rule(2)).
rule(2,literal(3),literal_tuple(10)).
literal(3,"pos",comparison(4)).
comparison(4,number(5),guard_tuple(6)).
number(5,1).
guard_tuple(6,0,"<",number(7)).
number(7,2).
guard_tuple(6,1,"!=",number(8)).
number(8,3).
guard_tuple(6,2,">", number(9)).
number(9,4).
