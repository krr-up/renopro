program("base",constant_tuple(0),statement_tuple(1)).
statement_tuple(1,0,rule(2)).
rule(2,literal(3),literal_tuple(10)).
literal(3,"pos",comparison(4)).
comparison(4,number(5),guard_tuple(6)).
number(5,1).
