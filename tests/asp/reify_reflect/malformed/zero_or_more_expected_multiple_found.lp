% malformed ast with multiple aggregate guards.

program(0,"base",constants(1),statements(2)).
statements(2,0,rule(3)).
rule(3,aggregate(4),body_literals(10)).
aggregate(4,guard(5),aggregate_elements(8),guard(9)).
guard(5,"<=",number(6)).
number(6,1).
guard(5,"<",number(7)).
number(7,3).
