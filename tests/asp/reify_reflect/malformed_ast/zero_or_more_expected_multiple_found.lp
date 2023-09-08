% malformed ast with multiple aggregate guards.

program("base",constants(0),statements(1)).
statements(1,0,rule(2)).
rule(2,aggregate(3),body_literals(9)).
aggregate(3,guard(4),agg_elements(7),guard(8)).
guard(4,"<=",number(5)).
number(5,1).
guard(4,"<",number(6)).
number(6,3).
