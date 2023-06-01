rule(literal(0,function("sad",term_tuple(0)))
		 ,literal_tuple(0)).

literal_tuple(0,0,literal(0,function("prev",term_tuple(2)))).
literal_tuple(0,1,literal(1,function("cheer_up",term_tuple(5)))).


term_tuple(0,0,function("robot",term_tuple(1))).
term_tuple(1,0,variable("X")).

term_tuple(2,0,function("sad",term_tuple(3))).
term_tuple(3,0,function("robot",term_tuple(4))).
term_tuple(4,0,variable("X")).

term_tuple(5,0,function("robot",term_tuple(6))).
term_tuple(6,0,variable("X")).
