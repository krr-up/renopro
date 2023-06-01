new(literal_tuple(L,M+1,literal(1,function("bad",term_tuple(T)))))
:- rule(literal(0,function("good",term_tuple(T))),
				literal_tuple(L)),
	 M = #max{ P : literal_tuple(L,P,_) }.
											 
