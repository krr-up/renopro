% transform initial, dynamic, always and final subprograms

#program step(meta_telingo_subprograms).
#const meta_telingo_subprograms=0.

temporal_program(P) 
  :- program(P,X,constants(CS),_), not constants(CS,_,_),
		 X=("initial";"dynamic";"always";"final").

% move all the temporal subprograms into the base subprogram
ast(replace(program(P,X,constants(CS),statements(STS)),
						program(P,"base",constants(CS),statements(STS))))
  :- program(P,X,constants(CS),statements(STS)), temporal_program(P).

% tag bodies to be modified with temporal subprogram they appear in
body_program(BLS,X)
  :- program(_,X,constants(CS),statements(STS)), temporal_program(P), 
		 statements(STS,_,Statement), child(Statement,body_literals(BLS)).

ast(
		add(body_literals(BLS,-1,body_literal(BLS));
			 body_literal(BLS,"pos",symbolic_atom(BLS));
			 symbolic_atom(BLS,function(BLS));
			 function(BLS,initial,terms(BLS)))
)
  :- body_program(BLS,"initial").

ast(
		add(body_literals(BLS,-1,body_literal(BLS));
			 body_literal(BLS,"pos",symbolic_atom(BLS));
			 symbolic_atom(BLS,function(BLS));
			 function(BLS,final,terms(BLS)))
)
  :- body_program(BLS,"final").

ast(
		add(body_literals(BLS,-1,body_literal(BLS));
			 body_literal(BLS,"not",symbolic_atom(BLS));
			 symbolic_atom(BLS,function(BLS));
			 function(BLS,initial,terms(BLS)))
)
  :- body_program(BLS,"dynamic").


