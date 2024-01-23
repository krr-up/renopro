% transform initial, dynamic, always and final subprograms


% move all the temporal subprograms into the base subprogram
ast_operation(
		delete(program(P,X,constants(CS),statements(STS)));
		add(program(P,"base",constants(CS),statements(STS))))
  :- program(P,X,constants(CS),statements(STS)), not constants(CS,_,_),
		 X=("initial";"dynamic";"always";"final").

% tag bodies to be modified with temporal subprogram they appear in
body_program(BLS,X)
  :- program(_,X,constants(CS),statements(STS)), not constants(CS,_,_), statements(STS,_,Statement),
		 child(Statement,body_literals(BLS)), X=("initial";"dynamic";"always";"final").

ast_operation(
		add(body_literals(BLS,-1,body_literal(BLS));
			 body_literal(BLS,"pos",symbolic_atom(BLS));
			 symbolic_atom(BLS,function(BLS));
			 function(BLS,initially,terms(BLS)))
)
  :- body_program(BLS,"initial").

ast_operation(
		add(body_literals(BLS,-1,body_literal(BLS));
			 body_literal(BLS,"pos",symbolic_atom(BLS));
			 symbolic_atom(BLS,function(BLS));
			 function(BLS,finally,terms(BLS)))
)
  :- body_program(BLS,"final").

ast_operation(
		add(body_literals(BLS,-1,body_literal(BLS));
			 body_literal(BLS,"not",symbolic_atom(BLS));
			 symbolic_atom(BLS,function(BLS));
			 function(BLS,initially,terms(BLS)))
)
  :- body_program(BLS,"dynamic").

% since there will be a downstream meta-encoding, construct the
% child relation for the new ast elements.
ast_operation(
		add(child(body_literals(BLS),body_literal(BLS));
			  child(body_literal(BLS),symbolic_atom(BLS));
			  child(symbolic_atom(BLS),function(BLS)))
)
  :- body_program(BLS,_).


