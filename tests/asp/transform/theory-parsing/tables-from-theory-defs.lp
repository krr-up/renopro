operator_table(TermType,(Name,1),(Prio,0))
:- theory_term_definitions(_,_,TermType,theory_operator_definitions(OD)),
	 theory_operator_definitions(OD,_,Name,Prio,unary).

operator_table(TermType,(Name,2),(Prio,1))
:- theory_term_definitions(_,_,TermType,theory_operator_definitions(OD)),
	 theory_operator_definitions(OD,_,Name,Prio,binary_left).

operator_table(TermType,(Name,2),(Prio,0))
:- theory_term_definitions(_,_,TermType,theory_operator_definitions(OD)),
	 theory_operator_definitions(OD,_,Name,Prio,binary_right).

atom_table((Name,Arity),AtomType,ElementTermType,theory_guard_definition(GD))
:- theory_atom_definitions(_,_,AtomType,Name,Arity,
													 ElementTermType,theory_guard_definition(GD)).
