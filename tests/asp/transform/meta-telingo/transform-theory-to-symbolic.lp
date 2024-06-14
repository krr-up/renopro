theory_opertor2symbol(
  "<",1,prev ; "<:",1,weak_prev ; "<?",1,eventually_before; "<*",1,always_before ;
	 ">",1,next ; ">:",1,weak_next ; ">?",1,eventually_after ; ">*",1,always_after ;
	 ">*",2,release ; ">?",2,until ; "<*",2,trigger ; "<?",2,since ;
	 "&",1,keyword_prefix ; "~",1,neg ; "&",2,and ; "|",2,or 
).
arity(F,A) :- theory_function(F,_,theory_terms(TS)), A = #count{ P: theory_terms(TS,P,_) }.

tel_theory_atom(A,location(theory_atom(A),Begin,End))
  :- theory_atom(A,function(F),theory_atom_elements(E),_), 
		 function(F,tel,terms(T)), not terms(T,_,_), location(theory_atom(A),Begin,End).

tel_theory_atom_terms(TTS,Loc)
  :- theory_atom(A,function(F),theory_atom_elements(E),_), tel_theory_atom(A,Loc), 
		 theory_atom_elements(E,_,theory_terms(TTS),_).

tel_root_term(Term,Loc) :- tel_theory_atom_terms(TTS,Loc), theory_terms(TTS,0,Term).
tel_root_term_id(Term,F) :- tel_root_term(Term,_), Term=(function(F);theory_function(F)).
tel_root_term_desc(Term,Term) :- tel_root_term(Term,Loc).
tel_root_term_desc(Term,Child') :- tel_root_term_desc(Term,Child), child(Child,Child').

% each tel theory atom element should should have a single function or
% theory function in it's term tuple, as we rewrite elements as
% conditional literals or body literals, and these may only have one literal in their
% head.
#show log("error","{}: The term tuple of tel theory atom elements must contain a single (temporal) formula.",Loc)
  : tel_theory_atom_terms(TTS,Loc), theory_terms(TTS,0,X), #false: X=theory_function(_);
    #false: X=function(_).

#show log("error","{}: The term tuple of tel theory atom elements must contain a single (temporal) formula.",Loc)
  : tel_theory_atom_terms(TTS,Loc), #count{ P: theory_terms(TTS,P,_) } != 1.

#show log("error", "{}: Theory sequences not allowed in tel theory atoms, found sequence type {}.",location(theory_sequence(TS),Begin,End),SequenceType)
  : tel_root_term_desc(Term,theory_sequence(TS)), theory_sequence(TS,SequenceType,_),
    location(theory_sequence(TS),Begin,End).

#include "string-to-const.lp".

% Due to the way theory parsing is implemented upstream, we know that
% theory function names can syntactically be operators if they are
% defined tel theory operators
ast(replace(theory_function(TF,Name,theory_terms(TTS)),function(TF,SName,terms(TTS));
					  theory_terms(TTS,P,TT),terms(TTS,P,TT))
)
  :- tel_root_term_desc(Term,theory_function(TF)), theory_function(TF,Name,theory_terms(TTS)), 
		 theory_terms(TTS,P,TT),
		 arity(TF,Arity), theory_opertor2symbol(Name,Arity,SName).

ast(replace(theory_function(TF,Name,theory_terms(TTS)),function(TF,Const,terms(TTS));
					  theory_terms(TTS,P,TT),terms(TTS,P,TT))
)
  :- tel_root_term_desc(Term,theory_function(TF)), theory_function(TF,Name,theory_terms(TTS)),
		 theory_terms(TTS,P,TT),
		 arity(TF,Arity), not theory_opertor2symbol(Name,Arity,_), Const=@string_to_const(Name).

% theory atoms in the head are interpreted as disjunctions, in the
% body they are interpreted as conditional literals. As such, in the
% body, they may only have one element.

#show log("error", "{}: tel theory atoms occurring in the body may only have one element.",Loc)
  : body_literals(BLS,_,body_literal(BL)),
    body_literal(BL,_,theory_atom(A)), tel_theory_atom(A,Loc), 
		theory_atom(A,_,theory_atom_elements(E),_),
		#count{ P: theory_atom_elements(E,P,_,_) } > 1.

% body case

tel_body_theory_atom(BL,Sign,A,F,LS)
  :- body_literal(BL,Sign,theory_atom(A)), tel_theory_atom(A,_), 
		 theory_atom(A,_,theory_atom_elements(E),_), 
		 theory_atom_elements(E,_,theory_terms(TS),literals(LS)),
     theory_terms(TS,0,RootTerm), tel_root_term_id(RootTerm,F).

ast(replace(body_literal(BL,Sign,theory_atom(A)),
				    conditional_literal((A,1),literal((A,2)),literals(LS)));
		add(literal((A,2),Sign,symbolic_atom((A,3)));
			  symbolic_atom((A,3),function(F)))
)
  :- tel_body_theory_atom(BL,Sign,A,F,LS),
     literals(LS,_,_).

ast(replace(body_literal(BL,Sign,theory_atom(A)),
					  body_literal((A,0),Sign,symbolic_atom((A,1))));
		add(symbolic_atom((A,1),function(F)))
)
  :- tel_body_theory_atom(BL,Sign,A,F,LS),
     not literals(LS,_,_).

% head case

ast(replace(theory_atom(A,Func,theory_atom_elements(E),Guard),
						disjunction((A,1),conditional_literals((A,2))));
		add(conditional_literals((A,2),P,conditional_literal((A,3,P)));
			  conditional_literal((A,3,P),literal((A,4,P)),literals(LS));
				literal((A,4,P),"pos",symbolic_atom((A,5)));
			  symbolic_atom((A,5),function(F)))
)
  :- rule(_,theory_atom(A),_), tel_theory_atom(A,_),
		 theory_atom(A,Func,theory_atom_elements(E),Guard),
     theory_atom_elements(E,P,theory_terms(TS),literals(LS)),
     theory_terms(TS,0,RootTerm), tel_root_term_id(RootTerm,F).
