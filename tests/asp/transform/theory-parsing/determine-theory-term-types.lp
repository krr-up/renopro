#script (python)

from clingo.symbol import String, Number

def is_operator_name(name: String):
    name_str = name.string																	
    return Number(1) if (name_str and name_str[0] in "/!<=>+-*\\?&@|:;~^.") or (name_str == "not") else Number(0)

#end.

func_arity(F,A) :- function(F,_,terms(TS)), A = #count{ P: terms(TS,P,_) }.

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

theory_atom_occurrence(A,body) :- body_literal(_,_,theory_atom(A)).

theory_atom_occurrence(A,head)  
  :- rule(_,theory_atom(A),body_literals(B)), body_literals(B,_,_).

theory_atom_occurrence(A,directive)  
  :- rule(_,theory_atom(A),body_literals(B)), not body_literals(B,_,_).

theory_atom_occurrence(A,any) :- theory_atom_occurrence(A,_).

log("error", "{}: No theory atom definition of name '{}' and arity '{}'.",
		location(theory_atom(A),Begin,End),AtomName,Arity)
  :- theory_atom(A,function(F),_,_), function(F,AtomName,_), 
  	 func_arity(F,Arity), not atom_table((AtomName,Arity),_,_,_),
		 location(theory_atom(A),Begin,End).

log("error", "{}: Theory atom found in unexpected context, allowed context: '{}'.",
					location(theory_atom(A),Begin,End),AtomType)
  :- theory_atom(A,function(F),_,_), function(F,AtomName,_), 
     func_arity(F,Arity), atom_table((AtomName,Arity),AtomType,_,_),
     not theory_atom_occurrence(A,AtomType), location(theory_atom(A),Begin,End).


descendant(theory_atom_elements(X),Y):- child(theory_atom_elements(X),Y).
descendant(theory_guard(X),Y):- child(theory_guard(X),Y).
descendant(X,Z) :- descendant(X,Y), child(Y,Z).

root_term_type(theory_atom_elements(E),ElementTermType)
  :- theory_atom(A,function(F),theory_atom_elements(E),_), 
		 function(F,AtomName,_), 
		 func_arity(F,Arity), theory_atom_occurrence(A,AtomType),
		 atom_table((Name,Arity),AtomType,ElementTermType,_).

root_term_type(theory_guard(TG),GuardTermType)
  :- theory_atom(A,function(F),_,theory_guard(TG)), 
 		 function(F,AtomName,_), 
		 func_arity(F,Arity), theory_atom_occurrence(A,Type),
		 theory_guard(TG,Opname,_),
		 atom_table((Name,Arity),Type,_,theory_guard_definition(GD)),
		 theory_guard_definition(GD,theory_operators(TOS),GuardTermType),
		 theory_operators(TOS,_,Opname).

% assign the theory term type assigned to the root node to it's descendants.

theory_term_type(TF,Type)
  :- root_term_type(X,Type), descendant(X,theory_function(TF)).

theory_term_type(TU,Type)
  :- root_term_type(X,Type), descendant(X,theory_unparsed_term(TU)).

% check that theory functions that represent operations agree with the
% grammar

undefined_operator(Type,N,Arity,location(theory_function(TF),Begin,End))
:- theory_function(TF,N,theory_terms(TT)),
	 theory_term_type(TF,Type),
   Arity = #count{ P: theory_terms(TT,P,T) },
   @is_operator_name(N)=1, 
	 not operator_table(Type,(N,Arity),_),
   location(theory_function(TF),Begin,End).
