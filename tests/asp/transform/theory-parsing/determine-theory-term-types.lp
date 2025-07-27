#program step(parse_unparsed_theory_terms).

#script (python)

from clingo.symbol import String, Number

def is_operator_name(name: String):
    name_str = name.string																	
    return Number(1) if (name_str and name_str[0] in "/!<=>+-*\\?&@|:;~^.") or (name_str == "not") else Number(0)

#end.

operator_table(TermType,(Name,1),(Prio,0))
  :- theory_term_definition(TermType,TheoryOpDefs),
		 member(theory_operator_definition(Name,Prio,"unary"),TheoryOpDefs).

operator_table(TermType,(Name,2),(Prio,1))
  :- theory_term_definition(TermType,TheoryOpDefs),
		 member(theory_operator_definition(Name,Prio,"binary_left"),TheoryOpDefs).

operator_table(TermType,(Name,2),(Prio,0))
  :- theory_term_definition(TermType,TheoryOpDefs),
		 member(theory_operator_definition(Name,Prio,"binary_right"),TheoryOpDefs).

atom_table((Name,Arity),AtomType,ElementTermType,TheoryGuardDef)
  :- theory_atom_definition(AtomType,Name,Arity,ElementTermType,TheoryGuardDef).

theory_atom_occurrence(R,A,body)
  :- node(R), R=Rule(_,B), body_literal(_,_,A), A=theory_atom(_,_,_).

theory_atom_occurrence(R,A,head)  
  :- node(R),  R=rule(A,(First,Rest)), A=theory_atom(_,_,_).

theory_atom_occurrence(R,A,directive)
  :- node(R),  R=rule(A,()), A=theory_atom(_,_,_).

theory_atom_occurrence(R,A,any) :- theory_atom_occurrence(R,A,_).

log("error", "{}: No theory atom definition of name '{}' and arity '{}'.",
		@loc2str(location(theory_atom(A),Begin,End)),AtomName,Arity)
  :- node(A), A=theory_atom(function(Arity,AtomName),_,_),
		 not atom_table((AtomName,Arity),_,_,_), L=location(A,_,_).

log("error", "{}: Theory atom found in unallowed context.",
					LA)
  :- theory_atom_occurrence(R,A,AtomType), 
		 A=theory_atom(F,_,_), F=function(Arity,AtomName,_),
		 not atom_table((AtomName,Arity),AtomType,_,_),
		 child(LR,LA), LR=location(R,_,_), LA=location(R,_,_).

descendant(theory_atom_elements(X),Y):- child(theory_atom_elements(X),Y).
descendant(theory_guard(X),Y):- child(theory_guard(X),Y).
descendant(X,Z) :- descendant(X,Y), child(Y,Z).

root_term_type(theory_atom_elements(E),ElementTermType)
  :- theory_atom(A,function(F),theory_atom_elements(E),_), 
		 function(Arity,AtomName,_) theory_atom_occurrence(A,AtomType),
		 atom_table((Name,Arity),AtomType,ElementTermType,_).

root_term_type(theory_guard(TG),GuardTermType)
  :- theory_atom(A,function(F),_,theory_guard(TG)), 
 		 function(Arity,F,AtomName,_), theory_atom_occurrence(A,Type),
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
:- theory_function(TF,Arity,N,theory_terms(TT)),
	 theory_term_type(TF,Type),
   @is_operator_name(N)=1, 
	 not operator_table(Type,(N,Arity),_),
   location(theory_function(TF),Begin,End).
