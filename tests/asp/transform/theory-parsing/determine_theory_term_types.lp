func_arity(F,A) :- function(F,_,terms(TS)), A = #count{ P: terms(TS,P,_) }.

theory_atom_type(A,body) :- body_literals(_,_,theory_atom(A)).

theory_atom_type(A,head)  
:- rule(_,theory_atom(A), body_literals(B)), body_literals(B,_,_).

theory_atom_type(A,directive)  
:- rule(_,theory_atom(A), body_literals(B)), not body_literals(B,_,_).

#show log("error", location(A,Begin,End), ": No theory atom definition of name '",
					AtomName, "' and arity '", Arity, "'.")
: theory_atom(A,function(F),_,_), function(F,AtomName,_), 
	func_arity(F,Arity), not atom_table((Name,Arity),_,_,_),
  location(A,Begin,End).

#show log("error", location(A,Begin,End), ": Theory atom expected to occur in context '",
					AtomType, "', but found in context '", OccuranceType ,"'.")
: theory_atom(A,function(F),_,_), function(F,AtomName,_), 
  func_arity(F,Arity), atom_table((AtomName,Arity),AtomType,_,_),
  theory_atom_type(A,OccuranceType), OccuranceType != AtomType.


% descendant is the transitive closure of child, but only
% defined starting from a theory atom node downward
descendant(theory_atom(X),Y):- child(theory_atom(X),Y).
descendant(X,Z) :- descendant(X,Y), child(Y,Z).

root(theory_atom_elements(E),ElementTermType)
:- theory_atom(A,function(F),theory_atom_elements(E),_), 
	 function(F,AtomName,_), 
	 func_arity(F,Arity), theory_atom_type(A,AtomType),
	 atom_table((Name,Arity),AtomType,ElementTermType,_).

root(theory_guard(E),GuardTermType)
:- theory_atom(A,function(F),_,theory_guard(TG)), 
	 function(F,AtomName,_), 
	 func_arity(F,Arity), theory_atom_type(A,Type),
	 theory_guard(TG,Opname,_),
	 atom_table((Name,Arity),Type,_,theory_guard_definition(GD)),
	 theory_guard_definition(GD,theory_operators(TOS),GuardTermType),
   theory_operators(TOS,_,Opname).

#show log("error","")
: theory_atom(A,function(F),_,theory_guard(TG)), 
  function(F,AtomName,_), 
	func_arity(F,Arity), theory_atom_type(A,Type),
	theory_guard(TG,Opname,_),
	atom_table((Name,Arity),Type,_,theory_guard_definition(GD)),
	#false : theory_guard_definition(GD,theory_operators(TOS),GuardTermType),
 	         theory_operators(TOS,_,Opname).

% assign the theory term type assigned to the root node to it's descendants.

theory_term_type(TF,Type)
:- root(X,Type), descendant(X,theory_function(TF)).

theory_term_type(TU,Type)
:- root(X,Type), descendant(X,theory_unparsed_term(TU)).

% need to write some error handing, in case the atom occurs in a place
% it isn't supposed to, or the theory guard contains an element it's
% not supposed - right now no error message is raised, just no
% transformation occurs
