operator_arity((initial;final),0).
operator_arity((prev;weak_prev;next;weak_next;neg;
							 always_after;always_before;eventually_after;eventually_before),1).
operator_arity((until;since;release;trigger;and;or),2).
% Naturally, default negation is unsafe.
% weak_next/weak_prev are unsafe, as they are satisfied in the final/initial time step
% even if their operand is not.
arg((weak_prev;weak_next),0,unsafe).
% the left operand of binary temporal operators are considered unsafe,
% as they can be satisfied even if their left operand is not.
arg((until;since;release;trigger),0,unsafe).
% both operands of or are unsafe.
arg(or,(0;1),unsafe).
arg(Opname,Arg,safe)
  :- operator_arity(Opname,Arity), Arg=0..Arity-1, not arg(Opname,Arg,unsafe).

func_arity(F,A) :- function(F,_,terms(TS)), A = #count{ P: terms(TS,P,_) }.

% we call a function an operator if it matches the name and arity of
% an operator definition and has only function operands.
operator(function(F))
  :- function(F,Name,terms(T)), operator_arity(Name,Arity), 
		 func_arity(F,Arity).

operand(function(F),function(F'),Safety)
  :- operator(function(F)), function(F,Name,terms(T)), arg(Name,P,Safety),
		 terms(T,P,function(F')).

% an operator must have the same amount of function operands as it's arity
% if not, raise an error.
log("error", "{}: All arguments of operator '{}/{}' must be functions.", 
location(function(F),Begin,End), Name, Arity)
  :- operator(function(F)), function(F,Name,terms(T)), 
     N=#count{ Operand: operand(function(F),Operand,_) },
		 N < Arity, func_arity(F,Arity), location(function(F),Begin,End).

root_operator(Func)
  :- operator(Func), symbolic_atom(_,Func).

desc_root_op(Func,Child) :- root_operator(Func), child(Func,Child).
desc_root_op(Func,Child') :- desc_root_op(Func,Child), child(Child,Child').
has_var(Func) :- desc_root_op(Func,variable(V)).


operand_from_root(Root,Operand,Safety)
  :- root_operator(Root), operand(Root,Operand,Safety).
operand_from_root(Root,Operand',Safety)
  :- operand_from_root(Root,Operand,Safety), operand(Operand,Operand',safe).
operand_from_root(Root,Operand',unsafe)
  :- operand_from_root(Root,Operand,Safety), operand(Operand,Operand',unsafe).
leaf_operand_from_root(Root,Operand,Safety)
  :- operand_from_root(Root,Operand,Safety), not operator(Operand).

atom_sign(A,Sign)
  :- literal(_,Sign,symbolic_atom(A)), symbolic_atom(A,_).
atom_sign(A,Sign)
  :- body_literal(_,Sign,symbolic_atom(A)), symbolic_atom(A,_).

% determine which root operators occur in the head vs the body
possible_head_root_op(Func)
  :- root_operator(Func), symbolic_atom(S,Func), literal(L,_,symbolic_atom(S)),
     not literals(_,_,literal(L)).
% heads of conditional literals in the body of a rule are not
% cosidered head operators.
-head_root_op(Func)
  :- root_operator(Func), symbolic_atom(S,Func), literal(L,_,symbolic_atom(S)),
     conditional_literal(CL,literal(L),_), 
		 child((body_literals(_);aggregate_elements(_)),conditional_literal(CL)).
% Head atoms that are negated or double negated are just body atoms in disguise
%
% todo: show that we can shift them to the body and add an additional
% negation in the QTHT case as well, as clingo does this during
% grounding, so we need to actually prove that it is correct.
-head_root_op(Func) :- root_operator(Func), func_sign(Func,(("not";"not not"))).
head_root_op(Func) 
  :- possible_head_root_op(Func), not -head_root_op(Func).
body_root_op(Func) :- not head_root_op(Func), root_operator(Func).

% positive atoms in the body that are not operators are safe atomic (sub)formulae
safe_atomic_subformula(Func,Func) 
  :- symbolic_atom(A,Func), not root_operator(Func), atom_sign(A,"pos").
% the safe leaf operands of positive operators are safe atomic subformulae
safe_atomic_subformula(Func,Func')
  :- symbolic_atom(A,Func), root_operator(Func), 
		 atom_sign(A,"pos"), leaf_operand_from_root(Func,Func',safe).

% Map operators to atoms they are to be conditioned on.

desc_stm((statements(STM),Pos),Child) :- statements(STM,Pos,Child).
desc_stm(Statement,Child') 
  :- desc_stm(Statement,Child), not root_operator(Child), child(Child,Child').

root_op2extern_cond(Func,Func'')
  :- desc_stm(Statement,Func), root_operator(Func), 
		 desc_stm(Statement,body_literal(BL)), body_literal(BL,"pos",symbolic_atom(S)),
     symbolic_atom(S,Func'), safe_atomic_subformula(Func',Func'').

% todo: throw error of there is a conditional in the program, as this
% is not supported, since we don't have a proof of correctness.
%
%
%% If the operator is part of a condition (as head or as body), we also
%% include the condition in the set of external conditions. Probably
%% leave this out of the thesis as we have no proof of correctness for
%% conditions.
root_op2extern_cond(Func,Func'')
  :- root_operator(Func), symbolic_atom(S,Func), literal(L,_,symbolic_atom(S)),
		 conditional_literal(_,literal(L),literals(LS)), literals(LS,_,literal(L')),
     literal(L',"pos",symbolic_atom(S')), symbolic_atom(S',Func'), 
		 safe_atomic_subformula(Func',Func'').

root_op2extern_cond(Func,Func'')
  :- root_operator(Func), symbolic_atom(S,Func), literal(L,_,symbolic_atom(S)),
		 literals(LS,_,literal(L)), literals(LS,_,literal(L')),
     literal(L',Sign,symbolic_atom(S')), symbolic_atom(S',Func'), 
		 safe_atomic_subformula(Func',Func'').


% create a base subprogram under which the external will be added.
ast(add(program(externals,"base",constants(externals),statements(externals)))).

% head of external condition in head operator case.
%
% todo: check proof that it is enough to find leaf operands that are not in
% the scope of negation.
ast(add(statements(externals,pos(Func,Operand),external(Operand));
				external(Operand,symbolic_atom(Operand),body_literals(Func),false);
				symbolic_atom(Operand,Operand)))
  :- head_root_op(Func), leaf_operand_from_root(Func,Operand,_).

% head of external condition in body operator case.
ast(add(statements(externals,pos(Func),external(Func));
				external(Func,symbolic_atom(Func),body_literals(Func),false);
				symbolic_atom(Func,Func)))
  :- body_root_op(Func).

% body of external condition 
% when we have a propositional operator in the body, we leave the condition empty
ast(add(body_literals(Func,pos(Func'),body_literal(Func'));
				body_literal(Func',"pos",symbolic_atom(Func'));
				symbolic_atom(Func',Func')))
  :- root_operator(Func), root_op2extern_cond(Func,Func'), 
		 #false : not has_var(Func), body_root_op(Func).

