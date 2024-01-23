operator_arity((initially;finally),0).
operator_arity((prev;wprev;next;wnext;
							 always;always_before;eventually;eventually_before),1).
operator_arity((until;since;release;trigger),2).

func_arity(F,A) :- function(F,_,terms(TS)), A = #count{ P: terms(TS,P,_) }.
num_func_args(F,N) 
  :- function(F,_,terms(TS)), N = #count{ P: terms(TS,P,function(_)) }.

% we call a function an operator if it matches the name and arity of
% an operator definition and has only function operands.
operator(function(F))
  :- function(F,Name,terms(T)), operator_arity(Name,Arity), 
		 func_arity(F,Arity), num_func_args(F,Arity), 
		 Arity=#count{ F' : terms(T,_,function(F')) }.
root_operator(Func)
  :- operator(Func), symbolic_atom(_,Func).

%% #show.
%% #show (F,Name) : operator(funcion(F)), function(F,Name,_).

operand(function(F),Operand)
  :- operator(function(F)), function(F,_,terms(T)), terms(T,_,Operand).
% the left operand of binary temporal operators are considered unsafe,
% as they can be satisfied even if their left operand is not.
unsafe_operand(function(F),Operand)
  :- operator(function(F)), operator_arity(Name,2), function(F,Name,terms(T)), 
		 Name=(until;release), terms(T,0,Operand).
% wnext/wprev are unsafe, as they are satisfied in the final/initial time step
% even if their operand is not.
unsafe_operand(function(F),Operand)
  :- operator(function(F)), operator_arity(Name,1), function(F,Name,terms(T)), 
		 Name=(wprev;wnext), terms(T,0,Operand).
safe_operand(Func,Operand) 
  :- operand(Func,Operand), not unsafe_operand(Func,Operand).

desc_root_op2operand(Func,Operand) :- operand(Func,Operand), root_operator(Func).
desc_root_op2operand(Func,Operand') 
  :- desc_root_op2operand(Func,Operand), operand(Operand,Operand').
desc_root_op2safe_operand(Func,Operand) 
  :- safe_operand(Func,Operand), root_operator(Func).
desc_root_op2safe_operand(Func,Operand') 
  :- desc_root_op2safe_operand(Func,Operand), safe_operand(Operand,Operand').
root_op2leaf_operand(Func, Operand)
  :- desc_root_op2operand(Func,Operand), not operator(Operand).
root_op2safe_leaf_operand(Func, Operand)
  :- root_operator(Func), desc_root_op2safe_operand(Func,Operand), 
		 not operator(Operand).

root_op_sign(Func,Sign) 
  :- root_operator(Func), symbolic_atom(A,Func), 
		 literal(_,Sign,symbolic_atom(A)).
root_op_sign(Func,Sign) 
  :- root_operator(Func), symbolic_atom(A,Func), 
		 body_literal(_,Sign,symbolic_atom(A)).

%% #show.
%% #show (Name,Name') : root_op2leaf_operand(function(F),function(F')), function(F,Name,_), function(F',Name',_).

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
-head_root_op(Func) :- root_operator(Func), root_op_sign(Func,(("not";"not not"))).
head_root_op(Func) 
  :- possible_head_root_op(Func), not -head_root_op(Func).
body_root_op(Func) :- not head_root_op(Func), root_operator(Func).

% Map operators to atoms they could be conditioned on, and their sign

desc_stm((statements(STM),Pos),Child) :- statements(STM,Pos,Child).
desc_stm(Statement,Child') 
  :- desc_stm(Statement,Child), not root_operator(Child), child(Child,Child').
root_op2possible_extern_cond(Func,Sign,Func')
  :- desc_stm(Statement,Func), root_operator(Func), 
		 desc_stm(Statement,body_literal(BL)), body_literal(BL,Sign,symbolic_atom(S)),
     symbolic_atom(S,Func').
% If the operator is part of a condition, we also include the
% condition case where operator is in head of condition
root_op2possible_extern_cond(Func,Sign,Func')
  :- root_operator(Func), symbolic_atom(S,Func), literal(L,_,symbolic_atom(S)),
		 conditional_literal(_,literal(L),literals(LS)), literals(LS,_,literal(L')),
     literal(L',Sign,symbolic_atom(S')), symbolic_atom(S',Func').
% where operator is in body of condition (this covers body aggregates
% as well thankfully)
root_op2possible_extern_cond(Func,Sign,Func')
  :- root_operator(Func), symbolic_atom(S,Func), literal(L,_,symbolic_atom(S)),
		 literals(LS,_,literal(L)), literals(LS,_,literal(L')),
     literal(L',Sign,symbolic_atom(S')), symbolic_atom(S',Func').

% We condition the external statement for an operator on all
% non-operator literals in the body, and the safe leaf operands of
% operator literals, omitting the safe leaf operand of the operator in
% question if it is not positive.
-root_op2extern_cond(Func,Sign,Func')
  :- root_op2possible_extern_cond(Func,Sign,Func'), 
		 Func=Func', Sign=("not";"not not").
root_op2extern_cond(Func,Sign,Func')
  :- root_op2possible_extern_cond(Func,Sign,Func'),
		 not root_operator(Func').
root_op2extern_cond(Func,Sign,Func'')
  :- root_op2possible_extern_cond(Func,Sign,Func'),
		 not -root_op2extern_cond(Func,Sign,Func'), 
		 root_op2safe_leaf_operand(Func',Func'').

% create a base subprogram under which the external will be added.
ast_operation(
		add(program(externals,"base",constants(externals),statements(externals)))
).

% head of external condition in head operator case.
ast_operation(
		add(statements(externals,0,external(Operand));
				external(Operand,symbolic_atom(Operand),body_literals(Func),false);
				symbolic_atom(Operand,Operand)))
  :- head_root_op(Func), root_op2leaf_operand(Func,Operand).
% head of external condition in body operator case.
ast_operation(
		add(statements(externals,0,external(Func));
				external(Func,symbolic_atom(Func),body_literals(Func),false);
				symbolic_atom(Func,Func)))
  :- body_root_op(Func).
% body of external condition in both cases.
ast_operation(
		add(body_literals(Func,0,body_literal(Func'));
				body_literal(Func',Sign,symbolic_atom(Func'));
				symbolic_atom(Func',Func')))
  :- root_operator(Func), root_op2extern_cond(Func,Sign,Func').

