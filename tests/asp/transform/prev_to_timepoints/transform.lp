% auxiliary atoms
max_arg_index(TT,Idx)
:- Idx = #max{ Pos : terms(TT,Pos,_); -1 }, function(_,_,terms(TT)).

arity(F,N+1) :- function(F,_,terms(T)), max_arg_index(T,N).

prev_chain(A,function(F),O)
:- symbolic_atom(A,function(F)), function(F,prev,terms(T)), arity(F,1),
	 terms(T,0,O).

prev_chain(A,function(F),O)
:- prev_chain(A,_,function(F)), function(F,prev,terms(T)), arity(F,1),
	 terms(T,0,O).

% final operand must be a constant or function, as e.g. prev(1) is not a prev operation
final_operand(A,O)
:- prev_chain(A,F,O), not prev_chain(A,O,_),
	 O=function(_).

num_prevs(A,N) :- N = #count{ F : prev_chain(A,F,_) }, prev_chain(A,_,_).

first_prev(A,F) :- prev_chain(A,F,O), not prev_chain(A,_,F).


% when an symbolic_atom is not prev.

% add time point constant as additional argument
ast_operation(
		add(terms(T,N+1,function(new_id(T)));
				function(new_id(T),t,terms(new_id(T)))))
:- symbolic_atom(A,function(F)), function(F,Name,terms(T)),
	 Name!=prev, max_arg_index(T,N).

% when an symbolic_atom is a prev.

% replace function symbol of symbolic_atom with final operand, appending
% appropriate time point as additional argument.
ast_operation(
		delete(function(F,N,T1));
		add(function(F,Name,terms(T2));
				 terms(T2,I+1,binary_operation(new_id(O)));
				 binary_operation(new_id(O),"-",function(new_id(O)),number(new_id(O)));
				 function(new_id(O),t,terms(new_id(O)));
				 number(new_id(O),Num)))
:- first_prev(A,function(F)), function(F,N,T1), final_operand(A,function(O)),
	 function(O,Name,terms(T2)),  max_arg_index(T2,I), num_prevs(A,Num).
