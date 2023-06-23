% auxiliary atoms
max_arg_index(TT,Idx)
:- Idx = #max{ Pos : term_tuple(TT,Pos,_); -1 }, function(_,_,term_tuple(TT)).

arity(F,N+1) :- function(F,_,term_tuple(T)), max_arg_index(T,N).

prev_chain(A,function(F),O)
:- atom(A,function(F)), function(F,prev,term_tuple(T)), arity(F,1),
	 term_tuple(T,0,O).

prev_chain(A,function(F),O)
:- prev_chain(A,_,function(F)), function(F,prev,term_tuple(T)), arity(F,1),
	 term_tuple(T,0,O).

% final operand must be a constant or function, as e.g. prev(1) is not a prev operation
final_operand(A,O)
:- prev_chain(A,F,O), not prev_chain(A,O,_),
	 O=function(_).

num_prevs(A,N) :- N = #count{ F : prev_chain(A,F,_) }, prev_chain(A,_,_).

first_prev(A,F) :- prev_chain(A,F,O), not prev_chain(A,_,F).


% when an atom is not prev.

% add time point constant as additional argument
add(term_tuple(T,N+1,function(new_id(T)));
		 function(new_id(T),t,term_tuple(new_id(T))))
:- atom(A,function(F)), function(F,Name,term_tuple(T)),
	 Name!=prev, max_arg_index(T,N).

% when an atom is a prev.

% replace function symbol of atom with final operand, appending
% appropriate time point as additional argument.
replace(function(F,N,T1),function(F,Name,term_tuple(T2)))
:- first_prev(A,function(F)), function(F,N,T1), final_operand(A,function(O)),
	 function(O,Name,term_tuple(T2)).

add((term_tuple(T,I+1,binary_operation(new_id(O)));
		binary_operation(new_id(O),"-",function(new_id(O)),number(new_id(O)));
		function(new_id(O),t,term_tuple(new_id(O)));
		number(new_id(O),N)))
:- final_operand(A,function(O)), function(O,_,term_tuple(T)), max_arg_index(T,I),
	 num_prevs(A,N).

