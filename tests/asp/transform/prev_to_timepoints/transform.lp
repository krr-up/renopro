#program step(prev_to_timepoints).
#const prev_to_timepoints=0.

prev_path(A,(F,()))
  :- node(A), A = symbolic_atom(F), F = function(1,"prev",_).

prev_path(A,(O,P))
  :- prev_path(A,P), P=(F,R), F = function(1,"prev",(O,())).

% final operand must be a constant or function, as e.g. prev(1) is not a prev operation
final_operand(A,F)
  :- prev_path(A,(F,R)), F = function(_,Name,_), Name != "prev".

num_prevs(A,N-1) :- prev_path(A,); N = #count{ P : prev_path(A,P) }.

% when a symbolic_atom is not prev, add time point constant as additional argument
ast(
		replace(A,
						symbolic_atom(function(Arity+1,Name,(function(0,"t",()),Terms)))))
  :- node(A), A = symbolic_atom(F), F = function(Arity,Name,Terms), Name!="prev".

% when a symbolic_atom is a prev, replace function symbol of
% symbolic_atom with final operand, appending appropriate time point
% as additional argument.
ast(replace(A,
						symbolic_atom(function(Arity,Name,
																	 (binary_operation("-",function(0,"t",()),
																										     number(Num)),
													Terms)))))
  :- final_operand(A,function(Arity,Name,Terms)), num_preds(A,Num).
