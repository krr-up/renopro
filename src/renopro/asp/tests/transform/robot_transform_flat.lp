% auxiliary atoms
max_index(T,M) :- M = #max{ P : term_tuple(T,P,_); -1 }, term_tuple(T,_,_).


% when outermost function is not prev
add_var(T,"T") :- literal(L,_,A), function(A,N,T), N!="prev".

% definition of add_var
new(term_tuple(T,M+1,variable(id(T)))) :- add_var(T,Var).
new(variable(id(T),Var)) :- add_var(T,Var).


% when outermost function is prev
remove_prev(A,T,F) :- literal(L,_,A), function(A,"prev",T),
								      term_tuple(T,0,function(F)), not term_tuple(T,1,_).

replace(function(A,"prev",T),function(A,N,T')) :- remove_prev(A,T,F), function(F,N,T').

% add binary operator...
new() :- remove_prev(A,T,F).

