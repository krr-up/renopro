% auxiliary atoms
max_index(T,Index) :- Index = #max{ Pos : term_tuple(T,Pos,_); -1 }, term_tuple(T,_,_).
arity(F,N+1) :- function(F,_,term_tuple(T)), max_index(T,N).


% when a literal is not prev

% when atom of a literal is a predicate
add((term_tuple(T,N+1,constant(new_id(T)));
		 constant(new_id(T),t)))
:- literal(L,_,function(F)), function(F,Name,term_tuple(T)),
	 Name!=prev, max_index(T,N).

% when atom of a literal is a propositional constant
delete(literal(L,S,constant(F))) :- literal(L,S,constant(F)).
add((literal(L,S,function(F));
		 function(F,Name,term_tuple(new_id(F)));
		 term_tuple(new_id(F),0,constant(new_id(F)));
		 constant(new_id(F),t)))
:- literal(L,S,constant(F)), constant(F,Name).


prev_chain(L,function(F),O)
:- literal(L,_,function(F)), function(F,prev,term_tuple(T)), arity(F,1),
	 term_tuple(T,0,O).

prev_chain(L,function(F),O)
:- prev_chain(L,_,function(F)), function(F,prev,term_tuple(T)), arity(F,1),
	 term_tuple(T,0,O).

final_operand(L,O)
:- prev_chain(L,F,O), not prev_chain(L,O,_), O=(constant(_);function(_)).

first_prev(L,F)
:- prev_chain(L,F,O), not prev_chain(L,_,F).

num_prevs(L,N)
:- N = #count{ F : prev_chain(L,F,_) }, prev_chain(L,_,_).

% when final operand is a function
replace(function(F,N,A),function(F,Name,term_tuple(T)))
:- first_prev(L,function(F)), function(F,N,A), final_operand(L,function(O)),
	 function(O,Name,term_tuple(T)).

add(term_tuple(T,I+1,binary_operation(new_id(O)));
		binary_operation(new_id(O),"-",constant(new_id(O)),number(new_id(O)));
		constant(new_id(O),t);
		number(new_id(O),N))
:- final_operand(L,function(O)), function(O,_,term_tuple(T)), max_index(T,I),
	 num_prevs(L,N).

% when final operand is a constant
replace(function(F,N,A),function(F,Name,term_tuple(new_id(O))))
:- first_prev(L,function(F)), function(F,N,A), final_operand(L,constant(O)),
	 constant(O,Name).

add(term_tuple(new_id(O),0,binary_operation(new_id(O)));
		binary_operation(new_id(O),"-",constant(new_id(O)),number(new_id(O)));
		constant(new_id(O),t);
		number(new_id(O),N))
:- final_operand(L,constant(O)), num_prevs(L,N).
