% auxiliary atoms
max_arg_index(TT,Idx)
    :- Idx = #max{ Pos : terms(TT,Pos,_); -1 }, function(_,_,terms(TT)).

max_lit_index(LT,Idx)
    :- Idx = #max{ P : body_literals(LT,P,_); -1 }, rule(_,_,body_literals(LT)).

max_statement_index(S,Idx)
    :- Idx = #max{ P : statements(S,P,_); -1 }, 
    program("base",_,statements(S)).


arity(F,N+1) :- function(F,_,terms(T)), max_arg_index(T,N).

% chain modal operators
modal(prev).
modal(next).

chain(M,A,function(F),O)
    :- symbolic_atom(A,function(F)), function(F,M,terms(T)), arity(F,1),
	 terms(T,0,O),
     modal(M).

chain(M,A,function(F),O)
    :- chain(M,A,_,function(F)), function(F,M,terms(T)), arity(F,1),
	 terms(T,0,O),
     modal(M).

% final operand must be a constant or function, as e.g. prev(1) is not a prev operation
final_operand(A,O)
    :- chain(M,A,F,O), not chain(M,A,O,_),
	 O=function(_),
     modal(M).

num(M,A,N) :- N = #count{ F : chain(M,A,F,_) }, chain(M,A,_,_).

first(M,A,F) :- chain(M,A,F,O), not chain(M,A,_,F).


% ----- Prop
ast_operation(
		add(variable(time_variable(term(T)),"T");
		    terms(T,N+1,variable(time_variable(term(T))))))
    :- symbolic_atom(A,function(F)), function(F,Name,terms(T)),
	    not modal(Name), max_arg_index(T,N).

% ----- Prev
ast_operation(
		delete(function(F,N,T1));
		add(function(F,Name,terms(T2));
            variable(time_variable(function(F)),"T");
            terms(T2,I+1,binary_operation(new_id(O)));
            binary_operation(new_id(O),"-",variable(time_variable(function(F))),number(new_id(O)));
            number(new_id(O),Num)))
:- first(prev,A,function(F)), function(F,N,T1), final_operand(A,function(O)),
	 function(O,Name,terms(T2)),  max_arg_index(T2,I), num(prev,A,Num).

% ----- Next
ast_operation(
		delete(function(F,N,T1));
		add(function(F,Name,terms(T2));
                 variable(time_variable(function(F)),"T");
				 terms(T2,I+1,binary_operation(new_id(O)));
				 binary_operation(new_id(O),"+",variable(time_variable(function(F))),number(new_id(O)));
				 number(new_id(O),Num)))
:- first(next,A,function(F)), function(F,N,T1), final_operand(A,function(O)),
	 function(O,Name,terms(T2)),  max_arg_index(T2,I), num(next,A,Num).

% ----- Time steps as new facts (Not working)
time(0..1).
% time(0..2).
ast_operation(
    add(number(time_number(T),T));
    add(terms(time_terms(T),0,number(time_number(T))));
    add(function(time_function(T),time,terms(time_terms(T))));
    add(symbolic_atom(time_symbol(T),function(time_function(T))));
    add(literal(time_literal(T),"pos",symbolic_atom(time_symbol(T))));
    add(rule(time_rule(T),literal(time_literal(T)),body_literals(time_body_literals(T)))))
:-time(T).

ast_operation(add(statements(S,X+T+1,rule(time_rule(T)))))
    :-time(T), program("base",_,statements(S)), max_statement_index(S,X).

% Add time(T) in all rules
ast_operation(
    add(body_literals(B,MAX+1,body_literal(time_body_literal_var(R))));
    add(body_literal(time_body_literal_var(R),"pos",symbolic_atom(time_symbol_var(R))));
    add(variable(time_variable(R),"T"));
    add(symbolic_atom(time_symbol_var(R),function(time_function_var(R))));
    add(function(time_function_var(R),time,terms(time_terms_var(R))));
    add(terms(time_terms_var(R),0,variable(time_variable(R))))
    )
:- rule(R,_,body_literals(B)), max_lit_index(B,MAX).



