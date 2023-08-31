#const horizon = 1.


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
% modal(initially).

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

% ----- Time steps as new fact with interval 0..horizon
ast_operation(
		add(number(time_number_0,0));
		add(number(time_number_h,horizon));
		add(interval(time_interval,number(time_number_0),number(time_number_h)));
    add(terms(time_terms,0,interval(time_interval)));
    add(function(time_function,time,terms(time_terms)));
    add(symbolic_atom(time_symbol,function(time_function)));
    add(literal(time_literal,"pos",symbolic_atom(time_symbol)));
    add(rule(time_rule,literal(time_literal),body_literals(time_body_literals)))).

ast_operation(add(statements(S,X+1,rule(time_rule))))
    :- program("base",_,statements(S)), max_statement_index(S,X).

% Add time(T) in all rules
ast_operation(
    add(variable(time_variable(R),"T"));
    add(terms(time_terms_var(R),0,variable(time_variable(R))));
    add(function(time_function_var(R),time,terms(time_terms_var(R))));
    add(symbolic_atom(time_symbol_var(R),function(time_function_var(R))));
    add(body_literal(time_body_literal_var(R),"pos",symbolic_atom(time_symbol_var(R))));
    add(body_literals(B,MAX+1,body_literal(time_body_literal_var(R))))
    )
:- rule(R,_,body_literals(B)), max_lit_index(B,MAX).

% Add initially(0).
ast_operation(
    add(number(init_number,0));
    add(terms(init_terms,0,number(init_number)));
    add(function(init_function,initially,terms(init_terms)));
    add(symbolic_atom(init_symbol,function(init_function)));
    add(literal(init_literal,"pos",symbolic_atom(init_symbol)));
    add(rule(init_rule,literal(init_literal),body_literals(init_body_literals)))).

ast_operation(add(statements(S,X+2,rule(init_rule))))
    :- program("base",_,statements(S)), max_statement_index(S,X).



% Add finally(horizon).
ast_operation(
    add(number(final_number,horizon));
    add(terms(final_terms,0,number(final_number)));
    add(function(final_function,finally,terms(final_terms)));
    add(symbolic_atom(final_symbol,function(final_function)));
    add(literal(final_literal,"pos",symbolic_atom(final_symbol)));
    add(rule(final_rule,literal(final_literal),body_literals(final_body_literals)))).

ast_operation(add(statements(S,X+3,rule(final_rule))))
    :- program("base",_,statements(S)), max_statement_index(S,X).


