% tagging of input ast elements
ast(string(Id,Val)) :- string(Id,Val).
ast(number(Id,Val)) :- number(Id,Val).
ast(variable(Id,Name)) :- variable(Id,Name).
ast(constant(Id,Name)) :- constant(Id, Name).
ast(function(Id,Name,Args)) :- function(Id,Name,Args).
ast(term_tuple(Id,Pos,Element)) :- term_tuple(Id,Pos,Element).
ast(binary_operation(Id,Operator,Left,Right)) :- binary_operation(Id,Operator,Left,Right).
ast(guard_tuple(Id,Pos,Comp_Op,Term))  :- guard_tuple(Id,Pos,Comp_Op,Term).
ast(comparison(Id,Guards)) :- comparison(Id,Guards).
ast(symbolic_atom(Id,Symbol)) :- symbolic_atom(Id,Symbol).
ast(literal(Id,Sign,Function)) :- literal(Id,Sign,Function).
ast(literal_tuple(Id,Pos,Element)) :- literal_tuple(Id,Pos,Element).
ast(rule(Id,Head,Body)) :- rule(Id,Head,Body).
ast(statement_tuple(Id,Pos,Element)) :- statement_tuple(Id,Pos,Element).
ast(constant_tuple(Id,Pos,Element)) :- constant_tuple(Id,Pos,Element).
ast(program(Name,Params,Statements)) :- program(Name,Params,Statements).


final(A) :- ast(A), not ast_operation(delete(A)), not ast_operation(replace(A,_)).
final(R) :- ast_operation(replace(_,R)).
final(N) :- ast_operation(add(N)).

#show final/1.
