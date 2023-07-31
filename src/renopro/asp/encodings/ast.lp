% tag ast facts.
ast(string(Id,Val)) :- string(Id,Val).
ast(number(Id,Val)) :- number(Id,Val).
ast(variable(Id,Name)) :- variable(Id,Name).
ast(constant(Id,Name)) :- constant(Id, Name).
ast(function(Id,Name,Args)) :- function(Id,Name,Args).
ast(terms(Id,Pos,Element)) :- terms(Id,Pos,Element).
ast(binary_operation(Id,Operator,Left,Right)) :- binary_operation(Id,Operator,Left,Right).
ast(interval(Id,Left,Right)) :- interval(Id,Left,Right).
ast(guards(Id,Pos,Comp_Op,Term))  :- guards(Id,Pos,Comp_Op,Term).
ast(comparison(Id,Guards)) :- comparison(Id,Guards).
ast(boolean_constant(Id,Bool)) :- boolean_constant(Id,Bool).
ast(symbolic_atom(Id,Symbol)) :- symbolic_atom(Id,Symbol).
ast(literal(Id,Sign,Function)) :- literal(Id,Sign,Function).
ast(body_literals(Id,Pos,Element)) :- body_literals(Id,Pos,Element).
ast(conditional_literal(Id,Literal,Condition)) :- conditional_literal(Id,Literal,Condition).
ast(rule(Id,Head,Body)) :- rule(Id,Head,Body).
ast(statements(Id,Pos,Element)) :- statements(Id,Pos,Element).
ast(constants(Id,Pos,Element)) :- constants(Id,Pos,Element).
ast(program(Name,Params,Statements)) :- program(Name,Params,Statements).
