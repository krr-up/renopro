% auxiliary atoms
max_index(T,Index) :- Index = #max{ Pos : terms(T,Pos,_); -1 }, terms(T,_,_).
arity(F,N+1) :- function(F,_,terms(T)), max_index(T,N).

% collect modal operators and delete them from the AST

modal(OpName,Arity,Argument)
:- rule(_,literal(L),body_literals(Lt)), not body_literals(Lt,_,_)
	 literal(L,"pos",function(F)), function(F,modal,terms(T)),
	 terms(T,0,constant(C)), constant(C, OpName),
	 terms(T,1,number(N1)), number(N1, Arity),
	 terms(T,2,number(N2)), number(N2, Argument).

delete(rule(R,literal(L),body_literals(Lt)))
:- rule(R,literal(L),body_literals(Lt)), not body_literals(Lt,_,_)
	 literal(L,"pos",function(F)), function(F,modal,terms(T)),
	 terms(T,0,constant(C)), constant(C, OpName),
	 terms(T,1,number(N1)), number(N1, Arity),
	 terms(T,2,number(N2)), number(N2, Argument).

operator(OpName) :- modal(OpName,_).

op_tree()
:- modal(Op,Arity,Argument), literal(L,_,function(F)),
	 function(F,Op,terms(T)), arity(F,Arity), terms(T,Argument-1,O).
