reach(program(P,Name,Params,Stms)) :- transformed(program(P,Name,Params,Stms)).

reach(child(ParentId,ChildId);ChildFact) 
  :- reach(ParentFact), transformed_ast_fact2id(ParentFact,ParentId),
		 transformed(child(ParentId,ChildId)), transformed_ast_fact2id(ChildFact,ChildId).

final(R) :- reach(R).

%% #show log("warning", "{}",reach(F)) : reach(F).
%% #show log("warning", "{}",transformed(F)) : transformed(F), transformed_ast_fact2id(F,Id).
%% #show log("warning", "{}",final(F)) : final(F).
%% #show log("warning", "{}",ast(F)) : ast(F).
%% #show log("warning", "{}",ast_operation(F)) : ast_operation(F).
