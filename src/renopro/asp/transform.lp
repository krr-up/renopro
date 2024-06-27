%% #include "ast.lp".
%% #include "defined.lp".
%% #include "replace_id.lp".
%% #include "ast_fact2id.lp".

#program always.

% add operation should also generate an appropriate child/2 fact to link the new fact
% to it's children
ast(add(child(@ast_fact2id(Fact),@ast_fact2children(Fact)))) :- ast(add(Fact)), #false: Fact=child(_,_).

ast(delete(A);add(B)) :- ast(replace(A,B)).

ast(_replace_id(@ast_fact2id(A),@ast_fact2id(B)))  :- ast(replace(A,B)).

transform_step :- ast(add(X);delete(X)).

% delete stops inertia from old ast, and even deletes asts that would have been added
transformed(A) :- transform_step, ast((fact(A);add(A))), not ast(delete(A)).

% Only preserve AST facts which are reachable from the root
% AST facts. These are the facts over signature program/4.
reachable(program(P)) :- transformed(program(P,Name,Params,Stms)).
reachable(ChildID) :- reachable(ParentID), transformed(child(ParentID,ChildID)).
final(child(ParentID,ChildID)) :- reachable(ParentID), transformed(child(ParentID,ChildID)).
final(Fact) :- transformed(Fact), ID=@ast_fact2id(Fact), reachable(ID).
	
#program final.
not transform_step.
