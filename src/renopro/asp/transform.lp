%% #include "wrap_ast.lp".
%% #include "unwrap_ast.lp".
%% #include "defined.lp".
%% #include "reflace_id.lp".
%% #include "scripts.lp".
#program always.

% add operation should also generate an appropriate child/2 fact to link the new fact
% to it's children
ast(add(child(@ast_fact2id(Fact),@ast_fact2children_ids(Fact)))) :- ast(add(Fact)), #false: Fact=child(_,_).

ast(delete(A);add(B)) :- ast(replace(A,B)).

ast(_replace_id(@ast_fact2id(A),@ast_fact2id(B)))  :- ast(replace(A,B)).

% delete stops inertia from old ast, and even deletes asts that would have been added
_transformed(A) :- ast((fact(A);add(A))), not ast(delete(A)).

% Only preserve AST facts which are reachable from the root
% AST facts (facts over signature program/4).
reachable_id(program(P)) :- _transformed(program(P,Name,Params,Stms)).
reachable_id(ChildID) :- reachable_id(ParentID), _transformed(child(ParentID,ChildID)).

transformed(Fact) :- _transformed(Fact), ID=@ast_fact2id(Fact), reachable_id(ID).
transformed(child(ParentID,ChildID)) :- reachable_id(ParentID), _transformed(child(ParentID,ChildID)).
transformed(location(Id,Beg,End)) :- reachable_id(Id), _transformed(location(Id,Beg,End)).
