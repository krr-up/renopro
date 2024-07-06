%% #include "wrap_ast.lp".
%% #include "unwrap_ast.lp".
%% #include "defined.lp".
%% #include "replace_id.lp".
%% #include "scripts.lp".

% add operation should also generate an appropriate child/2 fact to link the new fact
% to it's children
ast(add(child(@ast_fact2id(Fact),@ast_fact2children_ids(Fact))),T) :- ast(add(Fact),T), #false: Fact=child(_,_).

ast((delete(A);add(B)),T) :- ast(replace(A,B),T).

ast(replace_id(@ast_fact2id(A),@ast_fact2id(B)),T)  :- ast(replace(A,B),T).

% delete stops inertia from old ast, and even deletes asts that would have been added
_transformed(A,T) :- ast((fact(A);add(A)),T), not ast(delete(A),T).

% Only preserve AST facts which are reachable from the root
% AST facts (facts over signature program/4).
reachable_id(program(P),T) :- _transformed(program(P,_,_,_),T).
reachable_id(ChildID,T) :- reachable_id(ParentID,T), _transformed(child(ParentID,ChildID),T).

transformed(Fact,T) :- _transformed(Fact,T), ID=@ast_fact2id(Fact), reachable_id(ID,T).
transformed(child(ParentID,ChildID),T) :- reachable_id(ParentID,T), _transformed(child(ParentID,ChildID),T).
transformed(location(Id,Beg,End),T) :- reachable_id(Id,T), _transformed(location(Id,Beg,End),T).

ast(fact(Fact),T+1) :- transformed(Fact,T), T < transform_steps.
