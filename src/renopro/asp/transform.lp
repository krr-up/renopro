#include "ast.lp".
#include "defined.lp".
#include "replace_id.lp".
#include "decompose.lp".
#include "add-children.lp".
#include "ast_fact2id.lp".

ast(delete(A);add(B)) :- ast(replace(A,B)).
ast(_replace_id(AId,BId))  :- ast(replace(A,B)), ast_fact2id(A,AId), ast_fact2id(B,BId).

% delete stops inertia from old ast
transformed(A) :- ast(fact(A)), not ast(delete(A)).
% and even overrides adding of new ast
transformed(A) :- ast(add(A)), not ast(delete(A)).

transformed(ast_fact2id(Fact,FactId)) :- transformed(Fact), ast_fact2id(Fact,FactId).

% root
final(program(P,Name,Params,Stms)) :- transformed(program(P,Name,Params,Stms)).
% children
final(child(ParentId,ChildId);ChildFact) 
  :- final(ParentFact), transformed(ast_fact2id(ParentFact,ParentId)),
		 transformed(child(ParentId,ChildId)), transformed(ast_fact2id(ChildFact,ChildId)).

#show final/1.

