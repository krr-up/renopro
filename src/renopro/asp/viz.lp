#include "defined.lp".
#const pos_body_lit_color= "#B5D99C".
#const neg_body_lit_color= "#D9A69C".
#const tab_color= "#E0E0C8".
#const program_color = "#A4C9DE".
#const rule_color = "#C8D7E0".

node(N2):-edge((N1,N2)).
% attr(node,N,label,"test") :- node(N).
attr(node,N,label,"<{{name}}{% if value is defined %} <br/><br/> <FONT POINT-SIZE='15pt'> <i><b>{{value}} </b></i></FONT> {% else %} {% endif %}>") :- node(N).
attr(node,N,(label,name),N) :- node(N).
attr(node,N,style,filled) :- node(N), not attr(node,N,shape,plaintext).
attr(edge,(N1,N2),fontsize,8):-edge((N1,N2)).


node(string(Id)) :- string(Id,Val).
attr(node,string(Id),(label,value),Val) :- string(Id,Val).
attr(node,string(Id),shape,plaintext) :- string(Id,Val).

node(number(Id)) :- number(Id,Val).
attr(node,number(Id),(label,value),Val) :- number(Id,Val).
attr(node,number(Id),shape,plaintext) :- number(Id,Val).


node(variable(Id)) :- variable(Id,Name).
attr(node,variable(Id),(label,value),Name) :- variable(Id,Name).
attr(node,variable(Id),shape,plaintext) :- node(variable(Id)).

% node(unary_operation(Id)) :- unary_operation(Id,Operator,Argument).
node(binary_operation(Id)) :- binary_operation(Id,Operator,Left,Right).
attr(node,binary_operation(Id),shape,cds) :- binary_operation(Id,Operator,Left,Right).
attr(node,binary_operation(Id),(label,value),Operator) :- binary_operation(Id,Operator,Left,Right).
edge((binary_operation(Id),Left)) :- binary_operation(Id,Operator,Left,Right).
attr(edge,(binary_operation(Id),Left),label,left) :- binary_operation(Id,Operator,Left,Right).
edge((binary_operation(Id),Right)) :- binary_operation(Id,Operator,Left,Right).
attr(edge,(binary_operation(Id),Right),label,right) :- binary_operation(Id,Operator,Left,Right).


% node(interval(Id)) :- interval(Id,Left,Right).

node(terms(Id)) :- terms(Id,Pos,Element).
attr(node,terms(Id),shape,tab) :- node(terms(Id)).
attr(node,terms(Id),fillcolor,tab_color) :- node(terms(Id)).
edge((terms(Id),Element)) :- terms(Id,Pos,Element).
attr(edge,(terms(Id),Element),label,Pos) :- terms(Id,Pos,Element).

node(function(Id)) :- function(Id,Name,Args).
edge((function(Id),Args)) :- function(Id,Name,Args).
attr(node,function(Id),(label,value),Name) :- function(Id,Name,Args).
attr(node,function(Id),shape,plaintext) :- function(Id,Name,Args).

% node(pool(Id)) :- pool(Id,Args).
% node(guard(Id)) :- guard(Id,Op,Term).
% node(guards(Id)) :- guards(Id,Pos,Element).
% node(comparison(Id)) :- comparison(Id,Guards).
% node(boolean_constant(Id)) :- boolean_constant(Id,Bool).

node(symbolic_atom(Id)) :- symbolic_atom(Id,Symbol).
attr(node,symbolic_atom(Id),shape,box) :- symbolic_atom(Id,Symbol).
edge((symbolic_atom(Id),Symbol)) :- symbolic_atom(Id,Symbol).

node(literal(Id)) :- literal(Id,Sign,Atom).
attr(node,literal(Id),fillcolor,pos_body_lit_color) :- literal(Id,"pos",Atom).
attr(node,literal(Id),fillcolor,neg_body_lit_color) :- literal(Id,"not",Atom).
edge((literal(Id),Atom)) :- literal(Id,Sign,Atom).

node(literals(Id)) :- literals(Id,Pos,Element).
attr(node,literals(Id),shape,tab) :- node(literals(Id)).
attr(node,literals(Id),fillcolor,tab_color) :- node(literals(Id)).
edge((literals(Id),Element)) :- literals(Id,Pos,Element).
attr(edge,(literals(Id),Element),label,pos) :- literals(Id,Pos,Element).

% node(conditional_literal(Id)) :- conditional_literal(Id,Literal,Condition).
% node(agg_elements(Id)) :- agg_elements(Id,Pos,Element).
% node(aggregate(Id)) :- aggregate(Id,LGuard,Elements,RGuard).
% node(body_agg_elements(Id)) :- body_agg_elements(Id,Pos,Terms,Condition).
% node(body_aggregate(Id)) :- body_aggregate(Id,LGuard,Elements,RGuard).

node(body_literal(Id)) :- body_literal(Id,Sign,Atom).
attr(node,body_literal(Id),fillcolor,pos_body_lit_color) :- body_literal(Id,"pos",Atom).
attr(node,body_literal(Id),fillcolor,neg_body_lit_color) :- body_literal(Id,"not",Atom).
edge((body_literal(Id),Atom)) :- body_literal(Id,Sign,Atom).

node(body_literals(Id)) :- body_literals(Id,Pos,Element).
attr(node,body_literals(Id),shape,tab) :- node(body_literals(Id)).
attr(node,body_literals(Id),fillcolor,tab_color) :- node(body_literals(Id)).
edge((body_literals(Id),Element)) :- body_literals(Id,Pos,Element).
attr(edge,(body_literals(Id),Element),label,Pos) :- body_literals(Id,Pos,Element).


% node(head_agg_elements(Id)) :- head_agg_elements(Id,Pos,Terms,Condition).
% node(head_aggregate(Id)) :- head_aggregate(Id,LGuard,Elements,RGuard).
% node(disjunction(Id)) :- disjunction(Id,Pos,Element).

node(rule(Id)) :- rule(Id,Head,Body).
attr(node,rule(Id),shape,rectangle) :- rule(Id,Head,Body).
attr(node,rule(Id),fillcolor,rule_color) :- rule(Id,Head,Body).
edge((rule(Id),Head)) :- rule(Id,Head,Body).
attr(edge,(rule(Id),Head),label,head) :- rule(Id,Head,Body).
edge((rule(Id),Body)) :- rule(Id,Head,Body).
attr(edge,(rule(Id),Body),label,body) :- rule(Id,Head,Body).


% node(constants(Id)) :- constants(Id,Pos,Element).

node(program(Name)) :- program(Name,Params,Statements).
attr(node,program(Name),shape,invhouse) :- program(Name,Params,Statements).
attr(node,program(Name),fillcolor,program_color) :- program(Name,Params,Statements).
edge((program(Name),Statements)) :- program(Name,Params,Statements).

node(statements(Id)) :- statements(Id,Pos,Element).
attr(node,statements(Id),shape,tab) :- node(statements(Id)).
attr(node,statements(Id),fillcolor,tab_color) :- node(statements(Id)).
edge((statements(Id),Element)) :- statements(Id,Pos,Element).
attr(edge,(statements(Id),Element),label,Pos) :- statements(Id,Pos,Element).

% node(external(At)) :- external(Atom,Body,Type).
