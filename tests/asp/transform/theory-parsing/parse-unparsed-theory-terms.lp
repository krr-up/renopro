% determine implicit arities of theory operators based on their position in the
% unparsed theory term.

#program step(parse_unparsed_theory_terms).
#const parse_unparsed_theory_terms=0.

% the first operator of the first unparsed term element must be unary
arity(theory_operators(Ops,0,Opname),1)
  :- theory_operators(Ops,0,Opname), 
		 theory_unparsed_term_elements(_,0,theory_operators(Ops),_).

% the first operator of any non-first unparsed term element must be binary
arity(theory_operators(Ops,0,Opname),2)
  :- theory_operators(Ops,0,Opname), 
		 theory_unparsed_term_elements(_,P,theory_operators(Ops),_), P>0.

% any non-first operator must be unary
arity(theory_operators(Ops,P,Opname),1) 
  :- theory_operators(Ops,P,Opname), P > 0.


% check that the implicit arities of all unparsed theory term operators
% match the arity in the operator table
log("error",
		"{}: No definition for operator '{}' of arity '{}' found for theory term type '{}'.",
		@loc2str(Location),Opname,Arity,Type)
  :-  undefined_operator(Type,Opname,Arity,Location).

undefined_operator(Type,Opname,Arity,location(theory_unpared_term(Id),Begin,End))
  :- theory_unparsed_term(Id,theory_unparsed_term_elements(Elements)),
		 theory_term_type(Id,Type),
		 location(theory_unparsed_term(Id),Begin,End),
		 theory_unparsed_term_elements(Elements,_,theory_operators(Ops),_),
		 arity(theory_operators(Ops,Pos,Opname),Arity),
		 not operator_table(Type,(Opname,Arity),_).

% determine priority and associativity based on operator table

prio_assoc(theory_operators(Ops,P,Opname),(Prio,Assoc))
  :- theory_unparsed_term(Id,theory_unparsed_term_elements(Elements)),
		 theory_term_type(Id,Type),
		 theory_operators(Ops,P,Opname), 
		 theory_unparsed_term_elements(Elements,_,theory_operators(Ops),_),
		 arity(theory_operators(Ops,P,Opname),Arity),
		 operator_table(Type,(Opname,Arity),(Prio,Assoc)).

% build sequence of terms and operators to be processed.
% We will go from back to front as it makes dealing with
% unary operators easier.

last_operator(theory_operators(O,P1,N))
  :- theory_operators(O,_,_), P1=#max{P: theory_operators(O,P,_)},
		 theory_operators(O,P1,N).

first_operator(theory_operators(O,P1,N))
  :- theory_operators(O,_,_),
		 P1=#min{P: theory_operators(O,P,_)}, theory_operators(O,P1,N).

_next(theory_operators(Ops,OpPos,OpName),TheoryTerm)
  :- theory_unparsed_term_elements(Elements,P,theory_operators(Ops),TheoryTerm),
		 last_operator(theory_operators(Ops,OpPos,OpName)).

_next(theory_operators(Ops,P-1,Opname2),theory_operators(Ops,P,Opname1))
  :- theory_operators(Ops,P,Opname1), theory_operators(Ops,P-1,Opname2).

_next(TheoryTerm,theory_operators(O,P,N))
  :- first_operator(theory_operators(O,P,N)),
		 theory_unparsed_term_elements(E,EP,theory_operators(O),_),
		 theory_unparsed_term_elements(E,EP-1,_,TheoryTerm).

% add a marker before the end of the next chain for use when parsing
next(X,Y) :- _next(X,Y).
next(alpha,X) :- _next(X,_), not _next(_,X).
last(T) :- next(_,T), not next(T,_).

name(theory_operators(Ops,P,Name),Name) :- theory_operators(Ops,P,Name).

% We now encode a (purely) functional Pratt parser by defining the function
% parse(Limit,Next,Token)
% where:
%   Limit is the precedence limit.
%   Next is the next token in our sequence.
%   Token is the token being parsed.
%
% The intuition behind pratt parsing, is that we set a limit, and only
% apply operators above that limit, as lower precedence operators bind
% less powerfully, and thus should be applied later.
% 
% The facts call(parse(L,N,T)) gives us the set of function calls that occur.
% return(parse(L,N,T),(N'T')) gives us the return value (O,N') of the call,
% where T' is a token, and N' designates the next token after O.

% the recur(X,Y) holds when the return value of call(X)
% is set to the return value of call(Y).
call(Y) :- recur(X,Y).
return(X,O) :- recur(X,Y), return(Y,O).

% If the next token is a unary operator with higher precedence than
% the limit, we can immediately apply it and recur.
recur(parse(Limit,Operator,Token),
			parse(Limit,Next,theory_function_(Name,(theory_terms_(0,Token),))))
  :- call(parse(Limit,Operator,Token)),
		 name(Operator,Name), arity(Operator,1),
		 prio_assoc(Operator,(Prec,_)),
		 Prec>Limit, next(Next,Operator).

% If the next token is a binary operator with higher precedence than
% the limit, we start a child parser from the left operand with a new
% precedence limit, adjusted by the associativity of the operator.
call(parse(Prec-Assoc,Next,Token1))
  :- call(parse(Limit,Operator,Token)), arity(Operator,2),
		 prio_assoc(Operator,(Prec,Assoc)), Prec>Limit, 
		 next(Next,Token1), next(Token1,Operator).

% and recur, having applied the binary operator to the token and
% whatever new token the child parser returned
recur(parse(Limit,Operator,Token),
			parse(Limit,Next1,theory_function_(Name,(theory_terms_(0,Token2),
																				       theory_terms_(1,Token)))))
  :- call(parse(Limit,Operator,Token)), arity(Operator,2),
		 prio_assoc(Operator,(Prec,Assoc)), Prec>Limit,
		 next(Next,Token1), next(Token1,Operator),
		 name(Operator,Name), 
		 return(parse(Prec-Assoc,Next,Token1),(Next1,Token2)).
	 
% If an operator has lower precedence than the limit, we are
% not allowed to parse it yet, just return.
return(parse(Limit,Operator,Token),(Operator,Token))
  :- call(parse(Limit,Operator,Token)), prio_assoc(Operator,(Prec,_)),
		 Prec<=Limit.

% Return if we run out of tokens.
return(parse(Limit,alpha,Token),(alpha,Token))
  :- call(parse(Limit,alpha,Token)).

% Start off recursion at the first token with precedence level 0.
call(parse(0,T1,T)) :- last(T), next(T1,T).

log("debug", "{}: Output of theory term parser: '{}'", 
		@loc2str(location(theory_unparsed_term(UT),Begin,End)),
		theory_function_(Name,Args))
  :-  return(parse(0,T1,T),(_,theory_function_(Name,Args))),
			last(T), next(T1,T), 
			theory_unparsed_term_elements(E,_,_,T),
			theory_unparsed_term(UT,theory_unparsed_term_elements(E)),
			location(theory_unparsed_term(UT),Begin,End).

% Finally, add the output of out Pratt parser to the AST in the correct place

parsed(@decompose(theory_function(UT,Name,Args)))
  :- return(parse(0,T1,T),(_,theory_function_(Name,Args))),
		 last(T), next(T1,T),
		 theory_unparsed_term_elements(E,_,_,T),
		 theory_unparsed_term(UT,theory_unparsed_term_elements(E)).

ast(add(A)) :- parsed(A).

ast(replace(theory_unparsed_term(UT,theory_unparsed_term_elements(E)),
						theory_function(UT,Name,A)))
  :- return(parse(0,T1,T),(_,theory_function_(Name,Args))),
		 last(T), next(T1,T),
		 theory_unparsed_term_elements(E,_,_,T),
		 theory_unparsed_term(UT,theory_unparsed_term_elements(E)),
		 parsed(theory_function(UT,Name,A)).
