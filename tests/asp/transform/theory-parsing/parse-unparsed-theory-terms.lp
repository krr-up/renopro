% determine implicit arities of theory operators based on their position in the
% unparsed theory term.

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
#show log("error",
					"{}: No definition for operator '{}' of arity '{}' found for theory term type '{}'.",
					Location,Opname,Arity,Type)
  : undefined_operator(Type,Opname,Arity,Location).

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

last_operator(theory_operators(O,P',N))
  :- theory_operators(O,_,_), P'=#max{P: theory_operators(O,P,_)},
		 theory_operators(O,P',N).

first_operator(theory_operators(O,P',N))
  :- theory_operators(O,_,_),
		 T'=#min{P: theory_operators(O,P,_)}, theory_operators(O,P',N).

_next(TheoryTerm,theory_operators(Ops,OpPos,OpName))
  :- theory_unparsed_term_elements(Elements,P,theory_operators(Ops),TheoryTerm),
		 last_operator(theory_operators(Ops,OpPos,OpName)).

_next(theory_operators(Ops,P,Opname1),theory_operators(Ops,P-1,Opname2))
  :- theory_operators(Ops,P,Opname1), theory_operators(Ops,P-1,Opname2).

_next(theory_operators(O,P,N),TheoryTerm)
  :- first_operator(theory_operators(O,P,N)),
		 theory_unparsed_term_elements(E,EP,theory_operators(O),_),
		 theory_unparsed_term_elements(E,EP-1,_,TheoryTerm).

next(X,Y) :- _next(X,Y).
% add a marker after the end of the next chain for use when parsing
next(Y,afterlast) :- _next(X,Y), not _next(Y,_).

first(T) :- next(T,T'), not next(_,T).

name(theory_operators(Ops,P,Name),Name) :- theory_operators(Ops,P,Name).

% We now encode a (purely) functional Pratt parser by defining the function
% parse(Limit,Right,Next)
% where:
%   Limit is the precedence limit.
%   Right is the token being parsed.
%   Next is the next token in our sequence.
%
% The intuition behind pratt parsing, is that we set a limit, and only
% apply operators above that limit, as lower precedence operators bind
% less powerfully, and thus should be applied later.
% 
% The facts call(parse(L,R,N)) gives us the set of function calls that occur.
% return(parse(L,R,N),(O,N')) gives us the return value (O,N') of the call,
% where O is a token, and N' designates the token after O.

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
		 Prec>Limit, next(Operator,Next).

% If the next token is a binary operator with higher precedence than
% the limit, we start a child parser from the left operand with a new
% precedence limit, adjusted by the associativity of the operator.
call(parse(Prec-Assoc,Next,Token'))
  :- call(parse(Limit,Operator,Token)), arity(Operator,2),
		 prio_assoc(Operator,(Prec,Assoc)), Prec>Limit, 
		 next(Operator,Token'), next(Token',Next).

% and recur, having applied the binary operator to the token and
% whatever new token the child parser returned
recur(parse(Limit,Operator,Token),
			parse(Limit,Next',theory_function_(Name,(theory_terms_(0,Token''),
																				       theory_terms_(1,Token)))))
  :- call(parse(Limit,Operator,Token)), arity(Operator,2),
		 prio_assoc(Operator,(Prec,Assoc)), Prec>Limit, 
		 next(Operator,Token'), next(Token',Next),
		 name(Operator,Name), 
		 return(parse(Prec-Assoc,Next,Token'),(Next',Token'')).
	 
% If an operator has lower precedence than the limit, we are
% not allowed to parse it yet, just return.
return(parse(Limit,Operator,Token),(Operator,Token))
  :- call(parse(Limit,Operator,Token)), prio_assoc(Operator,(Prec,_)),
		 Prec<=Limit.

% Return if we run out of tokens.
return(parse(Limit,afterlast,Token),(afterlast,Token))
  :- call(parse(Limit,afterlast,Token)).

% Start off recursion at the first token with precedence level 0.
call(parse(0,T',T)) :- first(T), next(T,T').

#show log("debug", "{}: Output of theory term parser: '{}'", 
					location(theory_unparsed_term(UT),Begin,End), 
					theory_function_(Name,Args))
  :  return(parse(0,T',T),(_,theory_function_(Name,Args))),
     first(T), next(T,T'), 
		 theory_unparsed_term_elements(E,_,_,T),
		 theory_unparsed_term(UT,theory_unparsed_term_elements(E)),
		 location(theory_unparsed_term(UT),Begin,End).

% Finally, add the output of out Pratt parser to the AST in the correct place

ast_operation(
		add(@decompose(theory_function(UT,Name,Args)));
		replace(theory_unparsed_term(UT),theory_function(UT)))
  :- return(parse(0,T',T),(_,theory_function_(Name,Args))),
		 first(T), next(T,T'),
		 theory_unparsed_term_elements(E,_,_,T),
		 theory_unparsed_term(UT,theory_unparsed_term_elements(E)),
		 theory_terms(TT,P,theory_unparsed_term(UT)).
