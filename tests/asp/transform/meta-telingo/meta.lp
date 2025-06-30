%  one world for each time point.
time(0..lambda-1).
conjunction(B,K) :- literal_tuple(B), time(K),
        hold(L,K) : literal_tuple(B, L), L > 0;
    not hold(L,K) : literal_tuple(B,-L), L > 0.

body(normal(B),K) :- rule(_,normal(B)), conjunction(B,K), time(K).
%% body(sum(B,G),K)  :- rule(_,sum(B,G)), time(K),
%%     #sum { W,L :     hold(L,K), weighted_literal_tuple(B, L,W), L > 0 ;
%%            W,L : not hold(L,K), weighted_literal_tuple(B,-L,W), L > 0 } >= G.

hold(A,K) : atom_tuple(H,A)   :- rule(disjunction(H),B), body(B,K), time(K).
{ hold(A,K) : atom_tuple(H,A) } :- rule(     choice(H),B), body(B,K), time(K).

%
% true(O,K) tells us that subformula O holds at time point K.
% The implicit show statements (everything is shown unless #show. is
% in the program) give us one output(O,B) with literal tuple B
% containing 1 element L for each atom occurring in the input
% program. Therefore, we can define the truth value of atom O at
% time point K as true(O,K) via hold(L,K). Note that O here is the
% symbolic representation of the atom, allowing us to also define the
% truth value of subformulas.

% I have some reservations about the next 2 rules
% show(a).
%
true(O,K) :- hold(L,K), output(O,B), literal_tuple(B,L).
hold(L,K) :- true(O,K), output(O,B), literal_tuple(B,L).

% The next rule is also needed to represent the facts O of the input
% logic program, for which clingo generates output(O,B) and no
% literal_tuple of the form literal_tuple(B,L)

true(O,K) :- time(K), output(O,B), not literal_tuple(B,_).

% top level formulas
subformula(O) :- output(O,_).
% mark subformulas as formulas
subformula((F;G)) :- subformula((and(F,G);or(F,G);until(F,G);since(F,G))).
subformula(F)	 :- subformula((neg(F);next(F);prev(F))).

%%% definitions of operators

% true/0
true(true,K) :- subformula(true), time(K).
% false/0
:- subformula(false), time(K), true(false,K).

% and/2 
true(and(F,G),K) :- subformula(and(F,G)), true(F,K), true(G,K).
true((F;G),K) :- subformula(and(F,G)), true(and(F,G),K).

% or/2
true(or(F,G),K) :- subformula(or(F,G)), true((F;G),K).
true(F,K);true(G,K) :- subformula(or(F,G)), true(or(F,G),K).

% neg/1 (default negation)
true(neg(F),K) :- subformula(neg(F)), time(K), not true(F,K).
not true(F,K) :- subformula(neg(F)), true(neg(F),K).

% initial/0
true(initial,0) :- subformula(initial).
:- true(initial,K), subformula(initial), K!=0.

% final/0.
true(final,lambda-1) :- subformula(final).
:- true(final,K), subformula(final), K!=lambda-1.

% next/1
true(F,K+1): time(K+1) :- subformula(next(F)), true(next(F),K).
true(next(F),K) :- true(F,K+1), subformula(next(F)), time(K).

% prev/1
true(F,K-1): time(K-1) :- subformula(prev(F)), true(prev(F),K).
true(prev(F),K) :- true(F,K-1), subformula(prev(F)), time(K).


% until/2
true(until_then(L,R,J),K): K<=J, time(J)
  :- subformula(until(L,R)), true(until(L,R),K), time(K).
true(until(L,R),K)
  :- subformula(until(L,R)), true(until_then(L,R,_),K).

true(L,I) :- true(until_then(L,R,J),K), I=K..J-1.
true(R,J) :- true(until_then(L,R,J),K).
true(until_then(L,R,J),K)
  :- subformula(until(L,R)), time(K), time(J), K<=J, true(R,J), 
     true(L,I) : I=K..J-1.

% since/2.
true(since_then(L,R,J),K): J<=K, time(J)
  :- subformula(since(L,R)), true(since(L,R),K), time(K).
true(since(L,R),K)
  :- subformula(since(L,R)), true(since_then(L,R,_),K).

true(L,I) :- true(since_then(L,R,J),K), I=J+1..K.
true(R,J) :- true(since_then(L,R,J),K).
true(since_then(L,R,J),K)
  :- subformula(since(L,R)), time(K), time(J), J<=K, true(R,J), 
     true(L,I) : I=J+1..K.

% the helper derive predicates allow us to derive an operator in terms
% of an already defined operator
subformula(F2) :- derive(F1,F2), subformula(F1).
true(F2,K) :- derive(F1,F2), true(F1,K).
true(F1,K) :- derive(F1,F2), true(F2,K).

% wnext/1
derive(wnext(F),or(next(F),final)) :- subformula(wnext(F)).
% wprev/1
derive(wprev(F),or(prev(F),initial)) :- subformula(wprev(F)).
% eventually_after/1
derive(eventually_after(F),until(true,F)) :- subformula(eventually_after(F)).
% eventually_before/1
derive(eventually_before(F),since(true,F)) :- subformula(eventually_before(F)).
% always_after/1
derive(always_after(F),release(false,F)) :- subformula(always_after(F)).
% always_before/1
derive(always_before(F),trigger(false,F)) :- subformula(always_before(F)).
% release/2
derive(release(L,R),until(R,and(R,or(L,final)))) :- subformula(release(L,R)).
% trigger/2
derive(trigger(L,R),since(R,and(R,or(L,initial)))) :- subformula(trigger(L,R)).

% output handling.

% distinguish operator subformulas

not_atom(F) 
     :- subformula(F), 
		 F=(true;false;initial;final;neg(_);next(_);prev(_);wprev(_);wnext(_);
				eventually_after(_);eventually_before(_);always_after(_);always_before(_);
				and(_,_);or(_,_);until(_,_);since(_,_);release(_,_);trigger(_,_);
			  show(_)).

% show statments are reified strangely
% the output statement generated for a shown term a.
% never matches the output for a symbolic atom.

% 1. Show statements with no condition 
% e.g.: 
% #show a.
% get reified in this manner:
% literal_tuple(3).
% literal_tuple(3,-3).
% output(a,3).
% where -3 is guaranteed to  not occur anywhere else in the program. 
% All other show statements with no condition share this literal tuple.

% 2. If there is a single condition which is not simplified away during grounding
% e.g.: 
% #show a : b. {b}.
% then we get
%% atom_tuple(0).
%% atom_tuple(0,1).
%% literal_tuple(0).
%% rule(choice(0),normal(0)).
%% literal_tuple(1).
%% literal_tuple(1,1).
%% output(b,1).
%% output(a,1).
% with the condition and the shown term are being linked to the same literal tuple

% 3. If there are multiple conditions which are not simplified away, 
% e.g.: #show a : b, c. {b}. {c}.
% then we get
%% atom_tuple(0).
%% atom_tuple(0,1).
%% literal_tuple(0).
%% rule(choice(0),normal(0)).
%% atom_tuple(1).
%% atom_tuple(1,2).
%% rule(choice(1),normal(0)).
%% literal_tuple(1).
%% literal_tuple(1,1).
%% output(c,1).
%% literal_tuple(2).
%% literal_tuple(2,2).
%% output(b,2).
%% atom_tuple(2).
%% atom_tuple(2,3).
%% literal_tuple(3).
%% literal_tuple(3,1).
%% literal_tuple(3,2).
%% rule(disjunction(2),normal(3)).
%% literal_tuple(4).
%% literal_tuple(4,3).
%% output(a,4).

#show.
% We imitate clingo's show statement behaviour
% Show all atomic subformulas occurring in the program that are true, 
% unless we have #show show. statement occurring in the program, which
% suppresses this behaviour.
#show (F,K) : subformula(F), not not_atom(F), true(F,K), not output(show,_).

% Additionally, we show any other terms (S,K) that are marked
% explicitly to be shown via #show show(S) : <body>. statements, if
% the body is satisfied.
#show (S,K) : output(show(S),B), conjunction(B,K).

% Finally, to handle the case of show statements with no condition,
% we need this rule.
#show (S,K) : time(K), output(show(S),B), literal_tuple(B,L), not atom_tuple(_,L).

% suppress warning messages for missing reified output atoms
#defined literal_tuple/1.
#defined literal_tuple/2.
#defined rule/2.
#defined atom_tuple/2.
#defined weighted_literal_tuple/3.
