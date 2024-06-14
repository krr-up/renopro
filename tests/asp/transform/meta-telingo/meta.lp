%  one world for each time point.

conjunction(B,T) :- literal_tuple(B), time(T),
        hold(L,T) : literal_tuple(B, L), L > 0;
    not hold(L,T) : literal_tuple(B,-L), L > 0.

body(normal(B),T) :- rule(_,normal(B)), conjunction(B,T), time(T).
%% body(sum(B,G),T)  :- rule(_,sum(B,G)), time(T),
%%     #sum { W,L :     hold(L,T), weighted_literal_tuple(B, L,W), L > 0 ;
%%            W,L : not hold(L,T), weighted_literal_tuple(B,-L,W), L > 0 } >= G.

  hold(A,T) : atom_tuple(H,A)   :- rule(disjunction(H),B), body(B,T), time(T).
{ hold(A,T) : atom_tuple(H,A) } :- rule(     choice(H),B), body(B,T), time(T).

time(0..lambda-1).
%
% true(O,T) tells us that formula O holds at time point T.
% The implicit show statements (everything is shown unless #show. is
% in the program) give us one output(O,B) with literal tuple B
% containing 1 element L for each atom occurring in the input
% program. Therefore, we can define the truth value of atom O at
% time point T as true(O,T) via hold(L,T). Note that O here is the
% symbolic representation of the atom, allowing us to also define the
% truth value of nested formulas.

true(O,T) :- hold(L,T), output(O,B), literal_tuple(B,L).
hold(L,T) :- true(O,T), output(O,B), literal_tuple(B,L).

% The next rule is also needed to represent the facts O of the input
% logic program, for which clingo generates output(O,0) and no
% literal_tuple of the form literal_tuple(0,L)

true(O,T) :- time(T), output(O,B), not literal_tuple(B,_).

% top level formulas
formula(O) :- output(O,_).
% mark subformulas as formulas
formula(F;G) :- formula(and(F,G);or(F,G);until(F,G);since(F,G)).
formula(F) :- formula(neg(F);next(F);prev(F)).
% distinguish operator formulas
not_atom(X)
  :- formula(X), 
		 X=(true;false;initial;final;neg(_);next(_);prev(_);weak_prev(_);weak_next(_);
				eventually_after(_);eventually_before(_);always_after(_);always_before(_);
				and(_,_);or(_,_);until(_,_);since(_,_);release(_,_);trigger(_,_)).

#show.
% Show all atomic formulas by default.
#show (F,T) : formula(F), not not_atom(F), true(F,T), not output(show,_).
% #show show. in the input program is interpreted similarly as #show. in plain ASP,
% hiding all output, except for terms t specifically marked to be shown via a
% #show show(t). statement
#show (S,T) : output(show(S),_), true(S,T), output(show,_).


%%% definitions of operators

% true/0
true(true,T) :- formula(true), time(T).

% false/0
:- formula(false), time(T), true(false,T).

% and/2 
true(and(F,G),T) :- formula(and(F,G)), true(F,T), true(G,T).
true((F;G),T) :- formula(and(F,G)), true(and(F,G),T).

% or/2
true(or(F,G),T) :- formula(or(F,G)), true((F;G),T).
true(F,T);true(G,T) :- formula(or(F,G)), true(or(F,G),T).

% neg/1 (default negation)
true(neg(F),T) :- formula(neg(F)), time(T), not true(F,T).
not true(F,T) :- formula(neg(F)), true(neg(F),T).

% initial/0
true(initial,T) :- formula(initial), T=0.
:- true(initial,T), formula(initial), T!=0.

% final/0.
true(final,T) :- formula(final), T=lambda-1.
:- true(final,T), formula(final), T!=lambda-1.

% the define predicates allow us to define an operator in terms of an
% already defined operator
formula(F2) :- define(F1,F2), formula(F1).
true(F2,T) :- define(F1,F2), true(F1,T).
true(F1,T) :- define(F1,F2), true(F2,T).

% next/1
true(F,T+1): time(T+1) :- formula(next(F)), true(next(F),T).
true(next(F),T) :- true(F,T+1), formula(next(F)), time(T).

% prev/1
true(F,T-1): time(T-1) :- formula(prev(F)), true(prev(F),T).
true(prev(F),T) :- true(F,T-1), formula(prev(F)), time(T).

% weak_prev/1
define(weak_prev(F),or(prev(F),initial)) :- formula(weak_prev(F)).

% weak_next/1
define(weak_next(F),or(next(F),final)) :- formula(weak_next(F)).

% until/2
true(until_then(L,R,J),T): T<=J, time(J)
  :- formula(until(L,R)), true(until(L,R),T), time(T).
true(until(L,R),T)
  :- formula(until(L,R)), true(until_then(L,R,_),T).

true(L,K) :- true(until_then(L,R,J),T), K=T..J-1.
true(R,J) :- true(until_then(L,R,J),T).
true(until_then(L,R,J),T)
  :- formula(until(L,R)), time(T), time(J), T<=J, true(R,J), 
     true(L,K) : K=T..J-1.

% since/2.
true(since_then(L,R,J),T): J<=T, time(J)
  :- formula(since(L,R)), true(since(L,R),T), time(T).
true(since(L,R),T)
  :- formula(since(L,R)), true(since_then(L,R,_),T).

true(L,K) :- true(since(L,R,J),T), K=J+1..T.
true(R,J) :- true(since(L,R,J),T).
true(since(L,R,J),T)
  :- formula(since(L,R)), time(T), time(J), J<=T, true(R,J), 
     true(L,K) : K=J+1..T.

% eventually_after/1
define(eventually_after(F),until(true,F)) :- formula(eventually_after(F)).
% eventually_before/1
define(eventually_before(F),since(true,F)) :- formula(eventually_before(F)).

% define derived operators
% release/2
define(release(L,R),until(R,and(R,or(L,final)))) :- formula(release(L,R)).
% always_after/1
define(always_after(F),release(false,F)) :- formula(always_after(F)).
% trigger/2
define(trigger(L,R),since(R,and(R,or(L,initial)))) :- formula(trigger(L,R)).
% always_before/1
define(always_before(F),trigger(false,F)) :- formula(always_before(F)).

% suppress warning messages for missing reified output atoms
#defined literal_tuple/1.
#defined literal_tuple/2.
#defined rule/2.
#defined atom_tuple/2.
#defined weighted_literal_tuple/3.
