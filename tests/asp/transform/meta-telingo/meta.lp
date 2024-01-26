% Define one world for each time point.

conjunction(B,T) :- literal_tuple(B), time(T),
        hold(L,T) : literal_tuple(B, L), L > 0;
    not hold(L,T) : literal_tuple(B,-L), L > 0.

body(normal(B),T) :- rule(_,normal(B)), conjunction(B,T), time(T).
%% body(sum(B,G),T)  :- rule(_,sum(B,G)), time(T),
%%     #sum { W,L :     hold(L,T), weighted_literal_tuple(B, L,W), L > 0 ;
%%            W,L : not hold(L,T), weighted_literal_tuple(B,-L,W), L > 0 } >= G.

  hold(A,T) : atom_tuple(H,A)   :- rule(disjunction(H),B), body(B,T), time(T).
{ hold(A,T) : atom_tuple(H,A) } :- rule(     choice(H),B), body(B,T), time(T).

time(0..horizon).

% Only show formulas s for which we have a show statement #show show(s). in the input program

#show.
#show (S,T) : output(show(S),_), true(S,T).

%
% true(O,T) tells us that formula O holds at time point T.
%
% The implicit show statements (everything is shown unless #show. is
% in the program) give us one output(O,B) with literal tuple B
% containing 1 element L for each atom occurring in the input
% program. Therefore, we can define the truth value of atom O at
% time point T as true(O,T) via hold(L,T). Note that O here is the
% symbolic representation of the atom, allowing us to also define the
% truth value of nested formulas.
% 

true(O,T) :- hold(L,T), output(O,B), literal_tuple(B,L).
hold(L,T) :- true(O,T), output(O,B), literal_tuple(B,L).

%
% The next rule is also needed to represent the facts O of the input
% logic program, for which clingo generates output(O,0) and no
% literal_tuple of the form literal_tuple(0,L)
%

true(O,T) :- time(T), output(O,B), not literal_tuple(B,_).

% top level formulas
formula(O) :- output(O,_).

% true always holds, false never does.

true(true,T) :- formula(true), time(T).
:- formula(false), time(T), true(false,T).

% define and, or operators
formula((F;G)) :- formula(and(F,G);or(F,G)).

true(and(F,G),T) :- formula(and(F,G)), true(F,T), true(G,T).
true((F;G),T) :- formula(and(F,G)), true(and(F,G),T).

true(or(F,G),T) :- formula(or(F,G)), true((F;G),T).
true(F,T);true(G,T) :- formula(or(F,G)), true(or(F,G),T).

% define neg operator - behaves as default negation
formula(F) :- formula(neg(F)).
true(neg(F),T) :- formula(neg(F)), time(T), not true(F,T).
not true(F,T) :- formula(neg(F)), true(neg(F),T).

% define nullary initial
true(initial,0) :- formula(initial).
T=0 :- true(initial,T), formula(initial).

% define nullary final.
true(final,horizon) :- formula(final).
T=horizon :- true(final,T), formula(final).

% the define predicates allow us to define an operator in terms of an
% already defined operator

formula(F2) :- define(F1,F2), formula(F1).
true(F2,T) :- define(F1,F2), true(F1,T).
true(F1,T) :- define(F1,F2), true(F2,T).

end(-1,0).
end(1,horizon).

% define _bidir_next(F,D). It is a "next" operator that works
% _bidirectionally in time, with the next and previous operators being
% special cases of the operator. F gives the inner formula.  D denotes
% the direction we're looking: D=1 means we're looking towards the
% future, and gives us the next operator.  D=-1 means we're looking
% towards the past, and gives us the previous operator.

formula(F) :- formula(_bidir_next(F,_)).

true(F,T+D) : time(T+D)
  :- formula(_bidir_next(F,D)), true(_bidir_next(F,D),T), time(T).
true(_bidir_next(F,D),T)
  :- true(F,T+D), formula(_bidir_next(F,D)), time(T), time(T+D).

% define derived operators

% prev
define(prev(F),_bidir_next(F,-1)) :- formula(prev(F)).
% wprev
define(wprev(F),or(prev(F),initial)) :- formula(wprev(F)).
% next
define(next(F),_bidir_next(F,1)) :- formula(next(F)).
% wnext
define(wnext(F),or(next(F),final)) :- formula(wnext(F)).

% define _bidir_until(L,R,D). It is a generalization of the until and
% since operators D denotes the direction we're looking: D=1 means
% we're looking towards the future, and gives us until.  D=-1 means
% we're looking towards the past, and gives us since.

formula((L;R)) :- formula(_bidir_until(L,R,_)).

_true(_bidir_until(L,R,D),T,J) : end(D,E), M=|E-T|, J=0..M
  :- formula(_bidir_until(L,R,D)), true(_bidir_until(L,R,D),T), 
		 time(T).
true(_bidir_until(L,R,D),T)
  :- formula(_bidir_until(L,R,D)), _true(_bidir_until(L,R,D),T,_).

true(L,K) :- _true(_bidir_until(L,R,D),T,J), I=0..J-1, K=T+D*I, time(K).
true(R,T+D*J) :- _true(_bidir_until(L,R,D),T,J).
_true(_bidir_until(L,R,D),T,J)
  :- formula(_bidir_until(L,R,D)), time(T), end(D,E), M=|E-T|, J=0..M, true(R,T+D*J), 
     true(L,K) : I=0..J-1, K=T+D*I.

% until.
define(until(L,R),_bidir_until(L,R,1)) :- formula(until(L,R)).
% eventually
define(eventually(F),until(true,F)) :- formula(eventually(F)).
% since
define(since(L,R),_bidir_until(L,R,-1)) :- formula(since(L,R)).
% eventually_before
define(eventually_before(F),since(true,F)) :- formula(eventually_before(F)).

% define _bidir_release(L,R,D). It is a generalization of release and trigger.
% D denotes the direction we're looking:
% D=1 means we're looking towards the future, and gives us release.
% D=-1 means we're looking towards the past, and gives us trigger.

% note: these equivalences to not hold in the infinite case, see
define(_bidir_release(L,R,1),_bidir_until(R,and(R,or(L,final)),1))
  :- formula(_bidir_release(L,R,1)).
define(_bidir_release(L,R,-1),_bidir_until(R,and(R,or(L,initial)),-1))
  :- formula(_bidir_release(L,R,-1)).

% define derived operators
% release
define(release(L,R),_bidir_release(L,R,1)) :- formula(release(L,R)).
% always
define(always(F),release(false,F)) :- formula(always(F)).
% trigger
define(trigger(L,R),_bidir_release(L,R,-1)) :- formula(trigger(L,R)).
% always_before
define(always_before(F),trigger(false,F)) :- formula(always_before(F)).

% suppress warning messages for missing reified output atoms
#defined literal_tuple/1.
#defined literal_tuple/2.
#defined rule/2.
#defined atom_tuple/2.
#defined weighted_literal_tuple/3.
