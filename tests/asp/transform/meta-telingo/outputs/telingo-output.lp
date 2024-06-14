% previous
% weak previous
% eventually-
% always-
% initially
% next
% weak next
% eventually+
% always+
% release
% until
% trigger
% since
% and
% or
#theory tel {
  formula {
    < : 5, unary;
    <: : 5, unary;
    <? : 5, unary;
    <* : 5, unary;
    << : 5, unary;
    > : 5, unary;
    >: : 5, unary;
    >? : 5, unary;
    >* : 5, unary;
    >* : 4, binary, left;
    >? : 4, binary, left;
    <* : 4, binary, left;
    <? : 4, binary, left;
    & : 3, binary, left;
    | : 2, binary, left
  };
  &tel/0: formula, any
}.
#program base.
eventually(fail(X)) :- and(and(shoot(X),prev(eventually_before(shoot(X)))),always_before(unloaded(X))).
and(and(and(shoot(X),unloaded(X)),next(unloaded(X))),next(next(until(unloaded(X),prev(shoot(X)))))) :- and(initial,firearm(X)).
firearm(gun).
#show show(shoot(X)) : shoot(X).
#show show(fail(X)) : fail(X).
#show show(unloaded(X)) : unloaded(X).
#program base.
#external and(and(shoot(X),prev(eventually_before(shoot(X)))),always_before(unloaded(X))) : shoot(X); shoot(X); unloaded(X). [false]
#external and(initial,firearm(X)) : firearm(X). [false]
#external fail(X) : shoot(X); shoot(X); unloaded(X). [false]
#external shoot(X) : firearm(X). [false]
#external unloaded(X) : firearm(X). [false]
#external unloaded(X) : firearm(X). [false]
#external unloaded(X) : firearm(X). [false]
#external shoot(X) : firearm(X). [false]
