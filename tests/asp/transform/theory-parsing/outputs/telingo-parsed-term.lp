&tel { >?(fail(X)) } :- &tel { &(shoot(X),<(&(<?(shoot(X)),<*(unloaded(X))))) }.
% note: the parentheses here              ^
% is used for parenthesising operands of operator, and is
% not a theory sequence with one element. Here we apply next to 
% the entire subformula
#program base.
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
