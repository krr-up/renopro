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
&tel { >?(fail(X)) } :- &tel { &(shoot(X),<(&(<?(shoot(X)),<*(unloaded(X))))) }.
