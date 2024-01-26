%
&tel{ >? fail(X) } :- &tel { shoot(X) & < <? shoot(X) & <* unloaded(X) }.
&tel{ shoot(X) & unloaded(X) & > unloaded(X) & > > (unloaded(X) >? < shoot(X)) } 
  :- &tel{ initial & firearm(X) }.
firearm(gun).

#show show(shoot(X)) : shoot(X).
#show show(fail(X)) : fail(X).
#show show(unloaded(X)) : unloaded(X).
