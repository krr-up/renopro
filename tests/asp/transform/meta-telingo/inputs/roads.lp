road(A,B) :- road(B,A). % roads are bidirectional
city(A) :- road(A,B).
car(1). car(2).
road(lisbon,madrid). road(madrid,paris).
road(boston,ny). road(ny,nj).

wnext(at(X,A)) :- driveto(X,A).
driveto(X,B); no_driveto(X,B) :- at(X,A), road(A,B).
wnext(at(X,A)) :- at(X,A), not no_at(X,A).
no_at(X,A) :- at(X,B), A!=B, city(A).

:- at(X,A), at(X,B), A!=B.

% initial state
at(1,madrid) :- initial.
at(2,ny) :- initial.

#show show.
#show show(at(X,Y)) : at(X,Y).
