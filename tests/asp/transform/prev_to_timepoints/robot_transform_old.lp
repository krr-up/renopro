% helper predicates
max_index(T,M) :- M = #max{ P : terms(T,P,_) }, terms(T,_,_).


% head literal (assume that prev is not allowed in the head).

new(terms(T,M+1,variable("T")))
:- rule(literal(S,function(N,terms(T))),_),
	 max_index(T,M).

% body literal with prev operator

replace(body_literals(L,P,literal(S,function("prev",terms(T)))),
				body_literals(L,P,literal(S,function(FN,terms(T')))))
:- body_literals(L,P,literal(S,function("prev",terms(T)))),
	 terms(T,0,function(FN, terms(T'))),
	 not terms(T,1,_).

new(terms(T',M+1,binary_operation("-",variable("T"),number(1))))
:- body_literals(L,P,literal(S,function("prev",terms(T)))),
	 terms(T,0,function(FN, terms(T'))),
	 not terms(T,1,_),
	 max_index(T',M).

% body literal without prev operator

new(terms(T,M+1,variable("T")))
:- body_literals(L,P,literal(S,function(N,terms(T)))),
	 N != "prev",
	 max_index(T,M).
