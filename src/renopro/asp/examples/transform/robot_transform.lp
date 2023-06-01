% helper predicates
max_index(T,M) :- M = #max{ P : term_tuple(T,P,_) }, term_tuple(T,_,_).


% head literal (assume that prev is not allowed in the head).

new(term_tuple(T,M+1,variable("T")))
:- rule(literal(S,function(N,term_tuple(T))),_),
	 max_index(T,M).

% body literal with prev operator

replace(literal_tuple(L,P,literal(S,function("prev",term_tuple(T)))),
				literal_tuple(L,P,literal(S,function(FN,term_tuple(T')))))
:- literal_tuple(L,P,literal(S,function("prev",term_tuple(T)))),
	 term_tuple(T,0,function(FN, term_tuple(T'))),
	 not term_tuple(T,1,_).

new(term_tuple(T',M+1,binary_operation("-",variable("T"),number(1))))
:- literal_tuple(L,P,literal(S,function("prev",term_tuple(T)))),
	 term_tuple(T,0,function(FN, term_tuple(T'))),
	 not term_tuple(T,1,_),
	 max_index(T',M).

% body literal without prev operator

new(term_tuple(T,M+1,variable("T")))
:- literal_tuple(L,P,literal(S,function(N,term_tuple(T)))),
	 N != "prev",
	 max_index(T,M).
