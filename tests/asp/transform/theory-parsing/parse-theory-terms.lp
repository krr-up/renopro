#script (python)

from clingo.symbol import String, Number

def is_operator_name(name: String):
    name_str = name.string																	
    return Number(1) if (name_str and name_str[0] in "/!<=>+-*\\?&@|:;~^.") or (name_str == "not") else Number(0)

#end.

% parse unparsed theory terms

#include "parse-unparsed-theory-terms.lp".

% check that theory functions of arity 1 or 2 that 
% represent operations agree with the grammar

undefined_operator(Type,N,Arity,location(theory_function(TF),Begin,End))
:- theory_function(TF,N,theory_terms(TT)),
	 theory_term_type(TF,Type),
   Arity = #count{ P: theory_terms(TT,P,T) }, Arity<3,
   @is_operator_name(N) = 1, 
	 not operator_table(Type,(N,Arity),_),
   location(theory_function(TF),Begin,End).
   
