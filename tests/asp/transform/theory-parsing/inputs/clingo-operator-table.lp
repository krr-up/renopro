% operator table for clingo's built-in operators
% in the second argument of the third argument, 
% 1 stands for left associative
% and 0 stands for right associative 
% or no associativity in unary case.
#program step(parse_unparsed_theory_terms).

operator_table(clingo,("-",1),(5,0)).
operator_table(clingo,("~",1),(5,0)).
operator_table(clingo,("**",2),(4,0)).
operator_table(clingo,("*",2),(3,1)).
operator_table(clingo,("/",2),(3,1)).
operator_table(clingo,("\\",2),(3,1)).
operator_table(clingo,("+",2),(2,1)).
operator_table(clingo,("-",2),(2,1)).
operator_table(clingo,("&",2),(1,1)).
operator_table(clingo,("?",2),(1,1)).
operator_table(clingo,("^",2),(1,1)).
operator_table(clingo,("..",2),(0,1)).

% interpret all theory terms as clingo terms
theory_term_type(Id,clingo) :- theory_unparsed_term(Id,_).
theory_term_type(TF,clingo) :- theory_function(TF,_,_).
