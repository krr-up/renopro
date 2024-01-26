% replace(A,B) replaces a child predicate identifier A
% with child predicate identifier B in each AST fact where A
% occurs as a term.

ast_operation(add(unary_operation(X0,X1,B));delete(unary_operation(X0,X1,A)))
 :- ast_operation(replace(A,B)), unary_operation(X0,X1,A).
ast_operation(add(unary_operation(X0,X1,B));delete(unary_operation(X0,X1,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(unary_operation(X0,X1,A))).

ast_operation(add(binary_operation(X0,X1,B,X3));delete(binary_operation(X0,X1,A,X3)))
 :- ast_operation(replace(A,B)), binary_operation(X0,X1,A,X3).
ast_operation(add(binary_operation(X0,X1,B,X3));delete(binary_operation(X0,X1,A,X3)))
 :- ast_operation(replace(A,B)), ast_operation(add(binary_operation(X0,X1,A,X3))).

ast_operation(add(binary_operation(X0,X1,X2,B));delete(binary_operation(X0,X1,X2,A)))
 :- ast_operation(replace(A,B)), binary_operation(X0,X1,X2,A).
ast_operation(add(binary_operation(X0,X1,X2,B));delete(binary_operation(X0,X1,X2,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(binary_operation(X0,X1,X2,A))).

ast_operation(add(interval(X0,B,X2));delete(interval(X0,A,X2)))
 :- ast_operation(replace(A,B)), interval(X0,A,X2).
ast_operation(add(interval(X0,B,X2));delete(interval(X0,A,X2)))
 :- ast_operation(replace(A,B)), ast_operation(add(interval(X0,A,X2))).

ast_operation(add(interval(X0,X1,B));delete(interval(X0,X1,A)))
 :- ast_operation(replace(A,B)), interval(X0,X1,A).
ast_operation(add(interval(X0,X1,B));delete(interval(X0,X1,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(interval(X0,X1,A))).

ast_operation(add(terms(X0,X1,B));delete(terms(X0,X1,A)))
 :- ast_operation(replace(A,B)), terms(X0,X1,A).
ast_operation(add(terms(X0,X1,B));delete(terms(X0,X1,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(terms(X0,X1,A))).

ast_operation(add(function(X0,X1,B));delete(function(X0,X1,A)))
 :- ast_operation(replace(A,B)), function(X0,X1,A).
ast_operation(add(function(X0,X1,B));delete(function(X0,X1,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(function(X0,X1,A))).

ast_operation(add(external_function(X0,X1,B));delete(external_function(X0,X1,A)))
 :- ast_operation(replace(A,B)), external_function(X0,X1,A).
ast_operation(add(external_function(X0,X1,B));delete(external_function(X0,X1,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(external_function(X0,X1,A))).

ast_operation(add(pool(X0,B));delete(pool(X0,A)))
 :- ast_operation(replace(A,B)), pool(X0,A).
ast_operation(add(pool(X0,B));delete(pool(X0,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(pool(X0,A))).

ast_operation(add(theory_terms(X0,X1,B));delete(theory_terms(X0,X1,A)))
 :- ast_operation(replace(A,B)), theory_terms(X0,X1,A).
ast_operation(add(theory_terms(X0,X1,B));delete(theory_terms(X0,X1,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(theory_terms(X0,X1,A))).

ast_operation(add(theory_sequence(X0,X1,B));delete(theory_sequence(X0,X1,A)))
 :- ast_operation(replace(A,B)), theory_sequence(X0,X1,A).
ast_operation(add(theory_sequence(X0,X1,B));delete(theory_sequence(X0,X1,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(theory_sequence(X0,X1,A))).

ast_operation(add(theory_function(X0,X1,B));delete(theory_function(X0,X1,A)))
 :- ast_operation(replace(A,B)), theory_function(X0,X1,A).
ast_operation(add(theory_function(X0,X1,B));delete(theory_function(X0,X1,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(theory_function(X0,X1,A))).

ast_operation(add(theory_unparsed_term_elements(X0,X1,B,X3));delete(theory_unparsed_term_elements(X0,X1,A,X3)))
 :- ast_operation(replace(A,B)), theory_unparsed_term_elements(X0,X1,A,X3).
ast_operation(add(theory_unparsed_term_elements(X0,X1,B,X3));delete(theory_unparsed_term_elements(X0,X1,A,X3)))
 :- ast_operation(replace(A,B)), ast_operation(add(theory_unparsed_term_elements(X0,X1,A,X3))).

ast_operation(add(theory_unparsed_term_elements(X0,X1,X2,B));delete(theory_unparsed_term_elements(X0,X1,X2,A)))
 :- ast_operation(replace(A,B)), theory_unparsed_term_elements(X0,X1,X2,A).
ast_operation(add(theory_unparsed_term_elements(X0,X1,X2,B));delete(theory_unparsed_term_elements(X0,X1,X2,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(theory_unparsed_term_elements(X0,X1,X2,A))).

ast_operation(add(theory_unparsed_term(X0,B));delete(theory_unparsed_term(X0,A)))
 :- ast_operation(replace(A,B)), theory_unparsed_term(X0,A).
ast_operation(add(theory_unparsed_term(X0,B));delete(theory_unparsed_term(X0,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(theory_unparsed_term(X0,A))).

ast_operation(add(guard(X0,X1,B));delete(guard(X0,X1,A)))
 :- ast_operation(replace(A,B)), guard(X0,X1,A).
ast_operation(add(guard(X0,X1,B));delete(guard(X0,X1,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(guard(X0,X1,A))).

ast_operation(add(guards(X0,X1,B));delete(guards(X0,X1,A)))
 :- ast_operation(replace(A,B)), guards(X0,X1,A).
ast_operation(add(guards(X0,X1,B));delete(guards(X0,X1,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(guards(X0,X1,A))).

ast_operation(add(comparison(X0,B,X2));delete(comparison(X0,A,X2)))
 :- ast_operation(replace(A,B)), comparison(X0,A,X2).
ast_operation(add(comparison(X0,B,X2));delete(comparison(X0,A,X2)))
 :- ast_operation(replace(A,B)), ast_operation(add(comparison(X0,A,X2))).

ast_operation(add(comparison(X0,X1,B));delete(comparison(X0,X1,A)))
 :- ast_operation(replace(A,B)), comparison(X0,X1,A).
ast_operation(add(comparison(X0,X1,B));delete(comparison(X0,X1,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(comparison(X0,X1,A))).

ast_operation(add(symbolic_atom(X0,B));delete(symbolic_atom(X0,A)))
 :- ast_operation(replace(A,B)), symbolic_atom(X0,A).
ast_operation(add(symbolic_atom(X0,B));delete(symbolic_atom(X0,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(symbolic_atom(X0,A))).

ast_operation(add(literal(X0,X1,B));delete(literal(X0,X1,A)))
 :- ast_operation(replace(A,B)), literal(X0,X1,A).
ast_operation(add(literal(X0,X1,B));delete(literal(X0,X1,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(literal(X0,X1,A))).

ast_operation(add(literals(X0,X1,B));delete(literals(X0,X1,A)))
 :- ast_operation(replace(A,B)), literals(X0,X1,A).
ast_operation(add(literals(X0,X1,B));delete(literals(X0,X1,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(literals(X0,X1,A))).

ast_operation(add(conditional_literal(X0,B,X2));delete(conditional_literal(X0,A,X2)))
 :- ast_operation(replace(A,B)), conditional_literal(X0,A,X2).
ast_operation(add(conditional_literal(X0,B,X2));delete(conditional_literal(X0,A,X2)))
 :- ast_operation(replace(A,B)), ast_operation(add(conditional_literal(X0,A,X2))).

ast_operation(add(conditional_literal(X0,X1,B));delete(conditional_literal(X0,X1,A)))
 :- ast_operation(replace(A,B)), conditional_literal(X0,X1,A).
ast_operation(add(conditional_literal(X0,X1,B));delete(conditional_literal(X0,X1,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(conditional_literal(X0,X1,A))).

ast_operation(add(aggregate_elements(X0,X1,B));delete(aggregate_elements(X0,X1,A)))
 :- ast_operation(replace(A,B)), aggregate_elements(X0,X1,A).
ast_operation(add(aggregate_elements(X0,X1,B));delete(aggregate_elements(X0,X1,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(aggregate_elements(X0,X1,A))).

ast_operation(add(aggregate(X0,B,X2,X3));delete(aggregate(X0,A,X2,X3)))
 :- ast_operation(replace(A,B)), aggregate(X0,A,X2,X3).
ast_operation(add(aggregate(X0,B,X2,X3));delete(aggregate(X0,A,X2,X3)))
 :- ast_operation(replace(A,B)), ast_operation(add(aggregate(X0,A,X2,X3))).

ast_operation(add(aggregate(X0,X1,B,X3));delete(aggregate(X0,X1,A,X3)))
 :- ast_operation(replace(A,B)), aggregate(X0,X1,A,X3).
ast_operation(add(aggregate(X0,X1,B,X3));delete(aggregate(X0,X1,A,X3)))
 :- ast_operation(replace(A,B)), ast_operation(add(aggregate(X0,X1,A,X3))).

ast_operation(add(aggregate(X0,X1,X2,B));delete(aggregate(X0,X1,X2,A)))
 :- ast_operation(replace(A,B)), aggregate(X0,X1,X2,A).
ast_operation(add(aggregate(X0,X1,X2,B));delete(aggregate(X0,X1,X2,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(aggregate(X0,X1,X2,A))).

ast_operation(add(theory_atom_elements(X0,X1,B,X3));delete(theory_atom_elements(X0,X1,A,X3)))
 :- ast_operation(replace(A,B)), theory_atom_elements(X0,X1,A,X3).
ast_operation(add(theory_atom_elements(X0,X1,B,X3));delete(theory_atom_elements(X0,X1,A,X3)))
 :- ast_operation(replace(A,B)), ast_operation(add(theory_atom_elements(X0,X1,A,X3))).

ast_operation(add(theory_atom_elements(X0,X1,X2,B));delete(theory_atom_elements(X0,X1,X2,A)))
 :- ast_operation(replace(A,B)), theory_atom_elements(X0,X1,X2,A).
ast_operation(add(theory_atom_elements(X0,X1,X2,B));delete(theory_atom_elements(X0,X1,X2,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(theory_atom_elements(X0,X1,X2,A))).

ast_operation(add(theory_guard(X0,X1,B));delete(theory_guard(X0,X1,A)))
 :- ast_operation(replace(A,B)), theory_guard(X0,X1,A).
ast_operation(add(theory_guard(X0,X1,B));delete(theory_guard(X0,X1,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(theory_guard(X0,X1,A))).

ast_operation(add(theory_atom(X0,B,X2,X3));delete(theory_atom(X0,A,X2,X3)))
 :- ast_operation(replace(A,B)), theory_atom(X0,A,X2,X3).
ast_operation(add(theory_atom(X0,B,X2,X3));delete(theory_atom(X0,A,X2,X3)))
 :- ast_operation(replace(A,B)), ast_operation(add(theory_atom(X0,A,X2,X3))).

ast_operation(add(theory_atom(X0,X1,B,X3));delete(theory_atom(X0,X1,A,X3)))
 :- ast_operation(replace(A,B)), theory_atom(X0,X1,A,X3).
ast_operation(add(theory_atom(X0,X1,B,X3));delete(theory_atom(X0,X1,A,X3)))
 :- ast_operation(replace(A,B)), ast_operation(add(theory_atom(X0,X1,A,X3))).

ast_operation(add(theory_atom(X0,X1,X2,B));delete(theory_atom(X0,X1,X2,A)))
 :- ast_operation(replace(A,B)), theory_atom(X0,X1,X2,A).
ast_operation(add(theory_atom(X0,X1,X2,B));delete(theory_atom(X0,X1,X2,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(theory_atom(X0,X1,X2,A))).

ast_operation(add(body_aggregate_elements(X0,X1,B,X3));delete(body_aggregate_elements(X0,X1,A,X3)))
 :- ast_operation(replace(A,B)), body_aggregate_elements(X0,X1,A,X3).
ast_operation(add(body_aggregate_elements(X0,X1,B,X3));delete(body_aggregate_elements(X0,X1,A,X3)))
 :- ast_operation(replace(A,B)), ast_operation(add(body_aggregate_elements(X0,X1,A,X3))).

ast_operation(add(body_aggregate_elements(X0,X1,X2,B));delete(body_aggregate_elements(X0,X1,X2,A)))
 :- ast_operation(replace(A,B)), body_aggregate_elements(X0,X1,X2,A).
ast_operation(add(body_aggregate_elements(X0,X1,X2,B));delete(body_aggregate_elements(X0,X1,X2,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(body_aggregate_elements(X0,X1,X2,A))).

ast_operation(add(body_aggregate(X0,B,X2,X3,X4));delete(body_aggregate(X0,A,X2,X3,X4)))
 :- ast_operation(replace(A,B)), body_aggregate(X0,A,X2,X3,X4).
ast_operation(add(body_aggregate(X0,B,X2,X3,X4));delete(body_aggregate(X0,A,X2,X3,X4)))
 :- ast_operation(replace(A,B)), ast_operation(add(body_aggregate(X0,A,X2,X3,X4))).

ast_operation(add(body_aggregate(X0,X1,X2,B,X4));delete(body_aggregate(X0,X1,X2,A,X4)))
 :- ast_operation(replace(A,B)), body_aggregate(X0,X1,X2,A,X4).
ast_operation(add(body_aggregate(X0,X1,X2,B,X4));delete(body_aggregate(X0,X1,X2,A,X4)))
 :- ast_operation(replace(A,B)), ast_operation(add(body_aggregate(X0,X1,X2,A,X4))).

ast_operation(add(body_aggregate(X0,X1,X2,X3,B));delete(body_aggregate(X0,X1,X2,X3,A)))
 :- ast_operation(replace(A,B)), body_aggregate(X0,X1,X2,X3,A).
ast_operation(add(body_aggregate(X0,X1,X2,X3,B));delete(body_aggregate(X0,X1,X2,X3,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(body_aggregate(X0,X1,X2,X3,A))).

ast_operation(add(body_literal(X0,X1,B));delete(body_literal(X0,X1,A)))
 :- ast_operation(replace(A,B)), body_literal(X0,X1,A).
ast_operation(add(body_literal(X0,X1,B));delete(body_literal(X0,X1,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(body_literal(X0,X1,A))).

ast_operation(add(body_literals(X0,X1,B));delete(body_literals(X0,X1,A)))
 :- ast_operation(replace(A,B)), body_literals(X0,X1,A).
ast_operation(add(body_literals(X0,X1,B));delete(body_literals(X0,X1,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(body_literals(X0,X1,A))).

ast_operation(add(head_aggregate_elements(X0,X1,B,X3));delete(head_aggregate_elements(X0,X1,A,X3)))
 :- ast_operation(replace(A,B)), head_aggregate_elements(X0,X1,A,X3).
ast_operation(add(head_aggregate_elements(X0,X1,B,X3));delete(head_aggregate_elements(X0,X1,A,X3)))
 :- ast_operation(replace(A,B)), ast_operation(add(head_aggregate_elements(X0,X1,A,X3))).

ast_operation(add(head_aggregate_elements(X0,X1,X2,B));delete(head_aggregate_elements(X0,X1,X2,A)))
 :- ast_operation(replace(A,B)), head_aggregate_elements(X0,X1,X2,A).
ast_operation(add(head_aggregate_elements(X0,X1,X2,B));delete(head_aggregate_elements(X0,X1,X2,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(head_aggregate_elements(X0,X1,X2,A))).

ast_operation(add(head_aggregate(X0,B,X2,X3,X4));delete(head_aggregate(X0,A,X2,X3,X4)))
 :- ast_operation(replace(A,B)), head_aggregate(X0,A,X2,X3,X4).
ast_operation(add(head_aggregate(X0,B,X2,X3,X4));delete(head_aggregate(X0,A,X2,X3,X4)))
 :- ast_operation(replace(A,B)), ast_operation(add(head_aggregate(X0,A,X2,X3,X4))).

ast_operation(add(head_aggregate(X0,X1,X2,B,X4));delete(head_aggregate(X0,X1,X2,A,X4)))
 :- ast_operation(replace(A,B)), head_aggregate(X0,X1,X2,A,X4).
ast_operation(add(head_aggregate(X0,X1,X2,B,X4));delete(head_aggregate(X0,X1,X2,A,X4)))
 :- ast_operation(replace(A,B)), ast_operation(add(head_aggregate(X0,X1,X2,A,X4))).

ast_operation(add(head_aggregate(X0,X1,X2,X3,B));delete(head_aggregate(X0,X1,X2,X3,A)))
 :- ast_operation(replace(A,B)), head_aggregate(X0,X1,X2,X3,A).
ast_operation(add(head_aggregate(X0,X1,X2,X3,B));delete(head_aggregate(X0,X1,X2,X3,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(head_aggregate(X0,X1,X2,X3,A))).

ast_operation(add(conditional_literals(X0,X1,B));delete(conditional_literals(X0,X1,A)))
 :- ast_operation(replace(A,B)), conditional_literals(X0,X1,A).
ast_operation(add(conditional_literals(X0,X1,B));delete(conditional_literals(X0,X1,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(conditional_literals(X0,X1,A))).

ast_operation(add(disjunction(X0,B));delete(disjunction(X0,A)))
 :- ast_operation(replace(A,B)), disjunction(X0,A).
ast_operation(add(disjunction(X0,B));delete(disjunction(X0,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(disjunction(X0,A))).

ast_operation(add(rule(X0,B,X2));delete(rule(X0,A,X2)))
 :- ast_operation(replace(A,B)), rule(X0,A,X2).
ast_operation(add(rule(X0,B,X2));delete(rule(X0,A,X2)))
 :- ast_operation(replace(A,B)), ast_operation(add(rule(X0,A,X2))).

ast_operation(add(rule(X0,X1,B));delete(rule(X0,X1,A)))
 :- ast_operation(replace(A,B)), rule(X0,X1,A).
ast_operation(add(rule(X0,X1,B));delete(rule(X0,X1,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(rule(X0,X1,A))).

ast_operation(add(definition(X0,X1,B,X3));delete(definition(X0,X1,A,X3)))
 :- ast_operation(replace(A,B)), definition(X0,X1,A,X3).
ast_operation(add(definition(X0,X1,B,X3));delete(definition(X0,X1,A,X3)))
 :- ast_operation(replace(A,B)), ast_operation(add(definition(X0,X1,A,X3))).

ast_operation(add(show_term(X0,B,X2));delete(show_term(X0,A,X2)))
 :- ast_operation(replace(A,B)), show_term(X0,A,X2).
ast_operation(add(show_term(X0,B,X2));delete(show_term(X0,A,X2)))
 :- ast_operation(replace(A,B)), ast_operation(add(show_term(X0,A,X2))).

ast_operation(add(show_term(X0,X1,B));delete(show_term(X0,X1,A)))
 :- ast_operation(replace(A,B)), show_term(X0,X1,A).
ast_operation(add(show_term(X0,X1,B));delete(show_term(X0,X1,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(show_term(X0,X1,A))).

ast_operation(add(minimize(X0,B,X2,X3,X4));delete(minimize(X0,A,X2,X3,X4)))
 :- ast_operation(replace(A,B)), minimize(X0,A,X2,X3,X4).
ast_operation(add(minimize(X0,B,X2,X3,X4));delete(minimize(X0,A,X2,X3,X4)))
 :- ast_operation(replace(A,B)), ast_operation(add(minimize(X0,A,X2,X3,X4))).

ast_operation(add(minimize(X0,X1,B,X3,X4));delete(minimize(X0,X1,A,X3,X4)))
 :- ast_operation(replace(A,B)), minimize(X0,X1,A,X3,X4).
ast_operation(add(minimize(X0,X1,B,X3,X4));delete(minimize(X0,X1,A,X3,X4)))
 :- ast_operation(replace(A,B)), ast_operation(add(minimize(X0,X1,A,X3,X4))).

ast_operation(add(minimize(X0,X1,X2,B,X4));delete(minimize(X0,X1,X2,A,X4)))
 :- ast_operation(replace(A,B)), minimize(X0,X1,X2,A,X4).
ast_operation(add(minimize(X0,X1,X2,B,X4));delete(minimize(X0,X1,X2,A,X4)))
 :- ast_operation(replace(A,B)), ast_operation(add(minimize(X0,X1,X2,A,X4))).

ast_operation(add(minimize(X0,X1,X2,X3,B));delete(minimize(X0,X1,X2,X3,A)))
 :- ast_operation(replace(A,B)), minimize(X0,X1,X2,X3,A).
ast_operation(add(minimize(X0,X1,X2,X3,B));delete(minimize(X0,X1,X2,X3,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(minimize(X0,X1,X2,X3,A))).

ast_operation(add(statements(X0,X1,B));delete(statements(X0,X1,A)))
 :- ast_operation(replace(A,B)), statements(X0,X1,A).
ast_operation(add(statements(X0,X1,B));delete(statements(X0,X1,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(statements(X0,X1,A))).

ast_operation(add(constants(X0,X1,B));delete(constants(X0,X1,A)))
 :- ast_operation(replace(A,B)), constants(X0,X1,A).
ast_operation(add(constants(X0,X1,B));delete(constants(X0,X1,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(constants(X0,X1,A))).

ast_operation(add(program(X0,X1,B,X3));delete(program(X0,X1,A,X3)))
 :- ast_operation(replace(A,B)), program(X0,X1,A,X3).
ast_operation(add(program(X0,X1,B,X3));delete(program(X0,X1,A,X3)))
 :- ast_operation(replace(A,B)), ast_operation(add(program(X0,X1,A,X3))).

ast_operation(add(program(X0,X1,X2,B));delete(program(X0,X1,X2,A)))
 :- ast_operation(replace(A,B)), program(X0,X1,X2,A).
ast_operation(add(program(X0,X1,X2,B));delete(program(X0,X1,X2,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(program(X0,X1,X2,A))).

ast_operation(add(external(X0,B,X2,X3));delete(external(X0,A,X2,X3)))
 :- ast_operation(replace(A,B)), external(X0,A,X2,X3).
ast_operation(add(external(X0,B,X2,X3));delete(external(X0,A,X2,X3)))
 :- ast_operation(replace(A,B)), ast_operation(add(external(X0,A,X2,X3))).

ast_operation(add(external(X0,X1,B,X3));delete(external(X0,X1,A,X3)))
 :- ast_operation(replace(A,B)), external(X0,X1,A,X3).
ast_operation(add(external(X0,X1,B,X3));delete(external(X0,X1,A,X3)))
 :- ast_operation(replace(A,B)), ast_operation(add(external(X0,X1,A,X3))).

ast_operation(add(edge(X0,B,X2,X3));delete(edge(X0,A,X2,X3)))
 :- ast_operation(replace(A,B)), edge(X0,A,X2,X3).
ast_operation(add(edge(X0,B,X2,X3));delete(edge(X0,A,X2,X3)))
 :- ast_operation(replace(A,B)), ast_operation(add(edge(X0,A,X2,X3))).

ast_operation(add(edge(X0,X1,B,X3));delete(edge(X0,X1,A,X3)))
 :- ast_operation(replace(A,B)), edge(X0,X1,A,X3).
ast_operation(add(edge(X0,X1,B,X3));delete(edge(X0,X1,A,X3)))
 :- ast_operation(replace(A,B)), ast_operation(add(edge(X0,X1,A,X3))).

ast_operation(add(edge(X0,X1,X2,B));delete(edge(X0,X1,X2,A)))
 :- ast_operation(replace(A,B)), edge(X0,X1,X2,A).
ast_operation(add(edge(X0,X1,X2,B));delete(edge(X0,X1,X2,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(edge(X0,X1,X2,A))).

ast_operation(add(heuristic(X0,B,X2,X3,X4,X5));delete(heuristic(X0,A,X2,X3,X4,X5)))
 :- ast_operation(replace(A,B)), heuristic(X0,A,X2,X3,X4,X5).
ast_operation(add(heuristic(X0,B,X2,X3,X4,X5));delete(heuristic(X0,A,X2,X3,X4,X5)))
 :- ast_operation(replace(A,B)), ast_operation(add(heuristic(X0,A,X2,X3,X4,X5))).

ast_operation(add(heuristic(X0,X1,B,X3,X4,X5));delete(heuristic(X0,X1,A,X3,X4,X5)))
 :- ast_operation(replace(A,B)), heuristic(X0,X1,A,X3,X4,X5).
ast_operation(add(heuristic(X0,X1,B,X3,X4,X5));delete(heuristic(X0,X1,A,X3,X4,X5)))
 :- ast_operation(replace(A,B)), ast_operation(add(heuristic(X0,X1,A,X3,X4,X5))).

ast_operation(add(heuristic(X0,X1,X2,B,X4,X5));delete(heuristic(X0,X1,X2,A,X4,X5)))
 :- ast_operation(replace(A,B)), heuristic(X0,X1,X2,A,X4,X5).
ast_operation(add(heuristic(X0,X1,X2,B,X4,X5));delete(heuristic(X0,X1,X2,A,X4,X5)))
 :- ast_operation(replace(A,B)), ast_operation(add(heuristic(X0,X1,X2,A,X4,X5))).

ast_operation(add(heuristic(X0,X1,X2,X3,B,X5));delete(heuristic(X0,X1,X2,X3,A,X5)))
 :- ast_operation(replace(A,B)), heuristic(X0,X1,X2,X3,A,X5).
ast_operation(add(heuristic(X0,X1,X2,X3,B,X5));delete(heuristic(X0,X1,X2,X3,A,X5)))
 :- ast_operation(replace(A,B)), ast_operation(add(heuristic(X0,X1,X2,X3,A,X5))).

ast_operation(add(heuristic(X0,X1,X2,X3,X4,B));delete(heuristic(X0,X1,X2,X3,X4,A)))
 :- ast_operation(replace(A,B)), heuristic(X0,X1,X2,X3,X4,A).
ast_operation(add(heuristic(X0,X1,X2,X3,X4,B));delete(heuristic(X0,X1,X2,X3,X4,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(heuristic(X0,X1,X2,X3,X4,A))).

ast_operation(add(project_atom(X0,B,X2));delete(project_atom(X0,A,X2)))
 :- ast_operation(replace(A,B)), project_atom(X0,A,X2).
ast_operation(add(project_atom(X0,B,X2));delete(project_atom(X0,A,X2)))
 :- ast_operation(replace(A,B)), ast_operation(add(project_atom(X0,A,X2))).

ast_operation(add(project_atom(X0,X1,B));delete(project_atom(X0,X1,A)))
 :- ast_operation(replace(A,B)), project_atom(X0,X1,A).
ast_operation(add(project_atom(X0,X1,B));delete(project_atom(X0,X1,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(project_atom(X0,X1,A))).

ast_operation(add(theory_term_definitions(X0,X1,X2,B));delete(theory_term_definitions(X0,X1,X2,A)))
 :- ast_operation(replace(A,B)), theory_term_definitions(X0,X1,X2,A).
ast_operation(add(theory_term_definitions(X0,X1,X2,B));delete(theory_term_definitions(X0,X1,X2,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(theory_term_definitions(X0,X1,X2,A))).

ast_operation(add(theory_guard_definition(X0,B,X2));delete(theory_guard_definition(X0,A,X2)))
 :- ast_operation(replace(A,B)), theory_guard_definition(X0,A,X2).
ast_operation(add(theory_guard_definition(X0,B,X2));delete(theory_guard_definition(X0,A,X2)))
 :- ast_operation(replace(A,B)), ast_operation(add(theory_guard_definition(X0,A,X2))).

ast_operation(add(theory_atom_definitions(X0,X1,X2,X3,X4,X5,B));delete(theory_atom_definitions(X0,X1,X2,X3,X4,X5,A)))
 :- ast_operation(replace(A,B)), theory_atom_definitions(X0,X1,X2,X3,X4,X5,A).
ast_operation(add(theory_atom_definitions(X0,X1,X2,X3,X4,X5,B));delete(theory_atom_definitions(X0,X1,X2,X3,X4,X5,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(theory_atom_definitions(X0,X1,X2,X3,X4,X5,A))).

ast_operation(add(theory_definition(X0,X1,B,X3));delete(theory_definition(X0,X1,A,X3)))
 :- ast_operation(replace(A,B)), theory_definition(X0,X1,A,X3).
ast_operation(add(theory_definition(X0,X1,B,X3));delete(theory_definition(X0,X1,A,X3)))
 :- ast_operation(replace(A,B)), ast_operation(add(theory_definition(X0,X1,A,X3))).

ast_operation(add(theory_definition(X0,X1,X2,B));delete(theory_definition(X0,X1,X2,A)))
 :- ast_operation(replace(A,B)), theory_definition(X0,X1,X2,A).
ast_operation(add(theory_definition(X0,X1,X2,B));delete(theory_definition(X0,X1,X2,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(theory_definition(X0,X1,X2,A))).

ast_operation(add(child(B,X1));delete(child(A,X1)))
 :- ast_operation(replace(A,B)), child(A,X1).
ast_operation(add(child(B,X1));delete(child(A,X1)))
 :- ast_operation(replace(A,B)), ast_operation(add(child(A,X1))).

ast_operation(add(child(X0,B));delete(child(X0,A)))
 :- ast_operation(replace(A,B)), child(X0,A).
ast_operation(add(child(X0,B));delete(child(X0,A)))
 :- ast_operation(replace(A,B)), ast_operation(add(child(X0,A))).

