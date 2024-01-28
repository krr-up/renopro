% replace(A,B) replaces a child predicate identifier A
% with child predicate identifier B in each AST fact where A
% occurs as a term.

ast(add(unary_operation(X0,X1,B));delete(unary_operation(X0,X1,A)))
 :- ast(_replace_id(A,B)), ast(fact(unary_operation(X0,X1,A));add(unary_operation(X0,X1,A))).

ast(add(binary_operation(X0,X1,B,X3));delete(binary_operation(X0,X1,A,X3)))
 :- ast(_replace_id(A,B)), ast(fact(binary_operation(X0,X1,A,X3));add(binary_operation(X0,X1,A,X3))).

ast(add(binary_operation(X0,X1,X2,B));delete(binary_operation(X0,X1,X2,A)))
 :- ast(_replace_id(A,B)), ast(fact(binary_operation(X0,X1,X2,A));add(binary_operation(X0,X1,X2,A))).

ast(add(interval(X0,B,X2));delete(interval(X0,A,X2)))
 :- ast(_replace_id(A,B)), ast(fact(interval(X0,A,X2));add(interval(X0,A,X2))).

ast(add(interval(X0,X1,B));delete(interval(X0,X1,A)))
 :- ast(_replace_id(A,B)), ast(fact(interval(X0,X1,A));add(interval(X0,X1,A))).

ast(add(terms(X0,X1,B));delete(terms(X0,X1,A)))
 :- ast(_replace_id(A,B)), ast(fact(terms(X0,X1,A));add(terms(X0,X1,A))).

ast(add(function(X0,X1,B));delete(function(X0,X1,A)))
 :- ast(_replace_id(A,B)), ast(fact(function(X0,X1,A));add(function(X0,X1,A))).

ast(add(external_function(X0,X1,B));delete(external_function(X0,X1,A)))
 :- ast(_replace_id(A,B)), ast(fact(external_function(X0,X1,A));add(external_function(X0,X1,A))).

ast(add(pool(X0,B));delete(pool(X0,A)))
 :- ast(_replace_id(A,B)), ast(fact(pool(X0,A));add(pool(X0,A))).

ast(add(theory_terms(X0,X1,B));delete(theory_terms(X0,X1,A)))
 :- ast(_replace_id(A,B)), ast(fact(theory_terms(X0,X1,A));add(theory_terms(X0,X1,A))).

ast(add(theory_sequence(X0,X1,B));delete(theory_sequence(X0,X1,A)))
 :- ast(_replace_id(A,B)), ast(fact(theory_sequence(X0,X1,A));add(theory_sequence(X0,X1,A))).

ast(add(theory_function(X0,X1,B));delete(theory_function(X0,X1,A)))
 :- ast(_replace_id(A,B)), ast(fact(theory_function(X0,X1,A));add(theory_function(X0,X1,A))).

ast(add(theory_unparsed_term_elements(X0,X1,B,X3));delete(theory_unparsed_term_elements(X0,X1,A,X3)))
 :- ast(_replace_id(A,B)), ast(fact(theory_unparsed_term_elements(X0,X1,A,X3));add(theory_unparsed_term_elements(X0,X1,A,X3))).

ast(add(theory_unparsed_term_elements(X0,X1,X2,B));delete(theory_unparsed_term_elements(X0,X1,X2,A)))
 :- ast(_replace_id(A,B)), ast(fact(theory_unparsed_term_elements(X0,X1,X2,A));add(theory_unparsed_term_elements(X0,X1,X2,A))).

ast(add(theory_unparsed_term(X0,B));delete(theory_unparsed_term(X0,A)))
 :- ast(_replace_id(A,B)), ast(fact(theory_unparsed_term(X0,A));add(theory_unparsed_term(X0,A))).

ast(add(guard(X0,X1,B));delete(guard(X0,X1,A)))
 :- ast(_replace_id(A,B)), ast(fact(guard(X0,X1,A));add(guard(X0,X1,A))).

ast(add(guards(X0,X1,B));delete(guards(X0,X1,A)))
 :- ast(_replace_id(A,B)), ast(fact(guards(X0,X1,A));add(guards(X0,X1,A))).

ast(add(comparison(X0,B,X2));delete(comparison(X0,A,X2)))
 :- ast(_replace_id(A,B)), ast(fact(comparison(X0,A,X2));add(comparison(X0,A,X2))).

ast(add(comparison(X0,X1,B));delete(comparison(X0,X1,A)))
 :- ast(_replace_id(A,B)), ast(fact(comparison(X0,X1,A));add(comparison(X0,X1,A))).

ast(add(symbolic_atom(X0,B));delete(symbolic_atom(X0,A)))
 :- ast(_replace_id(A,B)), ast(fact(symbolic_atom(X0,A));add(symbolic_atom(X0,A))).

ast(add(literal(X0,X1,B));delete(literal(X0,X1,A)))
 :- ast(_replace_id(A,B)), ast(fact(literal(X0,X1,A));add(literal(X0,X1,A))).

ast(add(literals(X0,X1,B));delete(literals(X0,X1,A)))
 :- ast(_replace_id(A,B)), ast(fact(literals(X0,X1,A));add(literals(X0,X1,A))).

ast(add(conditional_literal(X0,B,X2));delete(conditional_literal(X0,A,X2)))
 :- ast(_replace_id(A,B)), ast(fact(conditional_literal(X0,A,X2));add(conditional_literal(X0,A,X2))).

ast(add(conditional_literal(X0,X1,B));delete(conditional_literal(X0,X1,A)))
 :- ast(_replace_id(A,B)), ast(fact(conditional_literal(X0,X1,A));add(conditional_literal(X0,X1,A))).

ast(add(aggregate_elements(X0,X1,B));delete(aggregate_elements(X0,X1,A)))
 :- ast(_replace_id(A,B)), ast(fact(aggregate_elements(X0,X1,A));add(aggregate_elements(X0,X1,A))).

ast(add(aggregate(X0,B,X2,X3));delete(aggregate(X0,A,X2,X3)))
 :- ast(_replace_id(A,B)), ast(fact(aggregate(X0,A,X2,X3));add(aggregate(X0,A,X2,X3))).

ast(add(aggregate(X0,X1,B,X3));delete(aggregate(X0,X1,A,X3)))
 :- ast(_replace_id(A,B)), ast(fact(aggregate(X0,X1,A,X3));add(aggregate(X0,X1,A,X3))).

ast(add(aggregate(X0,X1,X2,B));delete(aggregate(X0,X1,X2,A)))
 :- ast(_replace_id(A,B)), ast(fact(aggregate(X0,X1,X2,A));add(aggregate(X0,X1,X2,A))).

ast(add(theory_atom_elements(X0,X1,B,X3));delete(theory_atom_elements(X0,X1,A,X3)))
 :- ast(_replace_id(A,B)), ast(fact(theory_atom_elements(X0,X1,A,X3));add(theory_atom_elements(X0,X1,A,X3))).

ast(add(theory_atom_elements(X0,X1,X2,B));delete(theory_atom_elements(X0,X1,X2,A)))
 :- ast(_replace_id(A,B)), ast(fact(theory_atom_elements(X0,X1,X2,A));add(theory_atom_elements(X0,X1,X2,A))).

ast(add(theory_guard(X0,X1,B));delete(theory_guard(X0,X1,A)))
 :- ast(_replace_id(A,B)), ast(fact(theory_guard(X0,X1,A));add(theory_guard(X0,X1,A))).

ast(add(theory_atom(X0,B,X2,X3));delete(theory_atom(X0,A,X2,X3)))
 :- ast(_replace_id(A,B)), ast(fact(theory_atom(X0,A,X2,X3));add(theory_atom(X0,A,X2,X3))).

ast(add(theory_atom(X0,X1,B,X3));delete(theory_atom(X0,X1,A,X3)))
 :- ast(_replace_id(A,B)), ast(fact(theory_atom(X0,X1,A,X3));add(theory_atom(X0,X1,A,X3))).

ast(add(theory_atom(X0,X1,X2,B));delete(theory_atom(X0,X1,X2,A)))
 :- ast(_replace_id(A,B)), ast(fact(theory_atom(X0,X1,X2,A));add(theory_atom(X0,X1,X2,A))).

ast(add(body_aggregate_elements(X0,X1,B,X3));delete(body_aggregate_elements(X0,X1,A,X3)))
 :- ast(_replace_id(A,B)), ast(fact(body_aggregate_elements(X0,X1,A,X3));add(body_aggregate_elements(X0,X1,A,X3))).

ast(add(body_aggregate_elements(X0,X1,X2,B));delete(body_aggregate_elements(X0,X1,X2,A)))
 :- ast(_replace_id(A,B)), ast(fact(body_aggregate_elements(X0,X1,X2,A));add(body_aggregate_elements(X0,X1,X2,A))).

ast(add(body_aggregate(X0,B,X2,X3,X4));delete(body_aggregate(X0,A,X2,X3,X4)))
 :- ast(_replace_id(A,B)), ast(fact(body_aggregate(X0,A,X2,X3,X4));add(body_aggregate(X0,A,X2,X3,X4))).

ast(add(body_aggregate(X0,X1,X2,B,X4));delete(body_aggregate(X0,X1,X2,A,X4)))
 :- ast(_replace_id(A,B)), ast(fact(body_aggregate(X0,X1,X2,A,X4));add(body_aggregate(X0,X1,X2,A,X4))).

ast(add(body_aggregate(X0,X1,X2,X3,B));delete(body_aggregate(X0,X1,X2,X3,A)))
 :- ast(_replace_id(A,B)), ast(fact(body_aggregate(X0,X1,X2,X3,A));add(body_aggregate(X0,X1,X2,X3,A))).

ast(add(body_literal(X0,X1,B));delete(body_literal(X0,X1,A)))
 :- ast(_replace_id(A,B)), ast(fact(body_literal(X0,X1,A));add(body_literal(X0,X1,A))).

ast(add(body_literals(X0,X1,B));delete(body_literals(X0,X1,A)))
 :- ast(_replace_id(A,B)), ast(fact(body_literals(X0,X1,A));add(body_literals(X0,X1,A))).

ast(add(head_aggregate_elements(X0,X1,B,X3));delete(head_aggregate_elements(X0,X1,A,X3)))
 :- ast(_replace_id(A,B)), ast(fact(head_aggregate_elements(X0,X1,A,X3));add(head_aggregate_elements(X0,X1,A,X3))).

ast(add(head_aggregate_elements(X0,X1,X2,B));delete(head_aggregate_elements(X0,X1,X2,A)))
 :- ast(_replace_id(A,B)), ast(fact(head_aggregate_elements(X0,X1,X2,A));add(head_aggregate_elements(X0,X1,X2,A))).

ast(add(head_aggregate(X0,B,X2,X3,X4));delete(head_aggregate(X0,A,X2,X3,X4)))
 :- ast(_replace_id(A,B)), ast(fact(head_aggregate(X0,A,X2,X3,X4));add(head_aggregate(X0,A,X2,X3,X4))).

ast(add(head_aggregate(X0,X1,X2,B,X4));delete(head_aggregate(X0,X1,X2,A,X4)))
 :- ast(_replace_id(A,B)), ast(fact(head_aggregate(X0,X1,X2,A,X4));add(head_aggregate(X0,X1,X2,A,X4))).

ast(add(head_aggregate(X0,X1,X2,X3,B));delete(head_aggregate(X0,X1,X2,X3,A)))
 :- ast(_replace_id(A,B)), ast(fact(head_aggregate(X0,X1,X2,X3,A));add(head_aggregate(X0,X1,X2,X3,A))).

ast(add(conditional_literals(X0,X1,B));delete(conditional_literals(X0,X1,A)))
 :- ast(_replace_id(A,B)), ast(fact(conditional_literals(X0,X1,A));add(conditional_literals(X0,X1,A))).

ast(add(disjunction(X0,B));delete(disjunction(X0,A)))
 :- ast(_replace_id(A,B)), ast(fact(disjunction(X0,A));add(disjunction(X0,A))).

ast(add(rule(X0,B,X2));delete(rule(X0,A,X2)))
 :- ast(_replace_id(A,B)), ast(fact(rule(X0,A,X2));add(rule(X0,A,X2))).

ast(add(rule(X0,X1,B));delete(rule(X0,X1,A)))
 :- ast(_replace_id(A,B)), ast(fact(rule(X0,X1,A));add(rule(X0,X1,A))).

ast(add(definition(X0,X1,B,X3));delete(definition(X0,X1,A,X3)))
 :- ast(_replace_id(A,B)), ast(fact(definition(X0,X1,A,X3));add(definition(X0,X1,A,X3))).

ast(add(show_term(X0,B,X2));delete(show_term(X0,A,X2)))
 :- ast(_replace_id(A,B)), ast(fact(show_term(X0,A,X2));add(show_term(X0,A,X2))).

ast(add(show_term(X0,X1,B));delete(show_term(X0,X1,A)))
 :- ast(_replace_id(A,B)), ast(fact(show_term(X0,X1,A));add(show_term(X0,X1,A))).

ast(add(minimize(X0,B,X2,X3,X4));delete(minimize(X0,A,X2,X3,X4)))
 :- ast(_replace_id(A,B)), ast(fact(minimize(X0,A,X2,X3,X4));add(minimize(X0,A,X2,X3,X4))).

ast(add(minimize(X0,X1,B,X3,X4));delete(minimize(X0,X1,A,X3,X4)))
 :- ast(_replace_id(A,B)), ast(fact(minimize(X0,X1,A,X3,X4));add(minimize(X0,X1,A,X3,X4))).

ast(add(minimize(X0,X1,X2,B,X4));delete(minimize(X0,X1,X2,A,X4)))
 :- ast(_replace_id(A,B)), ast(fact(minimize(X0,X1,X2,A,X4));add(minimize(X0,X1,X2,A,X4))).

ast(add(minimize(X0,X1,X2,X3,B));delete(minimize(X0,X1,X2,X3,A)))
 :- ast(_replace_id(A,B)), ast(fact(minimize(X0,X1,X2,X3,A));add(minimize(X0,X1,X2,X3,A))).

ast(add(statements(X0,X1,B));delete(statements(X0,X1,A)))
 :- ast(_replace_id(A,B)), ast(fact(statements(X0,X1,A));add(statements(X0,X1,A))).

ast(add(constants(X0,X1,B));delete(constants(X0,X1,A)))
 :- ast(_replace_id(A,B)), ast(fact(constants(X0,X1,A));add(constants(X0,X1,A))).

ast(add(program(X0,X1,B,X3));delete(program(X0,X1,A,X3)))
 :- ast(_replace_id(A,B)), ast(fact(program(X0,X1,A,X3));add(program(X0,X1,A,X3))).

ast(add(program(X0,X1,X2,B));delete(program(X0,X1,X2,A)))
 :- ast(_replace_id(A,B)), ast(fact(program(X0,X1,X2,A));add(program(X0,X1,X2,A))).

ast(add(external(X0,B,X2,X3));delete(external(X0,A,X2,X3)))
 :- ast(_replace_id(A,B)), ast(fact(external(X0,A,X2,X3));add(external(X0,A,X2,X3))).

ast(add(external(X0,X1,B,X3));delete(external(X0,X1,A,X3)))
 :- ast(_replace_id(A,B)), ast(fact(external(X0,X1,A,X3));add(external(X0,X1,A,X3))).

ast(add(edge(X0,B,X2,X3));delete(edge(X0,A,X2,X3)))
 :- ast(_replace_id(A,B)), ast(fact(edge(X0,A,X2,X3));add(edge(X0,A,X2,X3))).

ast(add(edge(X0,X1,B,X3));delete(edge(X0,X1,A,X3)))
 :- ast(_replace_id(A,B)), ast(fact(edge(X0,X1,A,X3));add(edge(X0,X1,A,X3))).

ast(add(edge(X0,X1,X2,B));delete(edge(X0,X1,X2,A)))
 :- ast(_replace_id(A,B)), ast(fact(edge(X0,X1,X2,A));add(edge(X0,X1,X2,A))).

ast(add(heuristic(X0,B,X2,X3,X4,X5));delete(heuristic(X0,A,X2,X3,X4,X5)))
 :- ast(_replace_id(A,B)), ast(fact(heuristic(X0,A,X2,X3,X4,X5));add(heuristic(X0,A,X2,X3,X4,X5))).

ast(add(heuristic(X0,X1,B,X3,X4,X5));delete(heuristic(X0,X1,A,X3,X4,X5)))
 :- ast(_replace_id(A,B)), ast(fact(heuristic(X0,X1,A,X3,X4,X5));add(heuristic(X0,X1,A,X3,X4,X5))).

ast(add(heuristic(X0,X1,X2,B,X4,X5));delete(heuristic(X0,X1,X2,A,X4,X5)))
 :- ast(_replace_id(A,B)), ast(fact(heuristic(X0,X1,X2,A,X4,X5));add(heuristic(X0,X1,X2,A,X4,X5))).

ast(add(heuristic(X0,X1,X2,X3,B,X5));delete(heuristic(X0,X1,X2,X3,A,X5)))
 :- ast(_replace_id(A,B)), ast(fact(heuristic(X0,X1,X2,X3,A,X5));add(heuristic(X0,X1,X2,X3,A,X5))).

ast(add(heuristic(X0,X1,X2,X3,X4,B));delete(heuristic(X0,X1,X2,X3,X4,A)))
 :- ast(_replace_id(A,B)), ast(fact(heuristic(X0,X1,X2,X3,X4,A));add(heuristic(X0,X1,X2,X3,X4,A))).

ast(add(project_atom(X0,B,X2));delete(project_atom(X0,A,X2)))
 :- ast(_replace_id(A,B)), ast(fact(project_atom(X0,A,X2));add(project_atom(X0,A,X2))).

ast(add(project_atom(X0,X1,B));delete(project_atom(X0,X1,A)))
 :- ast(_replace_id(A,B)), ast(fact(project_atom(X0,X1,A));add(project_atom(X0,X1,A))).

ast(add(theory_term_definitions(X0,X1,X2,B));delete(theory_term_definitions(X0,X1,X2,A)))
 :- ast(_replace_id(A,B)), ast(fact(theory_term_definitions(X0,X1,X2,A));add(theory_term_definitions(X0,X1,X2,A))).

ast(add(theory_guard_definition(X0,B,X2));delete(theory_guard_definition(X0,A,X2)))
 :- ast(_replace_id(A,B)), ast(fact(theory_guard_definition(X0,A,X2));add(theory_guard_definition(X0,A,X2))).

ast(add(theory_atom_definitions(X0,X1,X2,X3,X4,X5,B));delete(theory_atom_definitions(X0,X1,X2,X3,X4,X5,A)))
 :- ast(_replace_id(A,B)), ast(fact(theory_atom_definitions(X0,X1,X2,X3,X4,X5,A));add(theory_atom_definitions(X0,X1,X2,X3,X4,X5,A))).

ast(add(theory_definition(X0,X1,B,X3));delete(theory_definition(X0,X1,A,X3)))
 :- ast(_replace_id(A,B)), ast(fact(theory_definition(X0,X1,A,X3));add(theory_definition(X0,X1,A,X3))).

ast(add(theory_definition(X0,X1,X2,B));delete(theory_definition(X0,X1,X2,A)))
 :- ast(_replace_id(A,B)), ast(fact(theory_definition(X0,X1,X2,A));add(theory_definition(X0,X1,X2,A))).

ast(add(child(X0,B));delete(child(X0,A)))
  :- ast(_replace_id(A,B)), ast(fact(child(X0,A));add(child(X0,A))).