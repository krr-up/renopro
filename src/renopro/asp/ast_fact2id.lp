% map ast facts to their identifiers.

ast_fact2id(string(X0,X1),string(X0))
  :- ast(fact(string(X0,X1));add(string(X0,X1));delete(string(X0,X1))).

ast_fact2id(number(X0,X1),number(X0))
  :- ast(fact(number(X0,X1));add(number(X0,X1));delete(number(X0,X1))).

ast_fact2id(variable(X0,X1),variable(X0))
  :- ast(fact(variable(X0,X1));add(variable(X0,X1));delete(variable(X0,X1))).

ast_fact2id(unary_operation(X0,X1,X2),unary_operation(X0))
  :- ast(fact(unary_operation(X0,X1,X2));add(unary_operation(X0,X1,X2));delete(unary_operation(X0,X1,X2))).

ast_fact2id(binary_operation(X0,X1,X2,X3),binary_operation(X0))
  :- ast(fact(binary_operation(X0,X1,X2,X3));add(binary_operation(X0,X1,X2,X3));delete(binary_operation(X0,X1,X2,X3))).

ast_fact2id(interval(X0,X1,X2),interval(X0))
  :- ast(fact(interval(X0,X1,X2));add(interval(X0,X1,X2));delete(interval(X0,X1,X2))).

ast_fact2id(terms(X0,X1,X2),terms(X0))
  :- ast(fact(terms(X0,X1,X2));add(terms(X0,X1,X2));delete(terms(X0,X1,X2))).

ast_fact2id(function(X0,X1,X2),function(X0))
  :- ast(fact(function(X0,X1,X2));add(function(X0,X1,X2));delete(function(X0,X1,X2))).

ast_fact2id(external_function(X0,X1,X2),external_function(X0))
  :- ast(fact(external_function(X0,X1,X2));add(external_function(X0,X1,X2));delete(external_function(X0,X1,X2))).

ast_fact2id(pool(X0,X1),pool(X0))
  :- ast(fact(pool(X0,X1));add(pool(X0,X1));delete(pool(X0,X1))).

ast_fact2id(theory_terms(X0,X1,X2),theory_terms(X0))
  :- ast(fact(theory_terms(X0,X1,X2));add(theory_terms(X0,X1,X2));delete(theory_terms(X0,X1,X2))).

ast_fact2id(theory_sequence(X0,X1,X2),theory_sequence(X0))
  :- ast(fact(theory_sequence(X0,X1,X2));add(theory_sequence(X0,X1,X2));delete(theory_sequence(X0,X1,X2))).

ast_fact2id(theory_function(X0,X1,X2),theory_function(X0))
  :- ast(fact(theory_function(X0,X1,X2));add(theory_function(X0,X1,X2));delete(theory_function(X0,X1,X2))).

ast_fact2id(theory_operators(X0,X1,X2),theory_operators(X0))
  :- ast(fact(theory_operators(X0,X1,X2));add(theory_operators(X0,X1,X2));delete(theory_operators(X0,X1,X2))).

ast_fact2id(theory_unparsed_term_elements(X0,X1,X2,X3),theory_unparsed_term_elements(X0))
  :- ast(fact(theory_unparsed_term_elements(X0,X1,X2,X3));add(theory_unparsed_term_elements(X0,X1,X2,X3));delete(theory_unparsed_term_elements(X0,X1,X2,X3))).

ast_fact2id(theory_unparsed_term(X0,X1),theory_unparsed_term(X0))
  :- ast(fact(theory_unparsed_term(X0,X1));add(theory_unparsed_term(X0,X1));delete(theory_unparsed_term(X0,X1))).

ast_fact2id(guard(X0,X1,X2),guard(X0))
  :- ast(fact(guard(X0,X1,X2));add(guard(X0,X1,X2));delete(guard(X0,X1,X2))).

ast_fact2id(guards(X0,X1,X2),guards(X0))
  :- ast(fact(guards(X0,X1,X2));add(guards(X0,X1,X2));delete(guards(X0,X1,X2))).

ast_fact2id(comparison(X0,X1,X2),comparison(X0))
  :- ast(fact(comparison(X0,X1,X2));add(comparison(X0,X1,X2));delete(comparison(X0,X1,X2))).

ast_fact2id(boolean_constant(X0,X1),boolean_constant(X0))
  :- ast(fact(boolean_constant(X0,X1));add(boolean_constant(X0,X1));delete(boolean_constant(X0,X1))).

ast_fact2id(symbolic_atom(X0,X1),symbolic_atom(X0))
  :- ast(fact(symbolic_atom(X0,X1));add(symbolic_atom(X0,X1));delete(symbolic_atom(X0,X1))).

ast_fact2id(literal(X0,X1,X2),literal(X0))
  :- ast(fact(literal(X0,X1,X2));add(literal(X0,X1,X2));delete(literal(X0,X1,X2))).

ast_fact2id(literals(X0,X1,X2),literals(X0))
  :- ast(fact(literals(X0,X1,X2));add(literals(X0,X1,X2));delete(literals(X0,X1,X2))).

ast_fact2id(conditional_literal(X0,X1,X2),conditional_literal(X0))
  :- ast(fact(conditional_literal(X0,X1,X2));add(conditional_literal(X0,X1,X2));delete(conditional_literal(X0,X1,X2))).

ast_fact2id(aggregate_elements(X0,X1,X2),aggregate_elements(X0))
  :- ast(fact(aggregate_elements(X0,X1,X2));add(aggregate_elements(X0,X1,X2));delete(aggregate_elements(X0,X1,X2))).

ast_fact2id(aggregate(X0,X1,X2,X3),aggregate(X0))
  :- ast(fact(aggregate(X0,X1,X2,X3));add(aggregate(X0,X1,X2,X3));delete(aggregate(X0,X1,X2,X3))).

ast_fact2id(theory_atom_elements(X0,X1,X2,X3),theory_atom_elements(X0))
  :- ast(fact(theory_atom_elements(X0,X1,X2,X3));add(theory_atom_elements(X0,X1,X2,X3));delete(theory_atom_elements(X0,X1,X2,X3))).

ast_fact2id(theory_guard(X0,X1,X2),theory_guard(X0))
  :- ast(fact(theory_guard(X0,X1,X2));add(theory_guard(X0,X1,X2));delete(theory_guard(X0,X1,X2))).

ast_fact2id(theory_atom(X0,X1,X2,X3),theory_atom(X0))
  :- ast(fact(theory_atom(X0,X1,X2,X3));add(theory_atom(X0,X1,X2,X3));delete(theory_atom(X0,X1,X2,X3))).

ast_fact2id(body_aggregate_elements(X0,X1,X2,X3),body_aggregate_elements(X0))
  :- ast(fact(body_aggregate_elements(X0,X1,X2,X3));add(body_aggregate_elements(X0,X1,X2,X3));delete(body_aggregate_elements(X0,X1,X2,X3))).

ast_fact2id(body_aggregate(X0,X1,X2,X3,X4),body_aggregate(X0))
  :- ast(fact(body_aggregate(X0,X1,X2,X3,X4));add(body_aggregate(X0,X1,X2,X3,X4));delete(body_aggregate(X0,X1,X2,X3,X4))).

ast_fact2id(body_literal(X0,X1,X2),body_literal(X0))
  :- ast(fact(body_literal(X0,X1,X2));add(body_literal(X0,X1,X2));delete(body_literal(X0,X1,X2))).

ast_fact2id(body_literals(X0,X1,X2),body_literals(X0))
  :- ast(fact(body_literals(X0,X1,X2));add(body_literals(X0,X1,X2));delete(body_literals(X0,X1,X2))).

ast_fact2id(head_aggregate_elements(X0,X1,X2,X3),head_aggregate_elements(X0))
  :- ast(fact(head_aggregate_elements(X0,X1,X2,X3));add(head_aggregate_elements(X0,X1,X2,X3));delete(head_aggregate_elements(X0,X1,X2,X3))).

ast_fact2id(head_aggregate(X0,X1,X2,X3,X4),head_aggregate(X0))
  :- ast(fact(head_aggregate(X0,X1,X2,X3,X4));add(head_aggregate(X0,X1,X2,X3,X4));delete(head_aggregate(X0,X1,X2,X3,X4))).

ast_fact2id(conditional_literals(X0,X1,X2),conditional_literals(X0))
  :- ast(fact(conditional_literals(X0,X1,X2));add(conditional_literals(X0,X1,X2));delete(conditional_literals(X0,X1,X2))).

ast_fact2id(disjunction(X0,X1),disjunction(X0))
  :- ast(fact(disjunction(X0,X1));add(disjunction(X0,X1));delete(disjunction(X0,X1))).

ast_fact2id(rule(X0,X1,X2),rule(X0))
  :- ast(fact(rule(X0,X1,X2));add(rule(X0,X1,X2));delete(rule(X0,X1,X2))).

ast_fact2id(definition(X0,X1,X2,X3),definition(X0))
  :- ast(fact(definition(X0,X1,X2,X3));add(definition(X0,X1,X2,X3));delete(definition(X0,X1,X2,X3))).

ast_fact2id(show_signature(X0,X1,X2,X3),show_signature(X0))
  :- ast(fact(show_signature(X0,X1,X2,X3));add(show_signature(X0,X1,X2,X3));delete(show_signature(X0,X1,X2,X3))).

ast_fact2id(defined(X0,X1,X2,X3),defined(X0))
  :- ast(fact(defined(X0,X1,X2,X3));add(defined(X0,X1,X2,X3));delete(defined(X0,X1,X2,X3))).

ast_fact2id(show_term(X0,X1,X2),show_term(X0))
  :- ast(fact(show_term(X0,X1,X2));add(show_term(X0,X1,X2));delete(show_term(X0,X1,X2))).

ast_fact2id(minimize(X0,X1,X2,X3,X4),minimize(X0))
  :- ast(fact(minimize(X0,X1,X2,X3,X4));add(minimize(X0,X1,X2,X3,X4));delete(minimize(X0,X1,X2,X3,X4))).

ast_fact2id(script(X0,X1,X2),script(X0))
  :- ast(fact(script(X0,X1,X2));add(script(X0,X1,X2));delete(script(X0,X1,X2))).

ast_fact2id(statements(X0,X1,X2),statements(X0))
  :- ast(fact(statements(X0,X1,X2));add(statements(X0,X1,X2));delete(statements(X0,X1,X2))).

ast_fact2id(constants(X0,X1,X2),constants(X0))
  :- ast(fact(constants(X0,X1,X2));add(constants(X0,X1,X2));delete(constants(X0,X1,X2))).

ast_fact2id(program(X0,X1,X2,X3),program(X0))
  :- ast(fact(program(X0,X1,X2,X3));add(program(X0,X1,X2,X3));delete(program(X0,X1,X2,X3))).

ast_fact2id(external(X0,X1,X2,X3),external(X0))
  :- ast(fact(external(X0,X1,X2,X3));add(external(X0,X1,X2,X3));delete(external(X0,X1,X2,X3))).

ast_fact2id(edge(X0,X1,X2,X3),edge(X0))
  :- ast(fact(edge(X0,X1,X2,X3));add(edge(X0,X1,X2,X3));delete(edge(X0,X1,X2,X3))).

ast_fact2id(heuristic(X0,X1,X2,X3,X4,X5),heuristic(X0))
  :- ast(fact(heuristic(X0,X1,X2,X3,X4,X5));add(heuristic(X0,X1,X2,X3,X4,X5));delete(heuristic(X0,X1,X2,X3,X4,X5))).

ast_fact2id(project_atom(X0,X1,X2),project_atom(X0))
  :- ast(fact(project_atom(X0,X1,X2));add(project_atom(X0,X1,X2));delete(project_atom(X0,X1,X2))).

ast_fact2id(project_signature(X0,X1,X2,X3),project_signature(X0))
  :- ast(fact(project_signature(X0,X1,X2,X3));add(project_signature(X0,X1,X2,X3));delete(project_signature(X0,X1,X2,X3))).

ast_fact2id(theory_operator_definitions(X0,X1,X2,X3,X4),theory_operator_definitions(X0))
  :- ast(fact(theory_operator_definitions(X0,X1,X2,X3,X4));add(theory_operator_definitions(X0,X1,X2,X3,X4));delete(theory_operator_definitions(X0,X1,X2,X3,X4))).

ast_fact2id(theory_term_definitions(X0,X1,X2,X3),theory_term_definitions(X0))
  :- ast(fact(theory_term_definitions(X0,X1,X2,X3));add(theory_term_definitions(X0,X1,X2,X3));delete(theory_term_definitions(X0,X1,X2,X3))).

ast_fact2id(theory_guard_definition(X0,X1,X2),theory_guard_definition(X0))
  :- ast(fact(theory_guard_definition(X0,X1,X2));add(theory_guard_definition(X0,X1,X2));delete(theory_guard_definition(X0,X1,X2))).

ast_fact2id(theory_atom_definitions(X0,X1,X2,X3,X4,X5,X6),theory_atom_definitions(X0))
  :- ast(fact(theory_atom_definitions(X0,X1,X2,X3,X4,X5,X6));add(theory_atom_definitions(X0,X1,X2,X3,X4,X5,X6));delete(theory_atom_definitions(X0,X1,X2,X3,X4,X5,X6))).

ast_fact2id(theory_definition(X0,X1,X2,X3),theory_definition(X0))
  :- ast(fact(theory_definition(X0,X1,X2,X3));add(theory_definition(X0,X1,X2,X3));delete(theory_definition(X0,X1,X2,X3))).

ast_fact2id(comment(X0,X1,X2),comment(X0))
  :- ast(fact(comment(X0,X1,X2));add(comment(X0,X1,X2));delete(comment(X0,X1,X2))).

ast_fact2id(location(Id,Begin,End),Id) :- ast(fact(location(Id,Begin,End));add(location(Id,Begin,End));delete(location(Id,Begin,End))).
