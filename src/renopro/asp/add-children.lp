% Add child relations for facts added via ast_operation add.

ast_operation(add(child(unary_operation(X0),Child)))
  :- ast_operation(add(unary_operation(X0,X1,Child))).

ast_operation(add(child(binary_operation(X0),Child)))
  :- ast_operation(add(binary_operation(X0,X1,Child,X3))).

ast_operation(add(child(binary_operation(X0),Child)))
  :- ast_operation(add(binary_operation(X0,X1,X2,Child))).

ast_operation(add(child(interval(X0),Child)))
  :- ast_operation(add(interval(X0,Child,X2))).

ast_operation(add(child(interval(X0),Child)))
  :- ast_operation(add(interval(X0,X1,Child))).

ast_operation(add(child(terms(X0),Child)))
  :- ast_operation(add(terms(X0,X1,Child))).

ast_operation(add(child(function(X0),Child)))
  :- ast_operation(add(function(X0,X1,Child))).

ast_operation(add(child(external_function(X0),Child)))
  :- ast_operation(add(external_function(X0,X1,Child))).

ast_operation(add(child(pool(X0),Child)))
  :- ast_operation(add(pool(X0,Child))).

ast_operation(add(child(theory_terms(X0),Child)))
  :- ast_operation(add(theory_terms(X0,X1,Child))).

ast_operation(add(child(theory_sequence(X0),Child)))
  :- ast_operation(add(theory_sequence(X0,X1,Child))).

ast_operation(add(child(theory_function(X0),Child)))
  :- ast_operation(add(theory_function(X0,X1,Child))).

ast_operation(add(child(theory_unparsed_term_elements(X0),Child)))
  :- ast_operation(add(theory_unparsed_term_elements(X0,X1,Child,X3))).

ast_operation(add(child(theory_unparsed_term_elements(X0),Child)))
  :- ast_operation(add(theory_unparsed_term_elements(X0,X1,X2,Child))).

ast_operation(add(child(theory_unparsed_term(X0),Child)))
  :- ast_operation(add(theory_unparsed_term(X0,Child))).

ast_operation(add(child(guard(X0),Child)))
  :- ast_operation(add(guard(X0,X1,Child))).

ast_operation(add(child(guards(X0),Child)))
  :- ast_operation(add(guards(X0,X1,Child))).

ast_operation(add(child(comparison(X0),Child)))
  :- ast_operation(add(comparison(X0,Child,X2))).

ast_operation(add(child(comparison(X0),Child)))
  :- ast_operation(add(comparison(X0,X1,Child))).

ast_operation(add(child(symbolic_atom(X0),Child)))
  :- ast_operation(add(symbolic_atom(X0,Child))).

ast_operation(add(child(literal(X0),Child)))
  :- ast_operation(add(literal(X0,X1,Child))).

ast_operation(add(child(literals(X0),Child)))
  :- ast_operation(add(literals(X0,X1,Child))).

ast_operation(add(child(conditional_literal(X0),Child)))
  :- ast_operation(add(conditional_literal(X0,Child,X2))).

ast_operation(add(child(conditional_literal(X0),Child)))
  :- ast_operation(add(conditional_literal(X0,X1,Child))).

ast_operation(add(child(aggregate_elements(X0),Child)))
  :- ast_operation(add(aggregate_elements(X0,X1,Child))).

ast_operation(add(child(aggregate(X0),Child)))
  :- ast_operation(add(aggregate(X0,Child,X2,X3))).

ast_operation(add(child(aggregate(X0),Child)))
  :- ast_operation(add(aggregate(X0,X1,Child,X3))).

ast_operation(add(child(aggregate(X0),Child)))
  :- ast_operation(add(aggregate(X0,X1,X2,Child))).

ast_operation(add(child(theory_atom_elements(X0),Child)))
  :- ast_operation(add(theory_atom_elements(X0,X1,Child,X3))).

ast_operation(add(child(theory_atom_elements(X0),Child)))
  :- ast_operation(add(theory_atom_elements(X0,X1,X2,Child))).

ast_operation(add(child(theory_guard(X0),Child)))
  :- ast_operation(add(theory_guard(X0,X1,Child))).

ast_operation(add(child(theory_atom(X0),Child)))
  :- ast_operation(add(theory_atom(X0,Child,X2,X3))).

ast_operation(add(child(theory_atom(X0),Child)))
  :- ast_operation(add(theory_atom(X0,X1,Child,X3))).

ast_operation(add(child(theory_atom(X0),Child)))
  :- ast_operation(add(theory_atom(X0,X1,X2,Child))).

ast_operation(add(child(body_aggregate_elements(X0),Child)))
  :- ast_operation(add(body_aggregate_elements(X0,X1,Child,X3))).

ast_operation(add(child(body_aggregate_elements(X0),Child)))
  :- ast_operation(add(body_aggregate_elements(X0,X1,X2,Child))).

ast_operation(add(child(body_aggregate(X0),Child)))
  :- ast_operation(add(body_aggregate(X0,Child,X2,X3,X4))).

ast_operation(add(child(body_aggregate(X0),Child)))
  :- ast_operation(add(body_aggregate(X0,X1,X2,Child,X4))).

ast_operation(add(child(body_aggregate(X0),Child)))
  :- ast_operation(add(body_aggregate(X0,X1,X2,X3,Child))).

ast_operation(add(child(body_literal(X0),Child)))
  :- ast_operation(add(body_literal(X0,X1,Child))).

ast_operation(add(child(body_literals(X0),Child)))
  :- ast_operation(add(body_literals(X0,X1,Child))).

ast_operation(add(child(head_aggregate_elements(X0),Child)))
  :- ast_operation(add(head_aggregate_elements(X0,X1,Child,X3))).

ast_operation(add(child(head_aggregate_elements(X0),Child)))
  :- ast_operation(add(head_aggregate_elements(X0,X1,X2,Child))).

ast_operation(add(child(head_aggregate(X0),Child)))
  :- ast_operation(add(head_aggregate(X0,Child,X2,X3,X4))).

ast_operation(add(child(head_aggregate(X0),Child)))
  :- ast_operation(add(head_aggregate(X0,X1,X2,Child,X4))).

ast_operation(add(child(head_aggregate(X0),Child)))
  :- ast_operation(add(head_aggregate(X0,X1,X2,X3,Child))).

ast_operation(add(child(conditional_literals(X0),Child)))
  :- ast_operation(add(conditional_literals(X0,X1,Child))).

ast_operation(add(child(disjunction(X0),Child)))
  :- ast_operation(add(disjunction(X0,Child))).

ast_operation(add(child(rule(X0),Child)))
  :- ast_operation(add(rule(X0,Child,X2))).

ast_operation(add(child(rule(X0),Child)))
  :- ast_operation(add(rule(X0,X1,Child))).

ast_operation(add(child(definition(X0),Child)))
  :- ast_operation(add(definition(X0,X1,Child,X3))).

ast_operation(add(child(show_term(X0),Child)))
  :- ast_operation(add(show_term(X0,Child,X2))).

ast_operation(add(child(show_term(X0),Child)))
  :- ast_operation(add(show_term(X0,X1,Child))).

ast_operation(add(child(minimize(X0),Child)))
  :- ast_operation(add(minimize(X0,Child,X2,X3,X4))).

ast_operation(add(child(minimize(X0),Child)))
  :- ast_operation(add(minimize(X0,X1,Child,X3,X4))).

ast_operation(add(child(minimize(X0),Child)))
  :- ast_operation(add(minimize(X0,X1,X2,Child,X4))).

ast_operation(add(child(minimize(X0),Child)))
  :- ast_operation(add(minimize(X0,X1,X2,X3,Child))).

ast_operation(add(child(statements(X0),Child)))
  :- ast_operation(add(statements(X0,X1,Child))).

ast_operation(add(child(constants(X0),Child)))
  :- ast_operation(add(constants(X0,X1,Child))).

ast_operation(add(child(program(X0),Child)))
  :- ast_operation(add(program(X0,X1,Child,X3))).

ast_operation(add(child(program(X0),Child)))
  :- ast_operation(add(program(X0,X1,X2,Child))).

ast_operation(add(child(external(X0),Child)))
  :- ast_operation(add(external(X0,Child,X2,X3))).

ast_operation(add(child(external(X0),Child)))
  :- ast_operation(add(external(X0,X1,Child,X3))).

ast_operation(add(child(edge(X0),Child)))
  :- ast_operation(add(edge(X0,Child,X2,X3))).

ast_operation(add(child(edge(X0),Child)))
  :- ast_operation(add(edge(X0,X1,Child,X3))).

ast_operation(add(child(edge(X0),Child)))
  :- ast_operation(add(edge(X0,X1,X2,Child))).

ast_operation(add(child(heuristic(X0),Child)))
  :- ast_operation(add(heuristic(X0,Child,X2,X3,X4,X5))).

ast_operation(add(child(heuristic(X0),Child)))
  :- ast_operation(add(heuristic(X0,X1,Child,X3,X4,X5))).

ast_operation(add(child(heuristic(X0),Child)))
  :- ast_operation(add(heuristic(X0,X1,X2,Child,X4,X5))).

ast_operation(add(child(heuristic(X0),Child)))
  :- ast_operation(add(heuristic(X0,X1,X2,X3,Child,X5))).

ast_operation(add(child(heuristic(X0),Child)))
  :- ast_operation(add(heuristic(X0,X1,X2,X3,X4,Child))).

ast_operation(add(child(project_atom(X0),Child)))
  :- ast_operation(add(project_atom(X0,Child,X2))).

ast_operation(add(child(project_atom(X0),Child)))
  :- ast_operation(add(project_atom(X0,X1,Child))).

ast_operation(add(child(theory_term_definitions(X0),Child)))
  :- ast_operation(add(theory_term_definitions(X0,X1,X2,Child))).

ast_operation(add(child(theory_guard_definition(X0),Child)))
  :- ast_operation(add(theory_guard_definition(X0,Child,X2))).

ast_operation(add(child(theory_atom_definitions(X0),Child)))
  :- ast_operation(add(theory_atom_definitions(X0,X1,X2,X3,X4,X5,Child))).

ast_operation(add(child(theory_definition(X0),Child)))
  :- ast_operation(add(theory_definition(X0,X1,Child,X3))).

ast_operation(add(child(theory_definition(X0),Child)))
  :- ast_operation(add(theory_definition(X0,X1,X2,Child))).

