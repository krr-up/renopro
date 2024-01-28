% Add child relations for facts added via ast add.

ast(add(child(unary_operation(X0),Child)))
  :- ast(add(unary_operation(X0,X1,Child))).

ast(add(child(binary_operation(X0),Child)))
  :- ast(add(binary_operation(X0,X1,Child,X3))).

ast(add(child(binary_operation(X0),Child)))
  :- ast(add(binary_operation(X0,X1,X2,Child))).

ast(add(child(interval(X0),Child)))
  :- ast(add(interval(X0,Child,X2))).

ast(add(child(interval(X0),Child)))
  :- ast(add(interval(X0,X1,Child))).

ast(add(child(terms(X0),Child)))
  :- ast(add(terms(X0,X1,Child))).

ast(add(child(function(X0),Child)))
  :- ast(add(function(X0,X1,Child))).

ast(add(child(external_function(X0),Child)))
  :- ast(add(external_function(X0,X1,Child))).

ast(add(child(pool(X0),Child)))
  :- ast(add(pool(X0,Child))).

ast(add(child(theory_terms(X0),Child)))
  :- ast(add(theory_terms(X0,X1,Child))).

ast(add(child(theory_sequence(X0),Child)))
  :- ast(add(theory_sequence(X0,X1,Child))).

ast(add(child(theory_function(X0),Child)))
  :- ast(add(theory_function(X0,X1,Child))).

ast(add(child(theory_unparsed_term_elements(X0),Child)))
  :- ast(add(theory_unparsed_term_elements(X0,X1,Child,X3))).

ast(add(child(theory_unparsed_term_elements(X0),Child)))
  :- ast(add(theory_unparsed_term_elements(X0,X1,X2,Child))).

ast(add(child(theory_unparsed_term(X0),Child)))
  :- ast(add(theory_unparsed_term(X0,Child))).

ast(add(child(guard(X0),Child)))
  :- ast(add(guard(X0,X1,Child))).

ast(add(child(guards(X0),Child)))
  :- ast(add(guards(X0,X1,Child))).

ast(add(child(comparison(X0),Child)))
  :- ast(add(comparison(X0,Child,X2))).

ast(add(child(comparison(X0),Child)))
  :- ast(add(comparison(X0,X1,Child))).

ast(add(child(symbolic_atom(X0),Child)))
  :- ast(add(symbolic_atom(X0,Child))).

ast(add(child(literal(X0),Child)))
  :- ast(add(literal(X0,X1,Child))).

ast(add(child(literals(X0),Child)))
  :- ast(add(literals(X0,X1,Child))).

ast(add(child(conditional_literal(X0),Child)))
  :- ast(add(conditional_literal(X0,Child,X2))).

ast(add(child(conditional_literal(X0),Child)))
  :- ast(add(conditional_literal(X0,X1,Child))).

ast(add(child(aggregate_elements(X0),Child)))
  :- ast(add(aggregate_elements(X0,X1,Child))).

ast(add(child(aggregate(X0),Child)))
  :- ast(add(aggregate(X0,Child,X2,X3))).

ast(add(child(aggregate(X0),Child)))
  :- ast(add(aggregate(X0,X1,Child,X3))).

ast(add(child(aggregate(X0),Child)))
  :- ast(add(aggregate(X0,X1,X2,Child))).

ast(add(child(theory_atom_elements(X0),Child)))
  :- ast(add(theory_atom_elements(X0,X1,Child,X3))).

ast(add(child(theory_atom_elements(X0),Child)))
  :- ast(add(theory_atom_elements(X0,X1,X2,Child))).

ast(add(child(theory_guard(X0),Child)))
  :- ast(add(theory_guard(X0,X1,Child))).

ast(add(child(theory_atom(X0),Child)))
  :- ast(add(theory_atom(X0,Child,X2,X3))).

ast(add(child(theory_atom(X0),Child)))
  :- ast(add(theory_atom(X0,X1,Child,X3))).

ast(add(child(theory_atom(X0),Child)))
  :- ast(add(theory_atom(X0,X1,X2,Child))).

ast(add(child(body_aggregate_elements(X0),Child)))
  :- ast(add(body_aggregate_elements(X0,X1,Child,X3))).

ast(add(child(body_aggregate_elements(X0),Child)))
  :- ast(add(body_aggregate_elements(X0,X1,X2,Child))).

ast(add(child(body_aggregate(X0),Child)))
  :- ast(add(body_aggregate(X0,Child,X2,X3,X4))).

ast(add(child(body_aggregate(X0),Child)))
  :- ast(add(body_aggregate(X0,X1,X2,Child,X4))).

ast(add(child(body_aggregate(X0),Child)))
  :- ast(add(body_aggregate(X0,X1,X2,X3,Child))).

ast(add(child(body_literal(X0),Child)))
  :- ast(add(body_literal(X0,X1,Child))).

ast(add(child(body_literals(X0),Child)))
  :- ast(add(body_literals(X0,X1,Child))).

ast(add(child(head_aggregate_elements(X0),Child)))
  :- ast(add(head_aggregate_elements(X0,X1,Child,X3))).

ast(add(child(head_aggregate_elements(X0),Child)))
  :- ast(add(head_aggregate_elements(X0,X1,X2,Child))).

ast(add(child(head_aggregate(X0),Child)))
  :- ast(add(head_aggregate(X0,Child,X2,X3,X4))).

ast(add(child(head_aggregate(X0),Child)))
  :- ast(add(head_aggregate(X0,X1,X2,Child,X4))).

ast(add(child(head_aggregate(X0),Child)))
  :- ast(add(head_aggregate(X0,X1,X2,X3,Child))).

ast(add(child(conditional_literals(X0),Child)))
  :- ast(add(conditional_literals(X0,X1,Child))).

ast(add(child(disjunction(X0),Child)))
  :- ast(add(disjunction(X0,Child))).

ast(add(child(rule(X0),Child)))
  :- ast(add(rule(X0,Child,X2))).

ast(add(child(rule(X0),Child)))
  :- ast(add(rule(X0,X1,Child))).

ast(add(child(definition(X0),Child)))
  :- ast(add(definition(X0,X1,Child,X3))).

ast(add(child(show_term(X0),Child)))
  :- ast(add(show_term(X0,Child,X2))).

ast(add(child(show_term(X0),Child)))
  :- ast(add(show_term(X0,X1,Child))).

ast(add(child(minimize(X0),Child)))
  :- ast(add(minimize(X0,Child,X2,X3,X4))).

ast(add(child(minimize(X0),Child)))
  :- ast(add(minimize(X0,X1,Child,X3,X4))).

ast(add(child(minimize(X0),Child)))
  :- ast(add(minimize(X0,X1,X2,Child,X4))).

ast(add(child(minimize(X0),Child)))
  :- ast(add(minimize(X0,X1,X2,X3,Child))).

ast(add(child(statements(X0),Child)))
  :- ast(add(statements(X0,X1,Child))).

ast(add(child(constants(X0),Child)))
  :- ast(add(constants(X0,X1,Child))).

ast(add(child(program(X0),Child)))
  :- ast(add(program(X0,X1,Child,X3))).

ast(add(child(program(X0),Child)))
  :- ast(add(program(X0,X1,X2,Child))).

ast(add(child(external(X0),Child)))
  :- ast(add(external(X0,Child,X2,X3))).

ast(add(child(external(X0),Child)))
  :- ast(add(external(X0,X1,Child,X3))).

ast(add(child(edge(X0),Child)))
  :- ast(add(edge(X0,Child,X2,X3))).

ast(add(child(edge(X0),Child)))
  :- ast(add(edge(X0,X1,Child,X3))).

ast(add(child(edge(X0),Child)))
  :- ast(add(edge(X0,X1,X2,Child))).

ast(add(child(heuristic(X0),Child)))
  :- ast(add(heuristic(X0,Child,X2,X3,X4,X5))).

ast(add(child(heuristic(X0),Child)))
  :- ast(add(heuristic(X0,X1,Child,X3,X4,X5))).

ast(add(child(heuristic(X0),Child)))
  :- ast(add(heuristic(X0,X1,X2,Child,X4,X5))).

ast(add(child(heuristic(X0),Child)))
  :- ast(add(heuristic(X0,X1,X2,X3,Child,X5))).

ast(add(child(heuristic(X0),Child)))
  :- ast(add(heuristic(X0,X1,X2,X3,X4,Child))).

ast(add(child(project_atom(X0),Child)))
  :- ast(add(project_atom(X0,Child,X2))).

ast(add(child(project_atom(X0),Child)))
  :- ast(add(project_atom(X0,X1,Child))).

ast(add(child(theory_term_definitions(X0),Child)))
  :- ast(add(theory_term_definitions(X0,X1,X2,Child))).

ast(add(child(theory_guard_definition(X0),Child)))
  :- ast(add(theory_guard_definition(X0,Child,X2))).

ast(add(child(theory_atom_definitions(X0),Child)))
  :- ast(add(theory_atom_definitions(X0,X1,X2,X3,X4,X5,Child))).

ast(add(child(theory_definition(X0),Child)))
  :- ast(add(theory_definition(X0,X1,Child,X3))).

ast(add(child(theory_definition(X0),Child)))
  :- ast(add(theory_definition(X0,X1,X2,Child))).

