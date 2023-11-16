% child(X,Y) holds if the ast fact with identifier X has a child fact
% with identifier Y. Note that identifiers X, Y are "typed";
% that is, they are of the form ast_pred(id).

child(unary_operation(X),string(Y)) :- unary_operation(X,_,Y), string(Y,_).

child(unary_operation(X),number(Y)) :- unary_operation(X,_,Y), number(Y,_).

child(unary_operation(X),variable(Y)) :- unary_operation(X,_,Y), variable(Y,_).

child(unary_operation(X),unary_operation(Y)) :- unary_operation(X,_,Y), unary_operation(Y,_,_).

child(unary_operation(X),binary_operation(Y)) :- unary_operation(X,_,Y), binary_operation(Y,_,_,_).

child(unary_operation(X),interval(Y)) :- unary_operation(X,_,Y), interval(Y,_,_).

child(unary_operation(X),function(Y)) :- unary_operation(X,_,Y), function(Y,_,_).

child(unary_operation(X),external_function(Y)) :- unary_operation(X,_,Y), external_function(Y,_,_).

child(unary_operation(X),pool(Y)) :- unary_operation(X,_,Y), pool(Y,_).

child(binary_operation(X),string(Y)) :- binary_operation(X,_,Y,_), string(Y,_).

child(binary_operation(X),number(Y)) :- binary_operation(X,_,Y,_), number(Y,_).

child(binary_operation(X),variable(Y)) :- binary_operation(X,_,Y,_), variable(Y,_).

child(binary_operation(X),unary_operation(Y)) :- binary_operation(X,_,Y,_), unary_operation(Y,_,_).

child(binary_operation(X),binary_operation(Y)) :- binary_operation(X,_,Y,_), binary_operation(Y,_,_,_).

child(binary_operation(X),interval(Y)) :- binary_operation(X,_,Y,_), interval(Y,_,_).

child(binary_operation(X),function(Y)) :- binary_operation(X,_,Y,_), function(Y,_,_).

child(binary_operation(X),external_function(Y)) :- binary_operation(X,_,Y,_), external_function(Y,_,_).

child(binary_operation(X),pool(Y)) :- binary_operation(X,_,Y,_), pool(Y,_).

child(binary_operation(X),string(Y)) :- binary_operation(X,_,_,Y), string(Y,_).

child(binary_operation(X),number(Y)) :- binary_operation(X,_,_,Y), number(Y,_).

child(binary_operation(X),variable(Y)) :- binary_operation(X,_,_,Y), variable(Y,_).

child(binary_operation(X),unary_operation(Y)) :- binary_operation(X,_,_,Y), unary_operation(Y,_,_).

child(binary_operation(X),binary_operation(Y)) :- binary_operation(X,_,_,Y), binary_operation(Y,_,_,_).

child(binary_operation(X),interval(Y)) :- binary_operation(X,_,_,Y), interval(Y,_,_).

child(binary_operation(X),function(Y)) :- binary_operation(X,_,_,Y), function(Y,_,_).

child(binary_operation(X),external_function(Y)) :- binary_operation(X,_,_,Y), external_function(Y,_,_).

child(binary_operation(X),pool(Y)) :- binary_operation(X,_,_,Y), pool(Y,_).

child(interval(X),string(Y)) :- interval(X,Y,_), string(Y,_).

child(interval(X),number(Y)) :- interval(X,Y,_), number(Y,_).

child(interval(X),variable(Y)) :- interval(X,Y,_), variable(Y,_).

child(interval(X),unary_operation(Y)) :- interval(X,Y,_), unary_operation(Y,_,_).

child(interval(X),binary_operation(Y)) :- interval(X,Y,_), binary_operation(Y,_,_,_).

child(interval(X),interval(Y)) :- interval(X,Y,_), interval(Y,_,_).

child(interval(X),function(Y)) :- interval(X,Y,_), function(Y,_,_).

child(interval(X),external_function(Y)) :- interval(X,Y,_), external_function(Y,_,_).

child(interval(X),pool(Y)) :- interval(X,Y,_), pool(Y,_).

child(interval(X),string(Y)) :- interval(X,_,Y), string(Y,_).

child(interval(X),number(Y)) :- interval(X,_,Y), number(Y,_).

child(interval(X),variable(Y)) :- interval(X,_,Y), variable(Y,_).

child(interval(X),unary_operation(Y)) :- interval(X,_,Y), unary_operation(Y,_,_).

child(interval(X),binary_operation(Y)) :- interval(X,_,Y), binary_operation(Y,_,_,_).

child(interval(X),interval(Y)) :- interval(X,_,Y), interval(Y,_,_).

child(interval(X),function(Y)) :- interval(X,_,Y), function(Y,_,_).

child(interval(X),external_function(Y)) :- interval(X,_,Y), external_function(Y,_,_).

child(interval(X),pool(Y)) :- interval(X,_,Y), pool(Y,_).

child(terms(X),string(Y)) :- terms(X,_,Y), string(Y,_).

child(terms(X),number(Y)) :- terms(X,_,Y), number(Y,_).

child(terms(X),variable(Y)) :- terms(X,_,Y), variable(Y,_).

child(terms(X),unary_operation(Y)) :- terms(X,_,Y), unary_operation(Y,_,_).

child(terms(X),binary_operation(Y)) :- terms(X,_,Y), binary_operation(Y,_,_,_).

child(terms(X),interval(Y)) :- terms(X,_,Y), interval(Y,_,_).

child(terms(X),function(Y)) :- terms(X,_,Y), function(Y,_,_).

child(terms(X),external_function(Y)) :- terms(X,_,Y), external_function(Y,_,_).

child(terms(X),pool(Y)) :- terms(X,_,Y), pool(Y,_).

child(function(X),terms(Y)) :- function(X,_,Y), terms(Y,_,_).

child(external_function(X),terms(Y)) :- external_function(X,_,Y), terms(Y,_,_).

child(pool(X),terms(Y)) :- pool(X,Y), terms(Y,_,_).

child(theory_terms(X),string(Y)) :- theory_terms(X,_,Y), string(Y,_).

child(theory_terms(X),number(Y)) :- theory_terms(X,_,Y), number(Y,_).

child(theory_terms(X),function(Y)) :- theory_terms(X,_,Y), function(Y,_,_).

child(theory_terms(X),variable(Y)) :- theory_terms(X,_,Y), variable(Y,_).

child(theory_terms(X),theory_sequence(Y)) :- theory_terms(X,_,Y), theory_sequence(Y,_,_).

child(theory_terms(X),theory_function(Y)) :- theory_terms(X,_,Y), theory_function(Y,_,_).

child(theory_terms(X),theory_unparsed_term(Y)) :- theory_terms(X,_,Y), theory_unparsed_term(Y,_).

child(theory_sequence(X),theory_terms(Y)) :- theory_sequence(X,_,Y), theory_terms(Y,_,_).

child(theory_function(X),theory_terms(Y)) :- theory_function(X,_,Y), theory_terms(Y,_,_).

child(theory_unparsed_term_elements(X),theory_operators(Y)) :- theory_unparsed_term_elements(X,_,Y,_), theory_operators(Y,_,_).

child(theory_unparsed_term_elements(X),string(Y)) :- theory_unparsed_term_elements(X,_,_,Y), string(Y,_).

child(theory_unparsed_term_elements(X),number(Y)) :- theory_unparsed_term_elements(X,_,_,Y), number(Y,_).

child(theory_unparsed_term_elements(X),function(Y)) :- theory_unparsed_term_elements(X,_,_,Y), function(Y,_,_).

child(theory_unparsed_term_elements(X),variable(Y)) :- theory_unparsed_term_elements(X,_,_,Y), variable(Y,_).

child(theory_unparsed_term_elements(X),theory_sequence(Y)) :- theory_unparsed_term_elements(X,_,_,Y), theory_sequence(Y,_,_).

child(theory_unparsed_term_elements(X),theory_function(Y)) :- theory_unparsed_term_elements(X,_,_,Y), theory_function(Y,_,_).

child(theory_unparsed_term_elements(X),theory_unparsed_term(Y)) :- theory_unparsed_term_elements(X,_,_,Y), theory_unparsed_term(Y,_).

child(theory_unparsed_term(X),theory_unparsed_term_elements(Y)) :- theory_unparsed_term(X,Y), theory_unparsed_term_elements(Y,_,_,_).

child(guard(X),string(Y)) :- guard(X,_,Y), string(Y,_).

child(guard(X),number(Y)) :- guard(X,_,Y), number(Y,_).

child(guard(X),variable(Y)) :- guard(X,_,Y), variable(Y,_).

child(guard(X),unary_operation(Y)) :- guard(X,_,Y), unary_operation(Y,_,_).

child(guard(X),binary_operation(Y)) :- guard(X,_,Y), binary_operation(Y,_,_,_).

child(guard(X),interval(Y)) :- guard(X,_,Y), interval(Y,_,_).

child(guard(X),function(Y)) :- guard(X,_,Y), function(Y,_,_).

child(guard(X),external_function(Y)) :- guard(X,_,Y), external_function(Y,_,_).

child(guard(X),pool(Y)) :- guard(X,_,Y), pool(Y,_).

child(guards(X),guard(Y)) :- guards(X,_,Y), guard(Y,_,_).

child(comparison(X),string(Y)) :- comparison(X,Y,_), string(Y,_).

child(comparison(X),number(Y)) :- comparison(X,Y,_), number(Y,_).

child(comparison(X),variable(Y)) :- comparison(X,Y,_), variable(Y,_).

child(comparison(X),unary_operation(Y)) :- comparison(X,Y,_), unary_operation(Y,_,_).

child(comparison(X),binary_operation(Y)) :- comparison(X,Y,_), binary_operation(Y,_,_,_).

child(comparison(X),interval(Y)) :- comparison(X,Y,_), interval(Y,_,_).

child(comparison(X),function(Y)) :- comparison(X,Y,_), function(Y,_,_).

child(comparison(X),external_function(Y)) :- comparison(X,Y,_), external_function(Y,_,_).

child(comparison(X),pool(Y)) :- comparison(X,Y,_), pool(Y,_).

child(comparison(X),guards(Y)) :- comparison(X,_,Y), guards(Y,_,_).

child(symbolic_atom(X),function(Y)) :- symbolic_atom(X,Y), function(Y,_,_).

child(literal(X),symbolic_atom(Y)) :- literal(X,_,Y), symbolic_atom(Y,_).

child(literal(X),comparison(Y)) :- literal(X,_,Y), comparison(Y,_,_).

child(literal(X),boolean_constant(Y)) :- literal(X,_,Y), boolean_constant(Y,_).

child(literals(X),literal(Y)) :- literals(X,_,Y), literal(Y,_,_).

child(conditional_literal(X),literal(Y)) :- conditional_literal(X,Y,_), literal(Y,_,_).

child(conditional_literal(X),literals(Y)) :- conditional_literal(X,_,Y), literals(Y,_,_).

child(aggregate_elements(X),conditional_literal(Y)) :- aggregate_elements(X,_,Y), conditional_literal(Y,_,_).

child(aggregate(X),guard(Y)) :- aggregate(X,Y,_,_), guard(Y,_,_).

child(aggregate(X),aggregate_elements(Y)) :- aggregate(X,_,Y,_), aggregate_elements(Y,_,_).

child(aggregate(X),guard(Y)) :- aggregate(X,_,_,Y), guard(Y,_,_).

child(theory_atom_elements(X),theory_terms(Y)) :- theory_atom_elements(X,_,Y,_), theory_terms(Y,_,_).

child(theory_atom_elements(X),literals(Y)) :- theory_atom_elements(X,_,_,Y), literals(Y,_,_).

child(theory_guard(X),string(Y)) :- theory_guard(X,_,Y), string(Y,_).

child(theory_guard(X),number(Y)) :- theory_guard(X,_,Y), number(Y,_).

child(theory_guard(X),function(Y)) :- theory_guard(X,_,Y), function(Y,_,_).

child(theory_guard(X),variable(Y)) :- theory_guard(X,_,Y), variable(Y,_).

child(theory_guard(X),theory_sequence(Y)) :- theory_guard(X,_,Y), theory_sequence(Y,_,_).

child(theory_guard(X),theory_function(Y)) :- theory_guard(X,_,Y), theory_function(Y,_,_).

child(theory_guard(X),theory_unparsed_term(Y)) :- theory_guard(X,_,Y), theory_unparsed_term(Y,_).

child(theory_atom(X),function(Y)) :- theory_atom(X,Y,_,_), function(Y,_,_).

child(theory_atom(X),theory_atom_elements(Y)) :- theory_atom(X,_,Y,_), theory_atom_elements(Y,_,_,_).

child(theory_atom(X),theory_guard(Y)) :- theory_atom(X,_,_,Y), theory_guard(Y,_,_).

child(body_aggregate_elements(X),terms(Y)) :- body_aggregate_elements(X,_,Y,_), terms(Y,_,_).

child(body_aggregate_elements(X),literals(Y)) :- body_aggregate_elements(X,_,_,Y), literals(Y,_,_).

child(body_aggregate(X),guard(Y)) :- body_aggregate(X,Y,_,_,_), guard(Y,_,_).

child(body_aggregate(X),body_aggregate_elements(Y)) :- body_aggregate(X,_,_,Y,_), body_aggregate_elements(Y,_,_,_).

child(body_aggregate(X),guard(Y)) :- body_aggregate(X,_,_,_,Y), guard(Y,_,_).

child(body_literal(X),symbolic_atom(Y)) :- body_literal(X,_,Y), symbolic_atom(Y,_).

child(body_literal(X),comparison(Y)) :- body_literal(X,_,Y), comparison(Y,_,_).

child(body_literal(X),boolean_constant(Y)) :- body_literal(X,_,Y), boolean_constant(Y,_).

child(body_literal(X),aggregate(Y)) :- body_literal(X,_,Y), aggregate(Y,_,_,_).

child(body_literal(X),body_aggregate(Y)) :- body_literal(X,_,Y), body_aggregate(Y,_,_,_,_).

child(body_literal(X),theory_atom(Y)) :- body_literal(X,_,Y), theory_atom(Y,_,_,_).

child(body_literals(X),body_literal(Y)) :- body_literals(X,_,Y), body_literal(Y,_,_).

child(body_literals(X),conditional_literal(Y)) :- body_literals(X,_,Y), conditional_literal(Y,_,_).

child(head_aggregate_elements(X),terms(Y)) :- head_aggregate_elements(X,_,Y,_), terms(Y,_,_).

child(head_aggregate_elements(X),conditional_literal(Y)) :- head_aggregate_elements(X,_,_,Y), conditional_literal(Y,_,_).

child(head_aggregate(X),guard(Y)) :- head_aggregate(X,Y,_,_,_), guard(Y,_,_).

child(head_aggregate(X),head_aggregate_elements(Y)) :- head_aggregate(X,_,_,Y,_), head_aggregate_elements(Y,_,_,_).

child(head_aggregate(X),guard(Y)) :- head_aggregate(X,_,_,_,Y), guard(Y,_,_).

child(conditional_literals(X),conditional_literal(Y)) :- conditional_literals(X,_,Y), conditional_literal(Y,_,_).

child(disjunction(X),conditional_literals(Y)) :- disjunction(X,Y), conditional_literals(Y,_,_).

child(rule(X),literal(Y)) :- rule(X,Y,_), literal(Y,_,_).

child(rule(X),aggregate(Y)) :- rule(X,Y,_), aggregate(Y,_,_,_).

child(rule(X),head_aggregate(Y)) :- rule(X,Y,_), head_aggregate(Y,_,_,_,_).

child(rule(X),disjunction(Y)) :- rule(X,Y,_), disjunction(Y,_).

child(rule(X),theory_atom(Y)) :- rule(X,Y,_), theory_atom(Y,_,_,_).

child(rule(X),body_literals(Y)) :- rule(X,_,Y), body_literals(Y,_,_).

child(definition(X),string(Y)) :- definition(X,_,Y,_), string(Y,_).

child(definition(X),number(Y)) :- definition(X,_,Y,_), number(Y,_).

child(definition(X),variable(Y)) :- definition(X,_,Y,_), variable(Y,_).

child(definition(X),unary_operation(Y)) :- definition(X,_,Y,_), unary_operation(Y,_,_).

child(definition(X),binary_operation(Y)) :- definition(X,_,Y,_), binary_operation(Y,_,_,_).

child(definition(X),interval(Y)) :- definition(X,_,Y,_), interval(Y,_,_).

child(definition(X),function(Y)) :- definition(X,_,Y,_), function(Y,_,_).

child(definition(X),external_function(Y)) :- definition(X,_,Y,_), external_function(Y,_,_).

child(definition(X),pool(Y)) :- definition(X,_,Y,_), pool(Y,_).

child(show_term(X),string(Y)) :- show_term(X,Y,_), string(Y,_).

child(show_term(X),number(Y)) :- show_term(X,Y,_), number(Y,_).

child(show_term(X),variable(Y)) :- show_term(X,Y,_), variable(Y,_).

child(show_term(X),unary_operation(Y)) :- show_term(X,Y,_), unary_operation(Y,_,_).

child(show_term(X),binary_operation(Y)) :- show_term(X,Y,_), binary_operation(Y,_,_,_).

child(show_term(X),interval(Y)) :- show_term(X,Y,_), interval(Y,_,_).

child(show_term(X),function(Y)) :- show_term(X,Y,_), function(Y,_,_).

child(show_term(X),external_function(Y)) :- show_term(X,Y,_), external_function(Y,_,_).

child(show_term(X),pool(Y)) :- show_term(X,Y,_), pool(Y,_).

child(show_term(X),body_literals(Y)) :- show_term(X,_,Y), body_literals(Y,_,_).

child(minimize(X),string(Y)) :- minimize(X,Y,_,_,_), string(Y,_).

child(minimize(X),number(Y)) :- minimize(X,Y,_,_,_), number(Y,_).

child(minimize(X),variable(Y)) :- minimize(X,Y,_,_,_), variable(Y,_).

child(minimize(X),unary_operation(Y)) :- minimize(X,Y,_,_,_), unary_operation(Y,_,_).

child(minimize(X),binary_operation(Y)) :- minimize(X,Y,_,_,_), binary_operation(Y,_,_,_).

child(minimize(X),interval(Y)) :- minimize(X,Y,_,_,_), interval(Y,_,_).

child(minimize(X),function(Y)) :- minimize(X,Y,_,_,_), function(Y,_,_).

child(minimize(X),external_function(Y)) :- minimize(X,Y,_,_,_), external_function(Y,_,_).

child(minimize(X),pool(Y)) :- minimize(X,Y,_,_,_), pool(Y,_).

child(minimize(X),string(Y)) :- minimize(X,_,Y,_,_), string(Y,_).

child(minimize(X),number(Y)) :- minimize(X,_,Y,_,_), number(Y,_).

child(minimize(X),variable(Y)) :- minimize(X,_,Y,_,_), variable(Y,_).

child(minimize(X),unary_operation(Y)) :- minimize(X,_,Y,_,_), unary_operation(Y,_,_).

child(minimize(X),binary_operation(Y)) :- minimize(X,_,Y,_,_), binary_operation(Y,_,_,_).

child(minimize(X),interval(Y)) :- minimize(X,_,Y,_,_), interval(Y,_,_).

child(minimize(X),function(Y)) :- minimize(X,_,Y,_,_), function(Y,_,_).

child(minimize(X),external_function(Y)) :- minimize(X,_,Y,_,_), external_function(Y,_,_).

child(minimize(X),pool(Y)) :- minimize(X,_,Y,_,_), pool(Y,_).

child(minimize(X),terms(Y)) :- minimize(X,_,_,Y,_), terms(Y,_,_).

child(minimize(X),body_literals(Y)) :- minimize(X,_,_,_,Y), body_literals(Y,_,_).

child(statements(X),rule(Y)) :- statements(X,_,Y), rule(Y,_,_).

child(statements(X),definition(Y)) :- statements(X,_,Y), definition(Y,_,_,_).

child(statements(X),show_signature(Y)) :- statements(X,_,Y), show_signature(Y,_,_,_).

child(statements(X),defined(Y)) :- statements(X,_,Y), defined(Y,_,_,_).

child(statements(X),show_term(Y)) :- statements(X,_,Y), show_term(Y,_,_).

child(statements(X),minimize(Y)) :- statements(X,_,Y), minimize(Y,_,_,_,_).

child(statements(X),script(Y)) :- statements(X,_,Y), script(Y,_,_).

child(statements(X),external(Y)) :- statements(X,_,Y), external(Y,_,_,_).

child(statements(X),edge(Y)) :- statements(X,_,Y), edge(Y,_,_,_).

child(statements(X),heuristic(Y)) :- statements(X,_,Y), heuristic(Y,_,_,_,_,_).

child(statements(X),project_atom(Y)) :- statements(X,_,Y), project_atom(Y,_,_).

child(statements(X),project_signature(Y)) :- statements(X,_,Y), project_signature(Y,_,_,_).

child(statements(X),theory_definition(Y)) :- statements(X,_,Y), theory_definition(Y,_,_,_).

child(constants(X),function(Y)) :- constants(X,_,Y), function(Y,_,_).

child(program(X),constants(Y)) :- program(X,_,Y,_), constants(Y,_,_).

child(program(X),statements(Y)) :- program(X,_,_,Y), statements(Y,_,_).

child(external(X),symbolic_atom(Y)) :- external(X,Y,_,_), symbolic_atom(Y,_).

child(external(X),body_literals(Y)) :- external(X,_,Y,_), body_literals(Y,_,_).

child(edge(X),string(Y)) :- edge(X,Y,_,_), string(Y,_).

child(edge(X),number(Y)) :- edge(X,Y,_,_), number(Y,_).

child(edge(X),variable(Y)) :- edge(X,Y,_,_), variable(Y,_).

child(edge(X),unary_operation(Y)) :- edge(X,Y,_,_), unary_operation(Y,_,_).

child(edge(X),binary_operation(Y)) :- edge(X,Y,_,_), binary_operation(Y,_,_,_).

child(edge(X),interval(Y)) :- edge(X,Y,_,_), interval(Y,_,_).

child(edge(X),function(Y)) :- edge(X,Y,_,_), function(Y,_,_).

child(edge(X),external_function(Y)) :- edge(X,Y,_,_), external_function(Y,_,_).

child(edge(X),pool(Y)) :- edge(X,Y,_,_), pool(Y,_).

child(edge(X),string(Y)) :- edge(X,_,Y,_), string(Y,_).

child(edge(X),number(Y)) :- edge(X,_,Y,_), number(Y,_).

child(edge(X),variable(Y)) :- edge(X,_,Y,_), variable(Y,_).

child(edge(X),unary_operation(Y)) :- edge(X,_,Y,_), unary_operation(Y,_,_).

child(edge(X),binary_operation(Y)) :- edge(X,_,Y,_), binary_operation(Y,_,_,_).

child(edge(X),interval(Y)) :- edge(X,_,Y,_), interval(Y,_,_).

child(edge(X),function(Y)) :- edge(X,_,Y,_), function(Y,_,_).

child(edge(X),external_function(Y)) :- edge(X,_,Y,_), external_function(Y,_,_).

child(edge(X),pool(Y)) :- edge(X,_,Y,_), pool(Y,_).

child(edge(X),body_literals(Y)) :- edge(X,_,_,Y), body_literals(Y,_,_).

child(heuristic(X),symbolic_atom(Y)) :- heuristic(X,Y,_,_,_,_), symbolic_atom(Y,_).

child(heuristic(X),body_literals(Y)) :- heuristic(X,_,Y,_,_,_), body_literals(Y,_,_).

child(heuristic(X),string(Y)) :- heuristic(X,_,_,Y,_,_), string(Y,_).

child(heuristic(X),number(Y)) :- heuristic(X,_,_,Y,_,_), number(Y,_).

child(heuristic(X),variable(Y)) :- heuristic(X,_,_,Y,_,_), variable(Y,_).

child(heuristic(X),unary_operation(Y)) :- heuristic(X,_,_,Y,_,_), unary_operation(Y,_,_).

child(heuristic(X),binary_operation(Y)) :- heuristic(X,_,_,Y,_,_), binary_operation(Y,_,_,_).

child(heuristic(X),interval(Y)) :- heuristic(X,_,_,Y,_,_), interval(Y,_,_).

child(heuristic(X),function(Y)) :- heuristic(X,_,_,Y,_,_), function(Y,_,_).

child(heuristic(X),external_function(Y)) :- heuristic(X,_,_,Y,_,_), external_function(Y,_,_).

child(heuristic(X),pool(Y)) :- heuristic(X,_,_,Y,_,_), pool(Y,_).

child(heuristic(X),string(Y)) :- heuristic(X,_,_,_,Y,_), string(Y,_).

child(heuristic(X),number(Y)) :- heuristic(X,_,_,_,Y,_), number(Y,_).

child(heuristic(X),variable(Y)) :- heuristic(X,_,_,_,Y,_), variable(Y,_).

child(heuristic(X),unary_operation(Y)) :- heuristic(X,_,_,_,Y,_), unary_operation(Y,_,_).

child(heuristic(X),binary_operation(Y)) :- heuristic(X,_,_,_,Y,_), binary_operation(Y,_,_,_).

child(heuristic(X),interval(Y)) :- heuristic(X,_,_,_,Y,_), interval(Y,_,_).

child(heuristic(X),function(Y)) :- heuristic(X,_,_,_,Y,_), function(Y,_,_).

child(heuristic(X),external_function(Y)) :- heuristic(X,_,_,_,Y,_), external_function(Y,_,_).

child(heuristic(X),pool(Y)) :- heuristic(X,_,_,_,Y,_), pool(Y,_).

child(heuristic(X),string(Y)) :- heuristic(X,_,_,_,_,Y), string(Y,_).

child(heuristic(X),number(Y)) :- heuristic(X,_,_,_,_,Y), number(Y,_).

child(heuristic(X),variable(Y)) :- heuristic(X,_,_,_,_,Y), variable(Y,_).

child(heuristic(X),unary_operation(Y)) :- heuristic(X,_,_,_,_,Y), unary_operation(Y,_,_).

child(heuristic(X),binary_operation(Y)) :- heuristic(X,_,_,_,_,Y), binary_operation(Y,_,_,_).

child(heuristic(X),interval(Y)) :- heuristic(X,_,_,_,_,Y), interval(Y,_,_).

child(heuristic(X),function(Y)) :- heuristic(X,_,_,_,_,Y), function(Y,_,_).

child(heuristic(X),external_function(Y)) :- heuristic(X,_,_,_,_,Y), external_function(Y,_,_).

child(heuristic(X),pool(Y)) :- heuristic(X,_,_,_,_,Y), pool(Y,_).

child(project_atom(X),symbolic_atom(Y)) :- project_atom(X,Y,_), symbolic_atom(Y,_).

child(project_atom(X),body_literals(Y)) :- project_atom(X,_,Y), body_literals(Y,_,_).

child(theory_term_definitions(X),theory_operator_definitions(Y)) :- theory_term_definitions(X,_,_,Y), theory_operator_definitions(Y,_,_,_,_).

child(theory_guard_definition(X),theory_operators(Y)) :- theory_guard_definition(X,Y,_), theory_operators(Y,_,_).

child(theory_atom_definitions(X),theory_guard_definition(Y)) :- theory_atom_definitions(X,_,_,_,_,_,Y), theory_guard_definition(Y,_,_).

child(theory_definition(X),theory_term_definitions(Y)) :- theory_definition(X,_,Y,_), theory_term_definitions(Y,_,_,_).

child(theory_definition(X),theory_atom_definitions(Y)) :- theory_definition(X,_,_,Y), theory_atom_definitions(Y,_,_,_,_,_,_).

node(theory_definition(X)) :- theory_definition(X,_,_,Y).

