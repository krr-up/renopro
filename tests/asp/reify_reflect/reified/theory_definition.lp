% Reification of a theory definition for difference constraints:
%
% #theory dc {
%   constant { - : 0 , unary };
%   diff_term { - : 0 , binary, left };
%   domain_term { .. : 1 , binary, left };
%   show_term { / : 1 , binary, left };
%   &dom/0 : domain_term, {=} , diff_term, any;
%   &diff/0 : diff_term, { <=} , constant, any;
%   &show/0 : show_term, directive
% }.
program(0,"base",constants(1),statements(2)).
statements(2,0,theory_definition(3)).
% #theory dc {
theory_definition(3,dc,theory_term_definitions(4),theory_atom_definitions(9)).
%   constant { - : 0 , unary };
theory_term_definitions(4,0,constant,theory_operator_definitions(5)).
theory_operator_definitions(5,0,"-",0,unary).
%   diff_term { - : 0 , binary, left };
theory_term_definitions(4,1,diff_term,theory_operator_definitions(6)).
theory_operator_definitions(6,0,"-",0,binary_left).
%   domain_term { .. : 1 , binary, left };
theory_term_definitions(4,2,domain_term,theory_operator_definitions(7)).
theory_operator_definitions(7,0,"..",1,binary_left).
%   show_term { / : 1 , binary, left };
theory_term_definitions(4,3,show_term,theory_operator_definitions(8)).
theory_operator_definitions(8,0,"/",1,binary_left).
%   &dom/0 : domain_term, {=} , diff_term, any;
theory_atom_definitions(9,0,any,dom,0,domain_term,theory_guard_definition(10)).
theory_guard_definition(10,theory_operators(11),diff_term).
theory_operators(11,0,"=").
%   &diff/0 : diff_term, { <=} , constant, any;
theory_atom_definitions(9,1,any,diff,0,diff_term,theory_guard_definition(12)).
theory_guard_definition(12,theory_operators(13),constant).
theory_operators(13,0,"<=").
%   &show/0 : show_term, directive
theory_atom_definitions(9,2,directive,show,0,show_term,theory_guard_definition(14)).
% }.
