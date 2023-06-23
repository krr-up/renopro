"""Module implementing reification and de-reification of non-ground programs"""
import sys
from functools import singledispatchmethod
import logging
from typing import Iterable, Sequence, Union, Optional
from pathlib import Path

import clingo
from clingo import ast, symbol
from clingo.ast import (AST, ASTType, Location, Position,
                        parse_string, parse_files, BinaryOperator)
from clingo.symbol import Symbol, SymbolType
from clorm.clingo import Control
from clorm import (FactBase, control_add_facts,
                   SymbolPredicateUnifier, parse_fact_string,
                   UnifierNoMatchError, parse_fact_files)

import renopro.predicates as preds


class ChildQueryError(Exception):
    pass


class ChildrenQueryError(Exception):
    pass


logger = logging.getLogger(__name__)

DUMMY_LOC = Location(Position("<string>", 1, 1), Position("<string>", 1, 1))


class ReifiedAST:
    """Class for converting between reified and non-reified
    representation of ASP programs."""

    def __init__(self):
        self._reified = FactBase()
        self._program_statements = list()
        self._id_counter = -1

    def _new_id(self) -> int:
        self._id_counter += 1
        return self._id_counter

    def add_reified_facts(self, reified_facts:
                          Iterable[preds.AST_Predicate]) -> None:
        """Add factbase containing reified facts into internal factbase."""
        self._reified.update(reified_facts)

    def add_reified_string(self, reified_string: str) -> None:
        """Add string of reified facts into internal factbase."""
        facts = parse_fact_string(reified_string, unifier=preds.AST_Facts,
                                  raise_nomatch=True, raise_nonfact=True)
        self._reified.update(facts)

    def add_reified_files(self, reified_files: Sequence[Path]) -> None:
        """Add files containing reified facts into internal factbase."""
        reified_files = [str(f) for f in reified_files]
        facts = parse_fact_files(reified_files, unifier=preds.AST_Facts,
                                 raise_nomatch=True, raise_nonfact=True)
        self._reified.update(facts)

    def reify_string(self, prog_str: str) -> None:
        """Reify input program string, adding reified facts to the
        internal factbase."""
        parse_string(prog_str, self._reify_ast)

    def reify_files(self, files: Sequence[Path]) -> None:
        """Reify input program files, adding reified facts to the
        internal factbase."""
        for f in files:
            if not f.is_file():
                raise IOError(f"File {f} does not exist.")
        files = [str(f) for f in files]
        self._program_statements = list()
        parse_files(files, self._reify_ast)

    @property
    def program_string(self) -> str:
        return '\n'.join([str(statement) for statement
                          in self._program_statements])

    @property
    def program_ast(self) -> str:
        return self._program_statements

    @property
    def reified_facts(self) -> FactBase:
        return self._reified

    @property
    def reified_string(self) -> str:
        return self._reified.asp_str()

    @property
    def reified_string_doc(self) -> str:
        return self._reified.asp_str(commented=True)

    @staticmethod
    def dispatch_on_node_type(meth):
        """"Dispatch on node.ast_type if node is of type AST or
        node.type if node is of type Symbol.

        """
        registry = {}

        def dispatch(value):
            try:
                return registry[value]
            except KeyError:
                return meth

        def register(value, func=None):
            if func is None:
                return lambda f: register(value, f)

            registry[value] = func
            return func

        def wrapper(self, node, *args, **kw):
            if isinstance(node, AST):
                return dispatch(node.ast_type)(self, node, *args, **kw)
            elif isinstance(node, Symbol):
                return dispatch(node.type)(self, node, *args, **kw)
            else:
                raise RuntimeError(f"Nodes should be of type AST or Symbol, got: {type(node)}")

        wrapper.register = register
        wrapper.dispatch = dispatch
        wrapper.registry = registry

        return wrapper

    @dispatch_on_node_type
    def _reify_ast(self, node):
        """Reify the input ast node by adding it's clorm fact
        representation to the internal fact base.

        """
        raise NotImplementedError(
            f"Reification not implemented for nodes of type: {node.ast_type.name}."
        )

    @_reify_ast.register(ASTType.Program)
    def _reify_program(self, node):
        self._program_statements.append(node)
        program = preds.Program(
            name=node.name,
            parameters=preds.Constant_Tuple1(),
            statements=preds.Statement_Tuple1()
        )
        self._reified.add(program)
        self._statement_tup_id = program.statements.id
        self._statement_pos = 0
        for pos, param in enumerate(node.parameters):
            self._reified.add(preds.Constant_Tuple(
                id=program.parameters.id,
                position=pos,
                # note: this id refers to the clingo.ast.Id.id attribute
                element=param.id))
        return

    @_reify_ast.register(ASTType.External)
    def _reify_external(self, node):
        self._program_statements.append(node)
        ext_type = node.external_type.symbol.name
        external1 = preds.External1()
        external = preds.External(
            id=external1.id,
            atom=self._reify_ast(node.atom),
            body=preds.Literal_Tuple1(),
            external_type=ext_type
        )
        self._reified.add(external)
        statement_tup = preds.Statement_Tuple(
            id=self._statement_tup_id,
            position=self._statement_pos,
            element=external1
        )
        self._reified.add(statement_tup)
        self._statement_pos += 1
        for pos, element in enumerate(node.body, start=0):
            self._reified.add(preds.Literal_Tuple(
                    id=external.body.id,
                    position=pos,
                    element=self._reify_ast(element)))
        return

    @_reify_ast.register(ASTType.Rule)
    def _reify_rule(self, node):
        self._program_statements.append(node)
        rule1 = preds.Rule1()

        # assumption: head can only be a Literal
        head = self._reify_ast(node.head)
        rule = preds.Rule(id=rule1.id,
                          head=head,
                          body=preds.Literal_Tuple1())
        self._reified.add(rule)
        statement_tup = preds.Statement_Tuple(
            id=self._statement_tup_id,
            position=self._statement_pos,
            element=rule1
        )
        self._reified.add(statement_tup)
        self._statement_pos += 1
        for pos, element in enumerate(node.body, start=0):
            self._reified.add(preds.Literal_Tuple(
                    id=rule.body.id,
                    position=pos,
                    element=self._reify_ast(element)))
        return

    @_reify_ast.register(ASTType.Literal)
    def _reify_literal(self, node):
        # assumption: all literals contain only symbolic atoms
        lit1 = preds.Literal1()
        clorm_sign = preds.sign_ast2cl[node.sign]
        lit = preds.Literal(id=lit1.id, sig=clorm_sign,
                            atom=self._reify_ast(node.atom))
        self._reified.add(lit)
        return lit1

    @_reify_ast.register(ASTType.SymbolicAtom)
    def _reify_symbolic_atom(self, node):
        atom1 = preds.Atom1()
        atom = preds.Atom(id=atom1.id, symbol=self._reify_ast(node.symbol))
        self._reified.add(atom)
        return atom1

    @_reify_ast.register(ASTType.Function)
    def _reify_function(self, node):
        """Reify an ast node with node.ast_type of ASTType.Function.

        Note that clingo's ast also represents propositional constants
        as nodes with node.type of ASTType.Function and an empty
        node.arguments list; thus some additional care must be taken
        to create the correct clorm predicate.

        """
        function1 = preds.Function1()
        function = preds.Function(
            id=function1.id,
            name=node.name,
            arguments=preds.Term_Tuple1(),
        )
        self._reified.add(function)
        for pos, term in enumerate(node.arguments, start=0):
            self._reified.add(
                preds.Term_Tuple(
                    id=function.arguments.id,
                    position=pos,
                    element=self._reify_ast(term),
                )
            )
        return function1

    @_reify_ast.register(ASTType.Variable)
    def _reify_variable(self, node):
        variable1 = preds.Variable1()
        self._reified.add(preds.Variable(id=variable1.id,
                                         name=node.name))
        return variable1

    @_reify_ast.register(ASTType.SymbolicTerm)
    def _reify_symbolic_term(self, node):
        """Reify symbolic term.

        Note that the only possible child of a symbolic term is a
        clingo symbol denoting a number, variable, or constant, so we
        don't represent this ast node in our reification.

        """
        return self._reify_ast(node.symbol)

    @_reify_ast.register(ASTType.BinaryOperation)
    def _reify_binary_operation(self, node):
        clorm_operator = preds.binary_operator_ast2cl[node.operator_type]
        binop1 = preds.Binary_Operation1()
        binop = preds.Binary_Operation(
            id=binop1.id,
            operator=clorm_operator,
            left=self._reify_ast(node.left),
            right=self._reify_ast(node.right)
        )
        self._reified.add(binop)
        return binop1

    @_reify_ast.register(SymbolType.Number)
    def _reify_symbol_number(self, symb):
        number1 = preds.Number1()
        self._reified.add(preds.Number(id=number1.id,
                                       value=symb.number))
        return number1

    @_reify_ast.register(SymbolType.Function)
    def _reify_symbol_function(self, symb):
        """Reify constant term.

        Note that clingo represents constant terms as a
        clingo.Symbol.Function with empty argument list.

        """
        func1 = preds.Function1()
        self._reified.add(preds.Function(id=func1.id, name=symb.name,
                                         arguments=preds.Term_Tuple1()))
        return func1

    @_reify_ast.register(SymbolType.String)
    def _reify_symbol_string(self, symb):
        string1 = preds.String1()
        self._reified.add(preds.String(id=string1.id,
                                       value=symb.string))
        return string1

    def _reflect_child_pred(self, parent_pred, child_id_pred):
        """Utility function that takes a unary ast predicate
        containing only an identifier pointing to a child predicate,
        queries reified factbase for child predicate, and returns the
        child node obtained by reflecting the child predicate.

        """
        identifier = child_id_pred.id
        nonunary_pred = getattr(preds,
                                type(child_id_pred).__name__.rstrip("1"))
        query = self._reified.query(nonunary_pred)\
                             .where(nonunary_pred.id == identifier)
        child_preds = list(query.all())
        if len(child_preds) == 0:
            msg = (f"Error finding child fact of predicate:\n{parent_pred}:\n"
                   f"Expected single child fact for identifier {child_id_pred}"
                   ", found none.")
            raise ChildQueryError(msg)
        elif len(child_preds) > 1:
            child_pred_strings = [str(pred) for pred in child_preds]
            msg = (f"Error finding child fact of predicate:\n{parent_pred}:\n"
                   f"Expected single child fact for identifier {child_id_pred}"
                   ", found multiple:\n" + "\n".join(child_pred_strings))
            raise ChildQueryError(msg)
        else:
            child_pred = child_preds[0]
        return self._reflect_predicate(child_pred)

    def _reflect_child_preds(self, parent_pred, id_predicate):
        """Utility function that takes a unary ast predicate
        containing only an identifier pointing to a tuple of child
        predicates, and returns a list of the child nodes obtained by
        reflecting all child predicates in order.

        """
        identifier = id_predicate.id
        nonunary_pred = getattr(preds,
                                type(id_predicate).__name__.rstrip("1"))
        query = self._reified.query(nonunary_pred)\
                             .where(nonunary_pred.id == identifier)\
                             .order_by(nonunary_pred.position)
        tuples = list(query.all())
        child_nodes = list()
        for tup in tuples:
            child_nodes.append(self._reflect_child_pred(tup, tup.element))
        return child_nodes

    @singledispatchmethod
    def _reflect_predicate(self, pred: preds.AST_Predicate):  # nocoverage
        """Convert the input AST element's reified clorm predicate
        representation back into a the corresponding memer of clingo's
        abstract syntax tree.

        """
        raise NotImplementedError(
            f"reflection not implemented for predicate of type {type(pred)}."
        )

    @_reflect_predicate.register
    def _reflect_program(self, program: preds.Program) -> Sequence[AST]:
        subprogram = list()
        parameter_nodes = self._reflect_child_preds(program, program.parameters)
        subprogram.append(ast.Program(location=DUMMY_LOC,
                                      name=program.name,
                                      parameters=parameter_nodes))
        statement_nodes = self._reflect_child_preds(program, program.statements)
        subprogram.extend(statement_nodes)
        return subprogram

    @_reflect_predicate.register
    def _reflect_external(self, external: preds.External) -> AST:
        atom_node = self._reflect_child_pred(external, external.atom)
        body_nodes = self._reflect_child_preds(external, external.body)
        ext_type = ast.SymbolicTerm(
            location=DUMMY_LOC,
            symbol=symbol.Function(name=external.external_type, arguments=[]))
        return ast.External(location=DUMMY_LOC, atom=atom_node,
                            body=body_nodes, external_type=ext_type)

    @_reflect_predicate.register
    def _reflect_rule(self, rule: preds.Rule) -> AST:
        head_node = self._reflect_child_pred(rule, rule.head)
        body_nodes = self._reflect_child_preds(rule, rule.body)
        return ast.Rule(location=DUMMY_LOC, head=head_node, body=body_nodes)

    @_reflect_predicate.register
    def _reflect_literal(self, lit: preds.Literal) -> AST:
        sign = preds.sign_cl2ast[lit.sig]
        atom_node = self._reflect_child_pred(lit, lit.atom)
        return ast.Literal(location=DUMMY_LOC, sign=sign, atom=atom_node)

    @_reflect_predicate.register
    def _reflect_atom(self, atom: preds.Atom) -> AST:
        return ast.SymbolicAtom(symbol=self._reflect_child_pred(atom,
                                                                atom.symbol))

    @_reflect_predicate.register
    def _reflect_function(
            self, func: preds.Function) -> Union[AST, Symbol]:
        """Reflect function, which may represent a propositional
        constant, predicate, function symbol, or constant term"""
        arg_nodes = self._reflect_child_preds(func, func.arguments)
        return ast.Function(
            location=DUMMY_LOC, name=func.name, arguments=arg_nodes,
            external=0
        )

    @_reflect_predicate.register
    def _reflect_variable(self, var: preds.Variable) -> AST:
        return ast.Variable(location=DUMMY_LOC, name=var.name)

    @_reflect_predicate.register
    def _reflect_number(self, number: preds.Number) -> AST:
        return ast.SymbolicTerm(
            location=DUMMY_LOC, symbol=symbol.Number(number=number.value)
        )

    @_reflect_predicate.register
    def _reflect_string(self, string: preds.String) -> Symbol:
        return ast.SymbolicTerm(
            location=DUMMY_LOC, symbol=symbol.String(string=string.value)
        )

    @_reflect_predicate.register
    def _reflect_binary_operation(
            self, operation: preds.Binary_Operation) -> AST:
        ast_operator = preds.binary_operator_cl2ast[operation.operator]
        return ast.BinaryOperation(
            location=DUMMY_LOC, operator_type=ast_operator,
            left=self._reflect_child_pred(operation, operation.left),
            right=self._reflect_child_pred(operation, operation.right)
        )

    def reflect(self):
        """Convert the reified ast contained in the internal factbase
        back into a non-ground program."""
        # reset list of program statements before population via reflect
        self._program_statements = list()
        # should probably define an order in which programs are queried
        for prog in self._reified.query(preds.Program).all():
            subprogram = self._reflect_predicate(prog)
            self._program_statements.extend(subprogram)
        logger.info(f"Reflected program string:\n{self.program_string}")
        return

    def transform(self, meta_str: Optional[str] = None,
                  meta_files: Optional[Sequence[Path]] = None) -> None:
        """Transform the reified AST using meta encoding.

        Parameter meta_prog may be a string path to file containing
        meta-encoding, or just the meta-encoding program strin itself.

        """
        if len(self._reified) == 0:
            logger.warn("Reified AST to be transformed is empty.")
        if meta_str is None and meta_files is None:
            raise ValueError("No meta-program provided for transformation.")
        meta_prog = ""
        if meta_str is not None:
            meta_prog += meta_str
        if meta_files is not None:
            for meta_file in meta_files:
                with meta_file.open() as f:
                    meta_prog += f.read()

        ctl = Control(unifier=[preds.Final])
        ctl.add_facts(self._reified)
        ctl.add(meta_prog)
        ctl.load("./src/renopro/asp/encodings/transform.lp")
        ctl.ground()
        with ctl.solve(yield_=True) as handle:
            model = next(iter(handle))
            final_facts = model.facts(shown=True)
            self._reified = FactBase([final.ast for final in final_facts])


if __name__ == "__main__":  # nocoverage
    print(ReifiedAST.from_prog(sys.argv[1]))

#  LocalWords:  nocoverage
