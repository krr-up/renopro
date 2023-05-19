import sys
from functools import singledispatchmethod
from typing import Optional

import clingo.ast as ast
import clingo.symbol as symbol
from clingo.ast import AST, ASTType, Location, Position, parse_string
from clingo.symbol import Symbol, SymbolType
from clorm import FactBase

import renopro.clorm_predicates as preds

DUMMY_LOC = Location(Position("<string>", 1, 1), Position("<string>", 1, 1))


class ReifiedAST:
    def __init__(self, program: Optional[str] = None):
        self._factbase = FactBase()
        self._term_tuple_count = -1
        self._literal_tuple_count = -1
        if program is not None:
            self.reify_program(program)

    def __str__(self):
        return self._factbase.asp_str()

    def _inc_term_tuple(self):
        self._term_tuple_count += 1
        return self._term_tuple_count

    def _inc_literal_tuple(self):
        self._literal_tuple_count += 1
        return self._literal_tuple_count

    def reify_program(self, program: str) -> None:
        parse_string(program, self._reify_ast)

    def _reify_ast(self, node):
        """Reify the input ast node by returning or adding it's clorm fact
        representation to the internal fact base.
        """
        if isinstance(node, AST):
            if node.ast_type is ASTType.Rule:
                # assumption: head can only be a Literal
                self._factbase.add(
                    preds.Rule(
                        head=self._reify_ast(node.head),
                        body=preds.Literal_Tuple_Id(self._inc_literal_tuple()),
                    )
                )
                lit_tup_id = self._literal_tuple_count
                # assumpion: body only contains Literals
                for pos, literal in enumerate(node.body, start=0):
                    self._factbase.add(
                        preds.Literal_Tuple(
                            identifier=lit_tup_id,
                            position=pos,
                            element=self._reify_ast(literal),
                        )
                    )

            elif node.ast_type is ASTType.Literal:
                # assumption: all literals contain only symbolic atoms
                return preds.Literal(sign_=node.sign, atom=self._reify_ast(node.atom))

            elif node.ast_type is ASTType.SymbolicAtom:
                return self._reify_ast(node.symbol)

            elif node.ast_type is ASTType.Function:
                func = preds.Function(
                    name=node.name,
                    arguments=preds.Term_Tuple_Id(self._inc_term_tuple()),
                )
                term_tup_id = self._term_tuple_count
                for pos, term in enumerate(node.arguments, start=0):
                    self._factbase.add(
                        preds.Term_Tuple(
                            identifier=term_tup_id,
                            position=pos,
                            element=self._reify_ast(term),
                        )
                    )
                return func

            elif node.ast_type is ASTType.Variable:
                return preds.Variable(name=node.name)

            elif node.ast_type is ASTType.SymbolicTerm:
                return self._reify_ast(node.symbol)

        elif isinstance(node, Symbol):
            if node.type is SymbolType.Number:
                return preds.Integer(node.number)

            elif node.type is SymbolType.Function:
                return preds.Function(
                    node.name, preds.Term_Tuple_Id(self._inc_term_tuple())
                )

            elif node.type is SymbolType.String:
                return preds.String(node.name)

    @singledispatchmethod
    def _dereify_predicate(self, pred: preds.AST_Predicate):
        """Convert the input clorm predicate (representing an AST
        node) back into a the corresponding memer of clingo's
        abstract syntax tree."""
        raise NotImplementedError(
            "Dereification not implemented for predicate" f"of type {type(pred)}."
        )

    @_dereify_predicate.register
    def _(self, rule: preds.Rule) -> AST:
        lit_tup_id = rule.body.identifier
        lit_tup_preds = (
            self._factbase.query(preds.Literal_Tuple)
            .where(preds.Literal_Tuple.identifier == lit_tup_id)
            .all()
        )
        body_lit_preds = [pred.element for pred in lit_tup_preds]
        body_lit_asts = list(map(self._dereify_predicate, body_lit_preds))
        return ast.Rule(
            location=DUMMY_LOC,
            head=self._dereify_predicate(rule.head),
            body=body_lit_asts,
        )

    @_dereify_predicate.register
    def _(self, lit: preds.Literal) -> AST:
        return ast.Literal(
            location=DUMMY_LOC,
            sign=int(lit.sign),
            atom=ast.SymbolicAtom(self._dereify_predicate(lit.atom)),
        )

    @_dereify_predicate.register
    def _(self, func: preds.Function) -> AST:
        term_tup_id = func.arguments.identifier
        term_tup_preds = (
            self._factbase.query(preds.Term_Tuple)
            .where(preds.Term_Tuple.identifier == term_tup_id)
            .all()
        )
        func_arg_preds = [pred.element for pred in term_tup_preds]
        func_arg_asts = list(map(self._dereify_predicate, func_arg_preds))
        return ast.Function(
            location=DUMMY_LOC, name=str(func.name), arguments=func_arg_asts, external=0
        )

    @_dereify_predicate.register
    def _(self, var: preds.Variable) -> AST:
        return ast.Variable(location=DUMMY_LOC, name=var.name)

    @_dereify_predicate.register
    def _(self, integer: preds.Integer) -> AST:
        return ast.SymbolicTerm(
            location=DUMMY_LOC, symbol=symbol.Number(number=int(integer.value))
        )

    @_dereify_predicate.register
    def _(self, string: preds.String) -> Symbol:
        return symbol.String(string=str(string.value))

    def dereify(self):
        """Convert the reified ast contained in the internal factbase
        back into a non-ground program."""
        program_str = ""
        for rule in self._factbase.query(preds.Rule).all():
            program_str += str(self._dereify_predicate(rule))
        return program_str


if __name__ == "__main__":
    print(ReifiedAST(sys.argv[1]))
