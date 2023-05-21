"""Module implementing reification and de-reification of non-ground programs"""
import sys
from functools import singledispatchmethod

from clingo import ast
from clingo import symbol
from clingo.ast import AST, ASTType, Location, Position, parse_string
from clingo.symbol import Symbol, SymbolType
from clorm import FactBase

import renopro.clorm_predicates as preds

DUMMY_LOC = Location(Position("<string>", 1, 1), Position("<string>", 1, 1))


class ReifiedAST:
    """Class for converting between reified and non-reified
    representation of ASP programs."""

    def __init__(self):
        self.factbase = FactBase()
        self._term_tuple_count = -1
        self._literal_tuple_count = -1

    @classmethod
    def from_facts(cls, ast_facts: preds.AST_Facts):
        """Create reified AST instance from input set of facts
        representing a reified AST."""
        rast = cls()
        rast.add_ast_facts(ast_facts)
        return rast

    @classmethod
    def from_str(cls, prog_str: str):
        """Create ReifiedAST instance by reifying input program string."""
        rast = cls()
        rast.reify_program(prog_str)
        return rast

    def __str__(self):
        return self.factbase.asp_str()

    def _inc_term_tuple(self):
        self._term_tuple_count += 1
        return self._term_tuple_count

    def _inc_literal_tuple(self):
        self._literal_tuple_count += 1
        return self._literal_tuple_count

    def add_ast_facts(self, ast_facts: preds.AST_Facts) -> None:
        """Add reified AST elements to factbase."""
        self.factbase.add(ast_facts)

    def reify_program(self, prog_str: str) -> None:
        """Reify input program string, adding fact representation of
        AST to the factbase."""
        parse_string(prog_str, self._reify_ast)

    def _reify_ast(self, node):
        """Reify the input ast node by adding it's clorm fact
        representation to the internal fact base.

        """
        if isinstance(node, AST):
            if node.ast_type is ASTType.Rule:
                # assumption: head can only be a Literal
                self.factbase.add(
                    preds.Rule(
                        head=self._reify_ast(node.head),
                        body=preds.Literal_Tuple_Id(self._inc_literal_tuple()),
                    )
                )
                lit_tup_id = self._literal_tuple_count
                # assumpion: body only contains Literals
                for pos, literal in enumerate(node.body, start=0):
                    self.factbase.add(
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
                    self.factbase.add(
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

            # Clingo stores constant terms as an instance of
            # clingo.ast.SymbolicTerm with the inner symbol being a
            # clingo.Symbol.Function with empty argument list.
            if node.type is SymbolType.Function:
                return preds.Function(
                    node.name, preds.Term_Tuple_Id(self._inc_term_tuple())
                )

            if node.type is SymbolType.String:
                return preds.String(node.string)

    @singledispatchmethod
    def _reflect_predicate(self, pred: preds.AST_Predicate):  # nocoverage
        """Convert the input AST element's reified clorm predicate
        representation back into a the corresponding memer of clingo's
        abstract syntax tree.

        """
        raise NotImplementedError(
            "Dereification not implemented for predicate" f"of type {type(pred)}."
        )

    @_reflect_predicate.register
    def _reflect_rule(self, rule: preds.Rule) -> AST:
        lit_tup_id = rule.body.identifier
        lit_tup_preds = (
            self.factbase.query(preds.Literal_Tuple)
            .where(preds.Literal_Tuple.identifier == lit_tup_id)
            .all()
        )
        body_lit_preds = [pred.element for pred in lit_tup_preds]
        body_lit_asts = list(map(self._reflect_predicate, body_lit_preds))
        return ast.Rule(
            location=DUMMY_LOC,
            head=self._reflect_predicate(rule.head),
            body=body_lit_asts,
        )

    @_reflect_predicate.register
    def _reflect_literal(self, lit: preds.Literal) -> AST:
        return ast.Literal(
            location=DUMMY_LOC,
            sign=lit.sign_,
            atom=ast.SymbolicAtom(self._reflect_predicate(lit.atom)),
        )

    @_reflect_predicate.register
    def _reflect_function(self, func: preds.Function) -> AST:
        term_tup_id = func.arguments.identifier
        term_tup_preds = (
            self.factbase.query(preds.Term_Tuple)
            .where(preds.Term_Tuple.identifier == term_tup_id)
            .all()
        )
        func_arg_preds = [pred.element for pred in term_tup_preds]
        func_arg_asts = list(map(self._reflect_predicate, func_arg_preds))
        return ast.Function(
            location=DUMMY_LOC, name=func.name, arguments=func_arg_asts, external=0
        )

    @_reflect_predicate.register
    def _reflect_variable(self, var: preds.Variable) -> AST:
        return ast.Variable(location=DUMMY_LOC, name=var.name)

    @_reflect_predicate.register
    def _(self, integer: preds.Integer) -> AST:
        return ast.SymbolicTerm(
            location=DUMMY_LOC, symbol=symbol.Number(number=integer.value)
        )

    @_reflect_predicate.register
    def _reflect_string(self, string: preds.String) -> Symbol:
        return ast.SymbolicTerm(
            location=DUMMY_LOC, symbol=symbol.String(string=string.value)
        )

    def reflect(self):
        """Convert the reified ast contained in the internal factbase
        back into a non-ground program."""
        program_str = ""
        for rule in self.factbase.query(preds.Rule).all():
            program_str += str(self._reflect_predicate(rule))
        return program_str


if __name__ == "__main__":  # nocoverage
    print(ReifiedAST.from_str(sys.argv[1]))

#  LocalWords:  nocoverage
