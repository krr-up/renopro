"""Module implementing reification and de-reification of non-ground programs"""
import sys
from functools import singledispatchmethod, partial
import logging
from typing import Iterable, Sequence, Union, Optional
from pathlib import Path

import clingo
from clingo import ast, symbol
from clingo.ast import (AST, ASTType, Location, Position,
                        parse_string, parse_files, BinaryOperator)
from clingo.symbol import Symbol, SymbolType
from clorm.clingo import Control, Model
from clorm import (FactBase, control_add_facts,
                   SymbolPredicateUnifier, parse_fact_string,
                   UnifierNoMatchError, parse_fact_files)

import renopro.clorm_predicates as preds

logger = logging.getLogger(__name__)

DUMMY_LOC = Location(Position("<string>", 1, 1), Position("<string>", 1, 1))

binop_clorm2ast = {preds.BinaryOperator[op.name]: ast.BinaryOperator[op.name]
                   for op in ast.BinaryOperator}
binop_ast2clorm = {v: k for k, v in binop_clorm2ast.items()}


class ReifiedAST:
    """Class for converting between reified and non-reified
    representation of ASP programs."""

    def __init__(self):
        self.factbase = FactBase()
        self._term_tuple_count = -1
        self._literal_tuple_count = -1

    def __str__(self):
        return self.factbase.asp_str()

    def _inc_term_tuple(self) -> int:
        self._term_tuple_count += 1
        return self._term_tuple_count

    def _inc_literal_tuple(self) -> int:
        self._literal_tuple_count += 1
        return self._literal_tuple_count

    def add_reified_facts(self, reified_facts: Iterable[preds.AST_Predicate]) -> None:
        """Add factbase containing reified facts into internal factbase."""
        self.factbase.add(reified_facts)

    def add_reified_program(self, reified_program: str) -> None:
        """Add string of reified facts into internal factbase."""
        facts = parse_fact_string(reified_program, unifier=[preds.AST_Facts])
        self.factbase.add(facts)

    def add_reified_files(self, reified_files: Sequence[Path]) -> None:
        """Add files containing reified facts into internal factbase."""
        reified_files = [str(f) for f in reified_files]
        facts = parse_fact_files(reified_files, unifier=[preds.AST_Facts])
        self.factbase.add(facts)

    def reify_program(self, prog_str: str) -> None:
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
        parse_files(files, self._reify_ast)

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
                return preds.Literal(sign_=node.sign,
                                     atom=self._reify_ast(node.atom))

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

            elif node.ast_type is ASTType.BinaryOperation:
                clorm_operator = binop_ast2clorm[node.operator_type]
                return preds.Binary_Operation(
                    operator=clorm_operator,
                    left=self._reify_ast(node.left),
                    right=self._reify_ast(node.right))

        elif isinstance(node, Symbol):
            if node.type is SymbolType.Number:
                return preds.Number(node.number)

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
            .order_by(preds.Literal_Tuple.position)
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
            .order_by(preds.Term_Tuple.position)
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
        ast_operator = binop_clorm2ast[operation.operator]
        return ast.BinaryOperation(
            location=DUMMY_LOC, operator_type=ast_operator,
            left=self._reflect_predicate(operation.left),
            right=self._reflect_predicate(operation.right)
        )

    def reflect(self):
        """Convert the reified ast contained in the internal factbase
        back into a non-ground program."""
        program_str = ""
        for rule in self.factbase.query(preds.Rule).all():
            program_str += str(self._reflect_predicate(rule))
        return program_str

    def transform(self, meta_str: Optional[str] = None,
                  meta_files: Optional[Sequence[Path]] = None) -> None:
        """Transform the reified AST using meta encoding.

        Parameter meta_prog may be a string path to file containing
        meta-encoding, or just the meta-encoding program strin itself.

        """
        if len(self.factbase) == 0:
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
        ctl = clingo.Control()
        control_add_facts(ctl, self.factbase)
        ctl.add(meta_prog)
        ctl.load("./src/renopro/asp/encodings/transform.lp")
        ctl.ground()
        with ctl.solve(yield_=True) as handle:
            model = next(iter(handle))
            ast_str = "".join([str(final.arguments[0]) + "."
                               for final in model.symbols(shown=True)])
            self.factbase = parse_fact_string(ast_str, unifier=preds.AST_Facts)

        # TODO: do this with clorm like below.
        # could not get the unfication with preds.Final to work.
        # 
        # ctl = Control()
        # ctl.add_facts(self.factbase)
        # ctl.add(meta_prog)
        # ctl.load("./src/renopro/asp/encodings/transform.lp")
        # ctl.ground()
        # with ctl.solve(yield_=True) as handle:
        #     model = next(iter(handle))
        #     facts = model.facts(unifier=SymbolPredicateUnifier)
        #     logger.warn(facts)

        # def model_callback(fb: FactBase, model: Model):
        #     facts = model.facts(unifier=[preds.Final])
        #     logger.warn(facts)
        #     fb.add(facts)
        #     return False

        # ctl.solve(on_model=partial(model_callback, model_facts))
        # self.factbase = model_facts


if __name__ == "__main__":  # nocoverage
    print(ReifiedAST.from_prog(sys.argv[1]))

#  LocalWords:  nocoverage
