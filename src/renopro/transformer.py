import sys
import io
import os
from typing import List, Sequence, Callable, Optional, Tuple
import logging
from pathlib import Path
from contextlib import redirect_stdout
import subprocess
import tempfile
from collections import defaultdict
from importlib.resources import files

from clingo import Control
import clingo.ast as ast
import clingo.symbol as symbol
from clingo.application import Application, ApplicationOptions, clingo_main, Flag
from clingo.solving import Model
from clingo.symbol import SymbolType, Function, Symbol
from clingo.core import MessageCode
from clingo.script import enable_python

from renopro.utils.logger import (
    get_clingo_logger_callback,
    setup_logger,
    log_string2level,
)
import renopro.rast

Trace = dict[int, List[Symbol]]

DUMMY_LOC = ast.Location(ast.Position("<string>", 1, 1), ast.Position("<string>", 1, 1))

logger = logging.getLogger(__name__)
clingo_logger = get_clingo_logger_callback(logger)

enable_python()

log_lvl_str2int = {"debug": 10, "info": 20, "warning": 30, "error": 40}


class TransformationError(Exception):
    """Exception raised when a transformation meta-encoding derives an
    error or is unsatisfiable."""


class InputTransformer(ast.Transformer):
    """Adds initial to bodies of input AST facts."""

    initial = ast.Literal(
        DUMMY_LOC, 0, ast.SymbolicAtom(ast.Function(DUMMY_LOC, "initial", [], 0))
    )

    def visit_SymbolicAtom(self, node: ast.AST):
        node.symbol.arguments.append(ast.SymbolicTerm(DUMMY_LOC, symbol.Number(0)))
        return node


class MetaEncodingTransformer(ast.Transformer):
    """Transforms meta-encodings."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_program: Tuple[str, Optional[str]] = ("initial", None)

    def visit_Program(self, node: ast.AST):
        if len(node.parameters) == 0:
            for prog_name in ["initial", "final", "always", "dynamic"]:
                if node.name == prog_name:
                    self.current_program = (prog_name, None)
        elif len(node.parameters) == 1 and node.name == "step":
            self.current_program = ("step", node.parameters[0].name)
        return ast.Program(node.location, "base", [])

    def visit_Rule(self, node: ast.AST):
        prog_name, _ = self.current_program
        self.visit_children(node)
        if prog_name == "dynamic":
            node.body.append(
                ast.Literal(
                    DUMMY_LOC,
                    0,
                    ast.Comparison(
                        ast.Variable(DUMMY_LOC, "_TimePoint"),
                        [
                            ast.Guard(
                                ast.ComparisonOperator.LessThan,
                                ast.Function(DUMMY_LOC, "transform_steps", [], 0),
                            )
                        ],
                    ),
                )
            )
        return node

    def _visit_Atom(self, atom: ast.AST):
        prog_name, param = self.current_program
        if param is None:
            if prog_name == "initial":
                atom.arguments.append(ast.SymbolicTerm(DUMMY_LOC, symbol.Number(0)))
            elif prog_name == "final":
                atom.arguments.append(ast.Function(DUMMY_LOC, "transform_steps", [], 0))
            elif prog_name in ["always", "dynamic"]:
                atom.arguments.append(ast.Variable(DUMMY_LOC, "_TimePoint"))
        elif param is not None and prog_name == "step":
            atom.arguments.append(ast.Function(DUMMY_LOC, param, [], 0))

    def visit_SymbolicAtom(self, node: ast.AST):
        prog_name, param = self.current_program
        if node.symbol.ast_type is ast.ASTType.Pool:
            for item in node.symbol.arguments:
                self._visit_Atom(item)
        elif node.symbol.ast_type is ast.ASTType.Function:
            self._visit_Atom(node.symbol)
        elif node.symbol.ast_type is ast.ASTType.UnaryOperation:
            self._visit_Atom(node.symbol.argument)
        else:
            raise RuntimeError("Branch should be unreachable.")
        return node


class MetaTransformerApp(Application):  # type: ignore
    """Application object as accepted by clingo.clingo_main()."""

    def __init__(self) -> None:
        """
        Initializes the application setting the program name.

        See clingo.clingo_main().
        """
        super().__init__()
        self.program_name = "meta-transformer"
        self.version = "0.1"
        self._print_inter = Flag(False)
        self._reify_input = Flag(False)
        self._reflect_output = Flag(False)
        self._meta_encodings: List[str] = []
        self._transform_steps: int = 1
        self.traces: List[Trace] = []
        self.transformed_asts: List[List[Symbol]] = []
        self._log_level: int = logging.WARNING
        self._exception: Optional[Exception] = None

    def logger(self, code: MessageCode, message: str) -> None:
        clingo_logger(code, message)

    def _parse_meta_encoding(self, value) -> bool:
        self._meta_encodings.append(value.strip())
        return True

    def _parse_transform_steps(self, value) -> bool:
        self._transform_steps = int(value)
        return True

    def _parse_log_levels(self, value) -> bool:
        if value in log_lvl_str2int.keys():
            self._log_level = log_lvl_str2int[value]
            return True
        return False

    def register_options(self, options: ApplicationOptions):
        group = "Meta-Transformer Options"
        options.add(
            group,
            "log,l",
            "Set log level. Valid values are error, warning, info, debug.",
            self._parse_log_levels,
        )
        options.add(
            group,
            "meta-encoding,m",
            "Meta-encodings defining AST transformation(s), to be applied.",
            self._parse_meta_encoding,
            multi=True,
        )
        options.add_flag(
            group,
            "reify",
            "Enable reification of input clingo program files into AST facts.",
            self._reify_input,
        )
        options.add_flag(
            group,
            "reflect",
            "Enable reflection of transformed AST facts into a clingo program.",
            self._reflect_output,
        )
        options.add_flag(
            group,
            "print-inter",
            "Enable printing of intermediate transformation steps.",
            self._print_inter,
        )
        options.add(
            group,
            "transform-steps",
            "Number of time steps in the transformation meta-encoding.",
            self._parse_transform_steps,
        )

    def _on_model(self, model: Model):
        trace: Trace = defaultdict(list)
        logs: dict[int, List[str]] = {40: [], 30: [], 20: [], 10: []}
        for symb in model.symbols(shown=True):
            # print(symb)
            trace[symb.arguments[-1].number].append(
                symbol.Function(symb.name, symb.arguments[:-1], symb.positive)
            )
            if (
                symb.type == SymbolType.Function
                and symb.positive is True
                and symb.name == "log"
                and len(symb.arguments) > 1
            ):
                self._log_log(symb, logs)
        self.traces.append(trace)
        ast_facts = [
            sym.arguments[0].arguments[0]
            for sym in trace[self._transform_steps]
            if sym.match("ast", 1) and sym.arguments[0].match("fact", 1)
        ]
        self.transformed_asts.append(ast_facts)
        for level, msgs in logs.items():
            for msg in msgs:
                if level == 40:
                    logger.error(
                        msg, exc_info=logger.getEffectiveLevel() == logging.DEBUG
                    )
                else:
                    logger.log(level, msg)
        if msgs := logs[40]:
            exception = TransformationError("\n".join(msgs))
            self._exception = exception
            raise exception

    def main(self, control: Control, files: Sequence[str]) -> None:
        setup_logger("renopro", self._log_level)
        input_string = ""
        input_files: List[str] = [path for path in files if path != "-"]
        expect_stdin: bool = "-" in files
        if self._reify_input:
            rast = renopro.rast.ReifiedAST()
            if expect_stdin:
                rast.reify_string(sys.stdin.read())
            rast.reify_files([Path(path) for path in input_files])
            input_string += rast.reified_string
        else:
            if expect_stdin:
                input_string += sys.stdin.read()
        # need to do some transformations on input and transform files
        input_tf = InputTransformer()
        transformed_stms: List[ast.AST] = []
        if len(input_files) > 0 and not self._reify_input:
            ast.parse_files(
                input_files, lambda stm: transformed_stms.append(input_tf(stm))
            )
        ast.parse_string(
            input_string, lambda stm: transformed_stms.append(input_tf(stm))
        )
        meta_enc_tf = MetaEncodingTransformer()
        if len(self._meta_encodings) > 0:
            ast.parse_files(
                self._meta_encodings,
                lambda stm: transformed_stms.append(meta_enc_tf(stm)),
            )
        transformed_prog_str = "\n".join([str(stm) for stm in transformed_stms])
        # print(transformed_prog_str)
        control.add(transformed_prog_str)
        asp_path = Path("src", "renopro", "asp")
        transform_files = [
            "transform.lp",
            "wrap_ast.lp",
            "unwrap_ast.lp",
            "defined.lp",
            "replace_id.lp",
            "scripts.lp",
            "ast_fact2id.lp",
        ]
        for f in transform_files:
            control.load(str(asp_path / f))
        control.add(f"#const transform_steps={self._transform_steps}.")
        control.ground()
        control.solve(on_model=self._on_model)

    def _log_log(self, symb: Symbol, logs: dict[int, List[str]]) -> None:
        log_lvl_symb = symb.arguments[0]
        log_step = symb.arguments[-1].number
        msg_format_str = str(symb.arguments[1]).strip('"')
        log_lvl_strings = log_lvl_str2int.keys()
        if (
            log_lvl_symb.type != SymbolType.String
            or log_lvl_symb.string not in log_lvl_strings
        ):
            exception = TransformationError(
                "First argument of log term must be one of the string symbols: '"
                + "', '".join(log_lvl_strings)
                + "'"
            )
            self._exception = exception
            raise exception
        level = log_lvl_str2int[log_lvl_symb.string]
        log_strings = [str(s).strip('"') for s in symb.arguments[2:-1]]
        msg_str = msg_format_str.format(*log_strings)
        logs[level].append(msg_str)

    def _print_step(self, step: int, trace: Trace) -> None:
        symbols = trace[step]
        sys.stdout.write(f"% State {step}:")
        s = ""
        sig = None
        ast_symbols: List[Symbol] = []
        for symb in sorted(symbols):
            if self._reflect_output.flag:
                if symb.match("ast", 1) and symb.arguments[0].match("fact", 1):
                    ast_symbols.append(symb.arguments[0].arguments[0])
            else:
                if (symb.name, len(symb.arguments), symb.positive) != sig:
                    s += "\n"
                    sig = (symb.name, len(symb.arguments), symb.positive)
                s += f" {symb}."
        if self._reflect_output.flag:
            rast = renopro.rast.ReifiedAST()
            rast.add_reified_symbols(ast_symbols)
            rast.reflect()
            sys.stdout.write("\n" + rast.program_string)
        s += "\n"
        sys.stdout.write(s)

    def print_model(self, model: Model, printer: Callable[[], None]) -> None:
        horizon = self._transform_steps
        trace = self.traces[-1]
        if self._print_inter.flag:
            for step in range(horizon):
                self._print_step(step, trace)
        self._print_step(horizon, trace)


def transform(
    input_files: Optional[List[Path]] = None,
    meta_files: Optional[List[Path]] = None,
    options: Optional[List[str]] = None,
) -> List[List[Symbol]]:
    """Transform the reified AST using meta encoding."""
    input_files = [] if input_files is None else input_files
    meta_files = [] if meta_files is None else meta_files
    options = [] if options is None else options
    options += [f"-m {str(meta_file)}" for meta_file in meta_files]
    meta_tf_app = MetaTransformerApp()
    args = [str(f) for f in input_files] + options + ["--outf=3"]
    clingo_main(meta_tf_app, args)
    if meta_tf_app._exception is not None:
        raise meta_tf_app._exception
    return meta_tf_app.transformed_asts


if __name__ == "__main__":
    app = MetaTransformerApp()
    clingo_main(app, sys.argv[1:])
