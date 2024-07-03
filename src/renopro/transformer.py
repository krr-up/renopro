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

from clingo import Control
import clingo.ast as ast
import clingo.symbol as symbol
from clingo.application import Application, ApplicationOptions, clingo_main, Flag
from clingo.solving import Model
from clingo.symbol import SymbolType, Function, Symbol
from clingo.core import MessageCode
from clingo.script import enable_python
from renopro.utils.logger import get_clingo_logger_callback
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

    def visit_Rule(self, node: ast.AST):
        # node.update(body=node.body.append(self.initial))
        node.body.append(self.initial)
        return node


class MetaEncodingTransformer(ast.Transformer):
    """Transforms meta-encodings."""

    @staticmethod
    def create_literal(name: str, args: List[ast.AST], sign: int) -> ast.AST:
        return ast.Literal(
            DUMMY_LOC, sign, ast.SymbolicAtom(ast.Function(DUMMY_LOC, name, args, 0))
        )

    @staticmethod
    def create_external(name: str, args: List[ast.AST], body: List[ast.AST]):
        return ast.External(
            DUMMY_LOC,
            ast.SymbolicAtom(ast.Function(DUMMY_LOC, name, args, 0)),
            body,
            ast.SymbolicTerm(DUMMY_LOC, symbol.Function("false", [])),
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_program: Tuple[str, Optional[str]] = ("initial", None)
        self.external_statements: List[ast.AST] = [
            self.create_external("initial", [], []),
            self.create_external("final", [], []),
        ]

    def visit_Program(self, node: ast.AST):
        if len(node.parameters) == 0:
            for prog_name in ["initial", "final", "always", "dynamic"]:
                if node.name == prog_name:
                    self.current_program = (prog_name, None)
        elif len(node.parameters) == 1 and node.name == "step":
            self.current_program = ("step", node.parameters[0].name)
            param_ast = ast.Function(DUMMY_LOC, node.parameters[0].name, [], 0)
            self.external_statements.append(
                self.create_external("step", [param_ast], [])
            )
        return ast.Program(node.location, "base", [])

    def visit_Rule(self, node: ast.AST):
        prog_name, param = self.current_program
        if param is None:
            if prog_name in ["initial", "final"]:
                node.body.append(self.create_literal(prog_name, [], 0))
            elif prog_name == "dynamic":
                node.body.append(self.create_literal("initial", [], 1))
        elif param is not None and prog_name == "step":
            node.body.append(
                self.create_literal(
                    prog_name, [ast.Function(DUMMY_LOC, param, [], 0)], 0
                )
            )
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
        self._lambda: int = 2
        self.traces: List[Trace] = []
        self.transformed_asts: List[List[Symbol]] = []

    def logger(self, code: MessageCode, message: str) -> None:
        clingo_logger(code, message)

    def _parse_meta_encoding(self, value) -> bool:
        self._meta_encodings.append(value.strip())
        return True

    def _parse_lambda(self, value) -> bool:
        self._lambda = int(value)
        return True

    def register_options(self, options: ApplicationOptions):
        group = "Meta-Transformer Options"
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
            "lambda,l",
            "Number of time steps in the transformation.",
            self._parse_lambda,
        )

    def _on_model(self, model: Model):
        trace: Trace = defaultdict(list)
        for sym in model.symbols(shown=True):
            print(sym)
            trace[sym.arguments[1].number].append(sym.arguments[0])
        self.traces.append(trace)
        self.transformed_asts.append(trace[self._lambda - 1])

    def main(self, control: Control, files: Sequence[str]) -> None:
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
        transform_files = [
            "./src/renopro/asp/transform.lp",
            "./src/renopro/asp/wrap_ast.lp",
            "./src/renopro/asp/unwrap_ast.lp",
            "./src/renopro/asp/defined.lp",
            "./src/renopro/asp/replace_id.lp",
            "./src/renopro/asp/scripts.lp",
        ] + self._meta_encodings
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
        ast.parse_files(
            transform_files, lambda stm: transformed_stms.append(meta_enc_tf(stm))
        )
        transformed_stms.extend(meta_enc_tf.external_statements)
        prog_str = "\n".join([str(stm) for stm in transformed_stms])
        reified_prog = subprocess.run(
            ["clingo", "--output=reify"],
            input=prog_str,
            encoding="utf-8",
            capture_output=True,
            check=True,
        ).stdout
        """
        when just adding the reified program string via control.add, 
        I get parsing error unexpected EOF from clingo. No idea why. 
        As a temporary workaround, we save the string to a temp file and then 
        load, which seems to work for some reason.
        """
        tmp = tempfile.NamedTemporaryFile()
        with open(tmp.name, "w", encoding="utf-8") as f:
            f.write(reified_prog)
        # print("% Reified program:\n")
        # print(reified_prog)
        # control.add(reified_prog, [], "base")
        control.load(tmp.name)
        control.load("./tests/asp/transform/meta-telingo/meta.lp")
        control.add(f"#const lambda={self._lambda}.")
        control.ground()
        control.solve(on_model=self._on_model)

    def _log_log(self, symb: Symbol, logs: dict[int, List[str]]) -> None:
        log_lvl_symb = symb.arguments[0]
        msg_format_str = str(symb.arguments[1])
        log_lvl_strings = log_lvl_str2int.keys()
        if (
            log_lvl_symb.type != SymbolType.String
            or log_lvl_symb.string not in log_lvl_strings
        ):
            raise TransformationError(
                "First argument of log term must be one of the string symbols: '"
                + "', '".join(log_lvl_strings)
                + "'"
            )
        level = log_lvl_str2int[log_lvl_symb.string]
        log_strings = [str(s).strip('"') for s in symb.arguments[2:]]
        msg_str = msg_format_str.format(*log_strings)
        logs[level].append(msg_str)

    def _print_step(
        self, step: int, trace: Trace, print_step: bool, reflect: bool
    ) -> None:
        symbols = trace[step]
        logs: dict[int, List[str]] = {40: [], 30: [], 20: [], 10: []}
        sys.stdout.write(f"% State {step}:")
        s = ""
        sig = None
        ast_symbols: List[Symbol] = []
        for symb in sorted(symbols):
            if print_step:
                if reflect:
                    if symb.match("ast", 1) and symb.arguments[0].match("fact", 1):
                        ast_symbols.append(symb.arguments[0].arguments[0])
                else:
                    if (symb.name, len(symb.arguments), symb.positive) != sig:
                        s += "\n"
                        sig = (symb.name, len(symb.arguments), symb.positive)
                    s += f" {symb}."
            if (
                symb.type == SymbolType.Function
                and symb.positive is True
                and symb.name == "log"
                and len(symb.arguments) > 1
            ):
                self._log_log(symb, logs)
        if reflect:
            rast = renopro.rast.ReifiedAST()
            rast.add_reified_symbols(ast_symbols)
            rast.reflect()
            sys.stdout.write(rast.program_string + "\n")
        s += "\n"
        sys.stdout.write(s)
        for level, msgs in logs.items():
            for msg in msgs:
                if level == 40:
                    logger.error(
                        msg, exc_info=logger.getEffectiveLevel() == logging.DEBUG
                    )
                else:
                    logger.log(level, msg)
        if msgs := logs[40]:
            raise TransformationError("\n".join(msgs))

    def print_model(self, model: Model, printer: Callable[[], None]) -> None:
        horizon = self._lambda - 1
        trace = self.traces[-1]
        for step in range(horizon):
            self._print_step(
                step, trace, self._print_inter.flag, self._reflect_output.flag
            )
        if self._reflect_output.flag:
            self._print_step(horizon, trace, False, False)
            rast = renopro.rast.ReifiedAST()
            rast.add_reified_symbols(self.transformed_asts[0])
            rast.reflect()
            sys.stdout.write(rast.program_string + "\n")
        else:
            self._print_step(horizon, trace, True, False)


def transform(
    meta_files: List[Path],
    input_files: Optional[List[Path]] = None,
    options: Optional[List[str]] = None,
) -> List[List[Symbol]]:
    """Transform the reified AST using meta encoding."""
    input_files = [] if input_files is None else input_files
    options = [] if options is None else options
    options += [f"-m {str(meta_file)}" for meta_file in meta_files]
    meta_tf_app = MetaTransformerApp()
    args = [str(f) for f in input_files] + options + ["--quiet=2,2,2", "-V0"]
    clingo_main(meta_tf_app, args)
    return meta_tf_app.transformed_asts


if __name__ == "__main__":
    app = MetaTransformerApp()
    clingo_main(app, sys.argv[1:])
