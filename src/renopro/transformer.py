import sys
import io
import os
from typing import List, Sequence, Callable, Optional
import logging
from pathlib import Path
from contextlib import redirect_stdout

from clingo import Control, ast
from clingo.application import ApplicationOptions, clingo_main, Flag
from clingo.solving import Model
from clingo.symbol import SymbolType, Function, Symbol
from clingo.core import MessageCode
from clingo.script import enable_python
from telingo import TelApp, transformers, imain

from renopro.utils.logger import get_clingo_logger_callback
import renopro.rast

Trace = dict[int, List[Symbol]]

logger = logging.getLogger(__name__)
clingo_logger = get_clingo_logger_callback(logger)

enable_python()

log_lvl_str2int = {"debug": 10, "info": 20, "warning": 30, "error": 40}


class TransformationError(Exception):
    """Exception raised when a transformation meta-encoding derives an
    error or is unsatisfiable."""


class MetaTransformerApp(TelApp):  # type: ignore
    """Application object as accepted by clingo.clingo_main().

    Basically the same as telingo.TelApp, apart from handling of found
    model and some minor setup before the incremental solving loop.

    """

    def _on_model(self, model: Model, horizon: int):
        self.__TelApp__horizon = horizon
        self._horizon = horizon
        trace: Trace = {}
        for sym in model.symbols(shown=True):
            if (
                sym.type == SymbolType.Function
                and len(sym.arguments) > 0
                and not sym.name.startswith("__")
            ):
                trace.setdefault(sym.arguments[-1].number, []).append(
                    Function(sym.name, sym.arguments[:-1], sym.positive)
                )
        self.traces.append(trace)
        self.transformed_asts.append(trace.get(horizon,[]))

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
        self.traces: List[Trace] = []
        self.transformed_asts: List[List[Symbol]] = []
        self._TelApp__on_model = self._on_model

    def logger(self, code: MessageCode, message: str) -> None:
        clingo_logger(code, message)

    def _parse_meta_encoding(self, value) -> bool:
        self._meta_encodings.append(value.strip())
        return True

    def register_options(self, options: ApplicationOptions):
        group = "Meta-Transformer Options"
        options.add(
            group,
            "meta-encoding,m",
            "File containing transformation meta-encoding.",
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
            "imin",
            "Minimum number of solving steps [0]",
            self._TelApp__parse_imin,
            argument="<n>",
        )
        options.add(
            group,
            "imax",
            "Maximum number of solving steps []",
            self._TelApp__parse_imax,
            argument="<n>",
        )

    def main(self, control: Control, files: Sequence[str]) -> None:
        with ast.ProgramBuilder(control) as bld:
            input_strings: List[str] = []
            if self._reify_input:
                rast = renopro.rast.ReifiedAST()
                if "-" in files:
                    rast.reify_string(sys.stdin.read())
                rast.reify_files([Path(path) for path in files])
                input_strings.append(rast.reified_string)
            else:
                input_strings.extend([
                    Path(path).read_text("utf-8") for path in files if path != "-"
                ])
                if "-" in files:
                    input_strings.append(sys.stdin.read())
            transform_files = [
                "./src/renopro/asp/transform.lp",
                "./src/renopro/asp/wrap_ast.lp",
                "./src/renopro/asp/unwrap_ast.lp",
                "./src/renopro/asp/defined.lp",
                "./src/renopro/asp/replace_id.lp",
                "./src/renopro/asp/scripts.lp",
            ] + self._meta_encodings
            input_strings.extend(
                [Path(path).read_text("utf-8") for path in transform_files]
            )
            future_sigs, program_parts = transformers.transform(input_strings, bld.add)

        imain(
            control,
            future_sigs,
            program_parts,
            self._TelApp__on_model,
            self._TelApp__imin,
            self._TelApp__imax,
            self._TelApp__istop,
        )

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
        self,
        step: int,
        table: dict[int, List[Symbol]],
        print_step: bool,
    ) -> None:
        symbols = table.get(step, [])
        logs: dict[int, List[str]] = {40: [], 30: [], 20: [], 10: []}
        sys.stdout.write(f"% State {step}:")
        s = ""
        sig = None
        for symb in sorted(symbols):
            if not symb.name.startswith("__"):
                if print_step:
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
        horizon = self._horizon
        trace = self.traces[-1]
        for step in range(horizon):
            self._print_step(step, trace, self._print_inter.flag)
        if self._reflect_output.flag:
            self._print_step(horizon, trace, False)
            rast = renopro.rast.ReifiedAST()
            rast.add_reified_symbols(self.transformed_asts[0])
            rast.reflect()
            sys.stdout.write(rast.program_string + "\n")
        else:
            self._print_step(horizon, trace, True)


def transform(
    meta_files: List[Path], input_files: Optional[List[Path]] = None, options: Optional[List[str]] = None
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
