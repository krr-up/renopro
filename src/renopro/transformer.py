import sys
import io
import os
from typing import List, Sequence, Callable
import logging

from clingo import Control, ast
from clingo.application import ApplicationOptions
from clingo.solving import Model
from clingo.symbol import SymbolType, Function, Symbol
from telingo import TelApp, transformers
from clorm import FactBase

from renopro.utils.logger import get_clingo_logger_callback
from renopro.rast import ReifiedAST


logger = logging.getLogger(__name__)


class MetaTransformerApp(TelApp):  # type: ignore
    """Application object as accepted by clingo.clingo_main().

    Basically the same as telingo.TelApp, apart from handling of found
    model and some minor setup before the incremental solving loop.

    """

    def __init__(self, rast: ReifiedAST) -> None:
        """
        Initializes the application setting the program name.

        See clingo.clingo_main().
        """
        super().__init__()
        self.program_name = "meta-transformer"
        self.version = "0.1"
        self.transformed_models: List[List[Symbol]] = []
        self.__meta_files: List[str] = []

    logger = get_clingo_logger_callback(logger)

    def __parse_meta_file(self, value: str) -> bool:
        if os.path.isfile(value):
            self.__meta_files.append(value)
            return True
        return False

    def register_options(self, options: ApplicationOptions) -> None:
        group = "Meta-Transformer Options"
        options.add(group, "meta-file,m", "Meta-encoding file defining (part of) the transformation.", self.__parse_meta_file, True)
        super().register_options(options)

    def main(self, control: Control, files: Sequence[str]) -> None:
        if len(self.__rast.reified_facts) == 0:
            logger.warning("Reified AST to be transformed is empty.")
        logger.debug(
            "Reified facts before applying transformation:\n%s", self.__rast.reified_string
        )
        with ast.ProgramBuilder(control) as bld:
            open_files = [open(path, encoding="utf-8") for path in files]
            if len(open_files) == 0:
                open_files.append(io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8"))
            open_files.append(open("./src/renopro/asp/transform.lp", encoding="utf-8"))
            future_sigs, program_parts = transformers.transform(
                [path.read() for path in open_files], bld.add
            )

        super().imain(
            control,
            future_sigs,
            program_parts,
            self.__on_model,
            self.__imin,
            self.__imax,
            self.__istop,
        )

    def print_model(self, model: Model, printer: Callable[[], None]) -> None:
        transformed_symbols: List[Symbol] = []
        logger.info("Sequence of reified symbols obtained during transformation:\n")
        table: dict[int, List[Symbol]] = {}
        for sym in model.symbols(shown=True):
            if sym.type == SymbolType.Function and len(sym.arguments) > 0:
                table.setdefault(sym.arguments[-1].number, []).append(
                    Function(sym.name, sym.arguments[:-1], sym.positive)
                )
        for step in range(self.__horizon + 1):
            symbols = table.get(step, [])
            logger.info(" State %s:", step)
            sig = None
            
            for sym in sorted(symbols):
                if not sym.name.startswith("__"):
                    if (sym.name, len(sym.arguments), sym.positive) != sig:
                        logger.info("\n")
                        sig = (sym.name, len(sym.arguments), sym.positive)
                        logger.info(" %s", sym)
                        sys.stdout.write("\n")
                        if step == self.__horizon:
                            transformed_symbols.append(sym)
        self.transformed_models.append(transformed_symbols)
                
