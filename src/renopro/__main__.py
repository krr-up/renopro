"""
The main entry point for the application.
"""

import sys

from clingo import clingo_main

from renopro.rast import ReifiedAST
from renopro.transformer import MetaTransformerApp

from .utils.logger import setup_logger
from .utils.parser import get_parser


def main():
    """
    Run the main function.
    """
    parser = get_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    if sys.argv[1] == "reify":
        args = parser.parse_args()
        setup_logger("renopro", args.log)
        rast = ReifiedAST()
        rast.reify_files(args.infiles)
        if args.commented:
            print(rast.reified_string_doc)
        else:
            print(rast.reified_string)

    elif sys.argv[1] == "reflect":
        args = parser.parse_args()
        setup_logger("renopro", args.log)
        rast = ReifiedAST()
        rast.add_reified_files(args.infiles)
        rast.reflect()
        print(rast.program_string)
    elif sys.argv[1] == "transform":
        sys.exit(int(clingo_main(MetaTransformerApp(), sys.argv[2:])))
    else:
        parser.print_help()
        sys.exit()


if __name__ == "__main__":
    main()
