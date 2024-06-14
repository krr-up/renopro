"""
The main entry point for the application.
"""

from renopro.rast import ReifiedAST

from .utils.logger import setup_logger
from .utils.parser import get_parser


def main():
    """
    Run the main function.
    """
    parser = get_parser()
    args = parser.parse_args()
    setup_logger("renopro", args.log)
    if args.command == "reify":
        rast = ReifiedAST()
        rast.reify_files(args.infiles)
        if args.commented:
            print(rast.reified_string_doc)
        else:
            print(rast.reified_string)

    elif args.command == "reflect":
        rast = ReifiedAST()
        rast.add_reified_files(args.infiles)
        rast.reflect()
        print(rast.program_string)

    elif args.command == "transform":
        rast = ReifiedAST()
        if args.input_format == "reified":
            rast.add_reified_files(args.infiles)
        elif args.input_format == "reflected":
            rast.reify_files(args.infiles)
        for meta_encoding in args.meta_encodings:
            rast.transform(meta_files=meta_encoding, clingo_options=args.clingo_options)
        if args.output_format == "reified":
            print(rast.reified_string)
        elif args.output_format == "reflected":
            rast.reflect()
            print(rast.program_string)


if __name__ == "__main__":
    main()
