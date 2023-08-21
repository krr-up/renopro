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
    setup_logger("main", args.log)
    rast = ReifiedAST()
    if args.command == "reify":
        rast.reify_files(args.infiles)
        if args.commented:
            print(rast.reified_string_doc)
        else:
            print(rast.reified_string)

    elif args.command == "reflect":
        rast.add_reified_files(args.infiles)
        rast.reflect()
        print(rast.program_string)

    elif args.command == "transform":
        if args.input_format == "reified":
            rast.add_reified_files(args.infiles)
        elif args.input_format == "reflected":
            rast.reify_files(args.infiles)
        rast.transform(meta_files=args.meta_encoding)
        if args.output_format == "reified":
            print(rast.reified_string)
        elif args.output_format == "reflected":
            rast.reflect()
            print(rast.program_string)


if __name__ == "__main__":
    main()
