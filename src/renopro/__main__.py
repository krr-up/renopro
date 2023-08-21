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
        if args.input_format == "refied":
            rast.add_reified_files(args.infiles)
        elif args.input_format == "reflected":
            rast.reify_files(args.infiles)
        meta_str = ""
        for opened_file in args.meta_encoding:
            contents = opened_file.read()
            meta_str += contents
        rast.transform(meta_str=meta_str)
        if args.output_format == "reified":
            print(rast.reified_string)
        elif args.output_format == "reflected":
            rast.reflect()
            print(rast.program_string)


if __name__ == "__main__":
    main()
