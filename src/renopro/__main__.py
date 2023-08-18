"""
The main entry point for the application.
"""
from .utils.logger import setup_logger
from .utils.parser import get_parser


def main():
    """
    Run the main function.
    """
    parser = get_parser()
    args = parser.parse_args()
    setup_logger(
        "main",
        args.log,
    )
    # rast = ReifiedAST()
    # if args.reify:
    #     if args.files:
    #         rast.reify_files(args.files)
    #     if args.string:
    #         rast.reify_string(args.string)
    #     print(rast.reified_string_doc)

    # elif args.reflect:
    #     if args.files:
    #         rast.add_reified_files(args.files)
    #     if args.string:
    #         rast.add_reified_string(args.string)
    #     rast.reflect()
    #     print(rast.program_string)

    # elif args.transform:
    #     pass


if __name__ == "__main__":
    main()
