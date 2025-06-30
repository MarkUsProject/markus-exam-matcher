"""
MarkUs Exam Matcher: __main__.py

Information
===============================
Environment for top-level code (entry point into package).
"""
import sys
from argparse import ArgumentParser

from .core.char_types import CharType
from .image_processing import read_chars
from .qr_scan import scan_qr


def config_arg_parser() -> ArgumentParser:
    """
    Configure a command line argument parser.
    :return: Command line argument parser.
    """
    # Initialize parser
    parser = ArgumentParser(
        prog="run_scanner.py", description="Predict handwritten characters in rectangular grids."
    )

    # Scan type argument
    parser.add_argument("scan_type", type=str, help="Scan type to be character or QR code")

    # Positional arguments
    parser.add_argument("image", type=str, help="Path to image to predict characters from.")

    # Optional arguments
    parser.add_argument(
        "--char_type",
        choices=["digit", "letter"],
        help="Type of character to classify. Only digits and letters are supported.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Specify whether to run program with debug mode enabled.",
    )

    return parser


if __name__ == "__main__":
    # Parse command line arguments
    parser = config_arg_parser()
    args = parser.parse_args(sys.argv[1:])

    if args.scan_type == "char":
        char_type = CharType.DIGIT if args.char_type == "digit" else CharType.LETTER

        # Make prediction
        pred = read_chars.run(args.image, char_type=char_type, debug=args.debug)
        print(pred)
    else:
        scanned_result = scan_qr.read_qr(args.image)
        print(scanned_result)
