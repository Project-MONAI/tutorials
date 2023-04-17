# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import nbformat
import re


def define_parser(parser):
    """Define the parser for commands"""
    parser.add_argument("-f", "--filename", type=str, help="a .ipynb jupyter notebook", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-s", "--start", type=int, default=0, help="the starting index of target cells")
    group.add_argument("-i", "--index", type=int, help="the index of a target cell")
    parser.add_argument("-e", "--end", type=int, default=-1, help="the ending index of target cells")
    parser.add_argument("-k", "--keyword", type=str, default="", help="regex keyword to search in the target cell")
    parser.add_argument(
        "--type", type=str, default="markdown", help="the type of target cell (code/markdown). Default is markdown."
    )
    parser.add_argument(
        "--field",
        type=str,
        default="source",
        help="the field to extract in the target cell, such as metadata/source/outputs. Default is source.",
    )
    parser.add_argument(
        "-n", "--nestkey", type=str, default=None, help="the nesting key in the field of target. Default is None."
    )


def count_matches(filename, start, end, keyword, cell_type="markdown", field="source", nestkey=None):
    """Count the number of keyword matches from start index to end index"""
    occurrences = []

    with open(filename, "r") as f:
        notebook = nbformat.reads(f.read(), as_version=4)
        cells = notebook.cells[start:end]
        if not cells:
            raise ValueError(f"No cells extracted from index {start} to index {end}")
        for cell in cells:
            if cell.cell_type != cell_type:
                occurrences.append(0)
            else:
                #  treat backslashes as literal characters
                keyword_rawstring = repr(keyword)[1:-1]  # remove the single quotes before/after the word

                if nestkey is None:
                    content = cell[field]
                else:
                    if isinstance(cell[field], dict):
                        content = cell[field][nestkey]
                    elif isinstance(cell[field], list):
                        content = [v for v in cell[field] if isinstance(v, dict) and nestkey in v]
                    else:
                        raise ValueError(f"{type(cell[field])} is not dict or list.")
                occurrences.append(len(re.findall(keyword_rawstring, str(content))))  # value can be list of dict

    return occurrences


def print_verification_bool(*args, **kwargs):
    """Print true/false for whether target cells match the keyword and the cell_type to interface with bash"""

    result = "true" if sum(count_matches(*args, **kwargs)) else "false"
    print(result)


def print_count_array(*args, **kwargs):
    """Print number array of the matches to interface with bash"""

    print(" ".join(map(str, count_matches(*args, **kwargs))))


def main():
    parser = argparse.ArgumentParser()
    # Create a "subparsers" object to hold the subcommands
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    # Create a parser for the "verify" subcommand
    parser_verify = subparsers.add_parser("verify")
    define_parser(parser_verify)
    # Create a parser for the "count" subcommand
    parser_count = subparsers.add_parser("count")
    define_parser(parser_count)

    args = parser.parse_args()

    if "index" in args and args.index:
        args.start = args.index
        args.end = args.index + 1

    if args.subcommand == "verify":
        print_verification_bool(args.filename, args.start, args.end, args.keyword, args.type, args.field, args.nestkey)
    elif args.subcommand == "count":
        print_count_array(args.filename, args.start, args.end, args.keyword, args.type, args.field, args.nestkey)
    else:
        print("No subcommand provided. Please use -h for help")


if __name__ == "__main__":
    main()
