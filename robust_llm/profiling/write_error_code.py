#!/usr/bin/env python3
"""Runs a command and writes the error code to a file."""
import argparse
import subprocess
import sys
from pathlib import Path


def main(command: list[str], output_file: Path) -> int:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(command, check=False)
        with open(output_file, "w") as f:
            f.write(str(result.returncode))
        return result.returncode
    except (Exception, KeyboardInterrupt):
        with open(output_file, "w") as f:
            f.write("1")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Specify the command to run by passing it after '--'.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-file",
        "-o",
        type=Path,
        required=True,
        help="File to write the exit code to.",
    )
    default_usage = parser.format_usage().strip().removeprefix("usage: ")
    parser.usage = f"{default_usage} -- [command_to_run]"

    known_args, unknown_args = parser.parse_known_args()
    if len(unknown_args) < 2 or unknown_args[0] != "--":
        parser.error(
            "No command provided to run."
            " Use -- to separate script arguments from the command."
            f" Detected args: {unknown_args}",
        )
    command = unknown_args[1:]
    exit_code = main(command=command, output_file=known_args.output_file)
    sys.exit(exit_code)
