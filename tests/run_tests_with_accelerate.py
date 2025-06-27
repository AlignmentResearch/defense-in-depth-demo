#!/usr/bin/env python3
"""Runs tests using accelerate."""
import argparse
import logging
import os
import shlex
import subprocess
from pathlib import Path

from robust_llm.file_utils import compute_repo_path

logging.basicConfig(level=logging.INFO)


def get_pytest_and_accelerate_args(
    unknown_args: list[str],
) -> tuple[list[str], list[str]]:
    """Parses unknown_args to return (pytest_args, accelerate_args)."""
    # `--accelerate_args` separates pytest args from accelerate args.
    # We default to passing unknown args to pytest since adding additional
    # pytest args is more common than adding accelerate args.
    if "--accelerate_args" not in unknown_args:
        return unknown_args, []
    accelerate_index = unknown_args.index("--accelerate_args")
    accelerate_args = unknown_args[accelerate_index + 1 :]
    pytest_args = unknown_args[:accelerate_index]
    return pytest_args, accelerate_args


def main() -> None:
    git_root = Path(compute_repo_path())

    parser = argparse.ArgumentParser(
        description=__doc__,
        # There's no simple, concise way to handle passing args to three different
        # programs. We'll document here how the parsing works.
        #
        # The color-stripping sed command is from
        # https://superuser.com/a/380778/2132114.
        epilog=r"""
Unknown arguments listed after the flag `--accelerate_args` are passed to
accelerate, and other extra arguments are passed to pytest. Therefore a command
like `./run_tests_with_accelerate.py -m multigpu` is similar to running `pytest
-m multigpu` and runs all tests with the `multigpu` marker.

As a complicated example, the args
  foo --config_file config.yml -k bar --accelerate_args abc --coverage --xyz
would pass recognized flags `--config_file bar --coverage` to this script,
unknown args `foo -k bar` to pytest, and unknown args `abc --xyz` to
accelerate.

If a multi-process hypothesis test hangs without any clear error output, try
re-running with --hypothesis_verbose. One process may have failed on a
hypothesis example and moved on to another input, leaving the processes out of
sync.

Per-process logs are written to /tmp/accelerate_tests_output/process-$RANK.log
in case it's useful for debugging. Color codes are included, so view the logs
with something like `less -R`, or strip the colors with
    sed -e 's/\x1b\[[0-9;]*m//g'""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Don't allow flag abbreviations since its behavior with
        # parse_known_args() can be confusing.
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config_file",
        type=Path,
        default=git_root / "accelerate_config.yaml",
        help="Path to accelerate config file",
    )
    parser.add_argument(
        "--num_processes", type=int, default=2, help="Number of processes to use"
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default="/tmp/accelerate_tests_output/",
        help="Directory to store log files",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Check for coverage when running tests",
    )
    parser.add_argument(
        "--hypothesis_verbose",
        action="store_true",
        help="Print more information for hypothesis tests",
    )
    default_usage = parser.format_usage().strip().removeprefix("usage: ")
    parser.usage = f"{default_usage} [pytest_args] --accelerate_args [accelerate_args]"

    args, unknown_args = parser.parse_known_args()

    pytest_args, accelerate_args = get_pytest_and_accelerate_args(unknown_args)
    logging.info(f"pytest_args: {pytest_args}")
    logging.info(f"accelerate_args: {accelerate_args}")
    if args.num_processes > 1:
        pytest_args += ["--hypothesis-profile", "multigpu"]
    if args.hypothesis_verbose:
        pytest_args += ["--hypothesis-verbosity=verbose", "--capture=no"]

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    pytest_args_str = " ".join(shlex.quote(arg) for arg in pytest_args)
    pytest_command = (
        # -i makes tee ignore SIGINT and stay open so that whatever trace pytest
        # prints upon SIGINT is captured.
        f'pytest {pytest_args_str} 2>&1 | tee -i "{args.log_dir}/process-$RANK.log"'
    )
    if args.coverage:
        # We need --parallel-mode (or equivalently, setting parallel=True in
        # .coveragerc) to tell processes to write to different coverage files.
        # We call `coverage` instead of `pytest cov=` because pytest-cov seems
        # not to support parallel mode, as its docs at
        # https://pytest-cov.readthedocs.io/en/stable/config.html say "This
        # plugin overrides the parallel option of coverage".
        pytest_command = (
            f"coverage run --source=robust_llm,{git_root}/tests --parallel-mode -m "
            + pytest_command
        )
    command = [
        "accelerate",
        "launch",
        f"--config_file={args.config_file}",
        f"--num_processes={args.num_processes}",
        *accelerate_args,
        "--no_python",
        # Wrapping the pytest command in bash -c makes the per-process $RANK
        # substitution work for the per-process log filename, otherwise we could
        # just pass the pytest command directly here instead.
        "bash",
        "-co",
        "pipefail",
        pytest_command,
    ]
    try:
        subprocess.run(
            command,
            check=True,
            env={
                **os.environ,
                "WANDB_MODE": "offline",
                # Ensure color output despite pytest output being piped
                "PYTEST_ADDOPTS": "--color=yes",
            },
        )
    finally:
        if args.coverage:
            subprocess.run(["coverage", "combine"])


if __name__ == "__main__":
    main()
