#!/usr/bin/env python3
"""Run and profile a Python job using py-spy.

The Python job may use accelerate. Pass the accelerate command into this script
rather than calling this script through accelerate.

When viewing the generated profile in speedscope, you will need to click through
the different subprocesses to find the "real" process (or processes, if using
accelerate), since the main process profiled will be a different wrapper script
write_error_code.py. Look at the processes labeled "MainThread".

Profiles are saved to disk. The other obvious alternative is to upload them as
artifacts to W&B, but this has issues:
- A straightforward attempt to add an artifact to a completed run using the W&B
  API gives an error. The suggested way to do this is to resume the run:
  https://docs.wandb.ai/guides/artifacts/artifacts-faqs/#how-do-i-log-an-artifact-to-an-existing-run
  However this overwrites some info of the existing run. For example,
  https://wandb.ai/farai/robust-llm/runs/66gtgd5y/overview
  is a run from June 2024 that was resumed in October 2024 to add an artifact. Now
  the "Start time" is October, and the "Hostname" is "MacBookPro".
- Another way is to create a new run, add the artifact to that run, and then we
  can link the artifact the finished main job run. But it's convoluted to have a
  100% overhead of dummy runs that are only there to hold artifacts, plus W&B
  storage is likely more expensive than disk storage.
"""

import argparse
import json
import logging
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

from robust_llm.file_utils import RenamableTemporaryFile, get_shared_data_dir

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_with_profiling(
    command: list[str],
    save_directory: Path,
    sampling_rate: int,
    include_idle_threads: bool,
) -> None:
    # We use speedscope (out of the possible output formats [flamegraph,
    # speedscope, raw]) because it seems to take the least disk space.
    PROFILE_FORMAT = "speedscope"
    save_directory.mkdir(parents=True, exist_ok=True)

    profile_file = RenamableTemporaryFile(suffix=".json")
    # We want to use the W&B run name and ID in the profile's final filename.
    # The job will write them into wandb_info_file.
    wandb_info_file = tempfile.NamedTemporaryFile(suffix=".json")
    exit_code_file = tempfile.NamedTemporaryFile(suffix=".txt")

    # We directly pass the command through py-spy instead of using `py-spy
    # record --pid` because we can't use `--pid` on Kubernetes. It requires
    # enabling SYS_PTRACE
    # (https://github.com/benfred/py-spy?tab=readme-ov-file#how-do-i-run-py-spy-in-kubernetes),
    # and Flamingo's pod security settings ("baseline" in
    # https://kubernetes.io/docs/concepts/security/pod-security-standards/) do
    # not allow this.
    # However the error code of the command is thrown away by py-spy, so we save
    # the error code by wrapping the command with write_error_code.py.
    py_spy_flags = [
        "--rate",
        str(sampling_rate),
        "--output",
        profile_file.name,
        "--format",
        PROFILE_FORMAT,
        "--subprocesses",
    ]
    if include_idle_threads:
        py_spy_flags.append("--idle")
    full_command = (
        [
            "py-spy",
            "record",
        ]
        + py_spy_flags
        + [
            "--",
            str(Path(__file__).parent / "write_error_code.py"),
            "--output-file",
            exit_code_file.name,
            "--",
        ]
        + command
        + [f"environment.wandb_info_filename={wandb_info_file.name}"]
    )
    if platform.system() == "Darwin":
        logging.warning(
            "py-spy needs sudo on macOS. You may be prompted for your password."
        )
        full_command = ["sudo", "--preserve-env"] + full_command
    logging.info(f"Full command: {full_command}")

    try:
        subprocess.run(
            full_command,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        # Sometimes py-spy completes and writes a complete profile but it gives
        # an error `Error: No child process (os error 10)` at the very end:
        # https://github.com/benfred/py-spy/issues/243
        # Let's try to check for this error and ignore it: if the profile file
        # is non-empty we'll just consider the command a success.
        if Path(profile_file.name).stat().st_size > 0:
            logging.warning(
                f"Ignoring py-spy error because the profile file is non-empty: {e}"
            )
        else:
            raise

    command_exit_code = int(exit_code_file.read())
    if command_exit_code != 0:
        # The underlying command failed, though py-spy gave an exit code of 0.
        # We should still treat it as if check=True triggered.
        raise subprocess.CalledProcessError(
            cmd=command,
            returncode=command_exit_code,
        )

    # Rename the profiling data to contain W&B name. Path.rename() doesn't
    # work across filesystems, so we use shutil.move instead.
    wandb_info = json.loads(wandb_info_file.read())
    new_filename = save_directory / (
        f"{wandb_info['wandb_run_name']}-{wandb_info['wandb_run_id']}"
        f".{PROFILE_FORMAT}.json"
    )
    shutil.move(profile_file.name, new_filename)
    logging.info(f"Profiling data saved to: {new_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Specify the command to profile by passing it after '--'.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--save-directory",
        type=Path,
        default=Path(get_shared_data_dir()) / "profiles",
        help="Directory to save the profiling data",
    )
    parser.add_argument(
        "--rate",
        type=int,
        # Default py-spy value is 100, but we reduce it drastically to reduce
        # the disk space of the profile.
        default=1,
        help="The sampling rate for py-spy in samples per second.",
    )
    parser.add_argument(
        "--no-idle",
        action="store_false",
        # We probably usually want to include idle threads since otherwise the
        # amount of time spent in each function is misleading. However it does
        # make the profile larger.
        help="Don't include stack traces for idle threads",
        dest="idle",
    )
    default_usage = parser.format_usage().strip().removeprefix("usage: ")
    parser.usage = f"{default_usage} -- [command_to_profile]"

    known_args, unknown_args = parser.parse_known_args()
    if len(unknown_args) < 2 or unknown_args[0] != "--":
        parser.error(
            "No command provided to run and profile."
            " Use -- to separate script arguments from the command to profile."
            f" Detected args: {unknown_args}",
        )
    command = unknown_args[1:]
    run_with_profiling(
        command=command,
        save_directory=known_args.save_directory,
        sampling_rate=known_args.rate,
        include_idle_threads=known_args.idle,
    )
