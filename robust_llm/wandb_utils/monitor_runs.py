import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import wandb

from robust_llm.file_utils import compute_repo_path
from robust_llm.wandb_utils.wandb_api_tools import get_tracking_data_for_runs


def get_runs_by_group(group_names: str | list[str], min_dt: datetime) -> pd.DataFrame:
    api = wandb.Api(timeout=60)

    filter_date = min_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    runs = api.runs(
        "farai/robust-llm",
        filters={
            "$and": [
                {"$or": [{"group": group} for group in group_names]},
                {"created_at": {"$gt": filter_date}},
            ]
        },
        order="+created_at",
    )
    return get_tracking_data_for_runs(runs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor W&B runs and save to CSV.")
    parser.add_argument(
        "--group_names", nargs="+", help="Group names to filter runs by."
    )
    parser.add_argument(
        "--min_dt",
        type=lambda s: datetime.strptime(s, "%Y-%m-%dT%H:%M:%S"),
        help="Minimum datetime to filter runs (format: YYYY-MM-DDTHH:MM:SS).",
        default="2024-01-01T00:00:00",
    )

    args = parser.parse_args()

    output_path = (
        Path(compute_repo_path())
        / "outputs"
        / "monitor_runs"
        / "__".join(args.group_names)
    )
    if not output_path.exists():
        output_path.mkdir(parents=True)

    df = get_runs_by_group(args.group_names, args.min_dt)
    multi_index = df.set_index(
        ["run_name", "created_at", "really_finished", "wandb_run_id"]
    ).sort_index()
    multi_index.to_csv(output_path / "multiindex.csv")
    df["running"] = df.state.eq("running")
    grouped = df.groupby("run_name")[["really_finished", "running"]].any().reset_index()
    grouped.columns = ["run_name", "finished", "running"]
    grouped.to_csv(output_path / "grouped.csv", index=False)
    print(f"CSV files exported to '{output_path}'")
    print(
        f"In total, {grouped['finished'].sum()} runs are finished and "
        f"{grouped['running'].sum()} are running out of "
        f"{grouped.shape[0]} on wandb."
    )
