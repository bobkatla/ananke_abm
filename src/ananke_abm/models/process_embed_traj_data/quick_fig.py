# plots_activities_only.py
# Create:
#   (1) Aggregate Trip Starts by Hour (all non-Home purposes)
#   (2) Stacked Area: Trip Purpose Distribution by Hour
#
# Usage (defaults assume the CSV is in the same folder):
#   python plots_activities_only.py --activities activities_homebound_wd.csv --save

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main(activities_csv: str, save: bool):
    # --- Load ---
    df = pd.read_csv(activities_csv)

    # Expecting columns: ['persid','hhid','stopno','purpose','startime','total_duration']
    # Keep only non-Home stops for trip-start analysis
    non_home = df[df["purpose"] != "Home"].copy()

    # Bin start times to hours (integer hours since midnight; dataset may go beyond 24h)
    non_home["starthour"] = (non_home["startime"] // 60).astype(int)

    # Ensure hours are continuous in the index (helps plotting smooth x-axis)
    hour_min = int(non_home["starthour"].min())
    hour_max = int(non_home["starthour"].max())
    all_hours = pd.Index(range(hour_min, hour_max + 1), name="starthour")

    # --- Plot 1: Aggregate Trip Starts by Hour ---
    trips_by_hour = (
        non_home.groupby("starthour")
        .size()
        .reindex(all_hours, fill_value=0)
    )

    plt.figure(figsize=(12,6))
    plt.plot(trips_by_hour.index, trips_by_hour.values, marker="o")
    plt.title("Aggregate Trip Starts by Hour (All Purposes)")
    plt.xlabel("Hour of day")
    plt.ylabel("Number of trip starts")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        out = Path("aggregate_trip_starts_by_hour.png")
        plt.savefig(out, dpi=200)
        print(f"Saved {out.resolve()}")
    else:
        plt.show()

    # --- Plot 2: Stacked Area – Trip Purpose Distribution by Hour ---
    purpose_by_hour = (
        non_home.groupby(["starthour", "purpose"])
        .size()
        .unstack("purpose", fill_value=0)
        .reindex(all_hours, fill_value=0)
    )

    ax = purpose_by_hour.plot(
        kind="area",
        stacked=True,
        alpha=0.85,
        figsize=(12,6)
    )
    ax.set_title("Trip Purpose Distribution by Hour")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Number of trip starts")
    ax.legend(title="Purpose", ncol=3, frameon=False)
    plt.tight_layout()
    if save:
        out = Path("trip_purpose_distribution_by_hour.png")
        plt.savefig(out, dpi=200)
        print(f"Saved {out.resolve()}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--activities", type=str, default="activities_homebound_wd.csv",
                        help="Path to activities_homebound_wd.csv")
    parser.add_argument("--save", action="store_true", help="Save plots instead of showing")
    args = parser.parse_args()
    main(args.activities, args.save)
