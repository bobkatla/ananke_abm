#!/usr/bin/env python3
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------- Fixed order & colors ----------------
FIXED_ORDER = [
    "Home", "Work", "Education", "Social", "Shopping", "Accompanying", "Other"
]

CMAP = {
    "Home": "#1f77b4",         # blue
    "Work": "#ff7f0e",         # orange
    "Education": "#2ca02c",    # green
    "Social": "#8c564b",       # brown
    "Shopping": "#d62728",     # red
    "Accompanying": "#9467bd", # purple
    "Other": "#7f7f7f",        # gray
}


# ---------------- Helpers ----------------
def coerce_and_clip_minutes(s: pd.Series, upper: int) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).clip(lower=0, upper=upper).astype(int)


def build_occupancy_counts(
    df: pd.DataFrame,
    horizon_min: int = 30*60,
    step_min: int = 5,
) -> pd.DataFrame:
    """
    Build occupancy counts per purpose across a fixed time grid.

    Required columns in df: ['purpose', 'startime', 'total_duration'].
    Any purpose not in FIXED_ORDER is mapped to 'Other' to keep a fixed legend/order.
    """
    req = {"purpose", "startime", "total_duration"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Map unknown purposes to 'Other' to enforce fixed categories
    df = df.copy()
    df["purpose"] = df["purpose"].astype(str)
    df.loc[~df["purpose"].isin(FIXED_ORDER), "purpose"] = "Other"

    # Coerce times
    df["startime"] = coerce_and_clip_minutes(df["startime"], horizon_min)
    df["total_duration"] = coerce_and_clip_minutes(df["total_duration"], horizon_min)

    # Time grid
    grid = np.arange(0, horizon_min + 1, step_min, dtype=int)
    n_bins = len(grid)

    # Pre-allocate counters for fixed categories only (order fixed)
    counts = {p: np.zeros(n_bins, dtype=np.int64) for p in FIXED_ORDER}

    # Efficient binning: convert each activity to index range and += 1
    # We treat occupancy as [start, end), i.e., inclusive of start, exclusive of end.
    for _, r in df.iterrows():
        start = int(r["startime"])
        end = int(min(horizon_min, r["startime"] + r["total_duration"]))
        if end <= start:
            continue

        # Convert to indices on the grid
        start_idx = start // step_min
        # If end aligns exactly to a grid boundary, do not fill that bin (exclusive end)
        end_idx = max(start_idx, min(n_bins, (end + step_min - 1) // step_min))  # ceil(end/step)

        counts[r["purpose"]][start_idx:end_idx] += 1

    # Build DataFrame in fixed order
    out = pd.DataFrame({"minute": grid})
    for purpose in FIXED_ORDER:
        out[purpose] = counts[purpose]
    return out


def to_proportions(counts_df: pd.DataFrame) -> pd.DataFrame:
    """Row-normalize counts to proportions (each minute sums to 1)."""
    props = counts_df.copy()
    totals = props[FIXED_ORDER].sum(axis=1).replace(0, np.nan)
    props[FIXED_ORDER] = props[FIXED_ORDER].div(totals, axis=0)
    return props.fillna(0.0)


def plot_flipped_fixed(props_df: pd.DataFrame, title: str = "", save_path: str | None = None, dpi: int = 200):
    """
    Flipped stacked area chart with **fixed** stack order and legend order.
    Home appears as the top band. Legend order matches FIXED_ORDER.
    """
    # Ensure all fixed columns exist
    df = props_df.copy()
    for cat in FIXED_ORDER:
        if cat not in df.columns:
            df[cat] = 0.0

    # Force column order and set index to hours for x-axis
    df_plot = df[["minute"] + FIXED_ORDER].set_index("minute").apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df_plot.index = df_plot.index / 60.0  # minutes -> hours

    # Flip stacking: compute cumulative from top down
    reversed_cols = FIXED_ORDER[::-1]  # bottom to top (so first filled is bottom)
    cumsum_from_top = df_plot[reversed_cols].cumsum(axis=1)
    bottoms = cumsum_from_top.shift(axis=1).fillna(0.0)

    plt.figure(figsize=(14, 6))
    for col in reversed_cols:
        plt.fill_between(
            df_plot.index.values,
            bottoms[col].values,
            cumsum_from_top[col].values,
            color=CMAP.get(col, None),
            label=col,
        )

    # Legend strictly follows FIXED_ORDER (Home first ... Other last)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = {lab: h for h, lab in zip(handles, labels)}
    ordered_handles = [by_label[c] for c in FIXED_ORDER if c in by_label]
    plt.legend(ordered_handles, FIXED_ORDER, title="Purpose", loc="upper left", bbox_to_anchor=(1.02, 1))

    plt.title(title or "Proportional Activity Occupancy (Home on Top, Fixed Colors & Order)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Proportion of People")
    plt.ylim(0, 1)
    plt.xlim(df_plot.index.min(), df_plot.index.max())
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")
        plt.close()
    else:
        plt.show()


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Flipped proportional occupancy plot with fixed order & colors.")
    ap.add_argument("activities_csv", help="Path to activities CSV (requires columns: purpose,startime,total_duration).")
    ap.add_argument("-o", "--out", help="Optional output image path (e.g., occupancy.png).")
    ap.add_argument("--step-min", type=int, default=5, help="Time resolution in minutes (default: 5).")
    ap.add_argument("--horizon-min", type=int, default=30*60, help="Total horizon in minutes (default: 1800 = 30h).")
    ap.add_argument("--title", type=str, default="", help="Optional plot title.")
    ap.add_argument("--dpi", type=int, default=200, help="Output DPI if saving (default: 200).")
    args = ap.parse_args()

    df = pd.read_csv(args.activities_csv)
    counts = build_occupancy_counts(df, horizon_min=args.horizon_min, step_min=args.step_min)
    props = to_proportions(counts)
    plot_flipped_fixed(props, title=args.title, save_path=args.out, dpi=args.dpi)


if __name__ == "__main__":
    main()
