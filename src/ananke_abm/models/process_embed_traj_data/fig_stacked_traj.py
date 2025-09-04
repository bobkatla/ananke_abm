#!/usr/bin/env python3
"""
Make a flipped stacked proportional distribution plot from a buffer grid CSV.

Buffer CSV format:
- One row per person (persid column + time columns 0, X, 2X, ... , maxtime)
- Each time column contains an activity label (e.g., Home, Work, ...)

What this script does:
1) Converts each time column to proportions across people.
2) Stacks them (flipped inside the plot so Home is the "background" area).
3) X-axis displayed in HOURS (0..30).
4) Legend in a white box; colors match the plotted layers.

Usage:
  python make_stacked_prop.py buffer.csv --out-png stacked.png
  python make_stacked_prop.py buffer.csv --out-csv props.csv
  python make_stacked_prop.py buffer.csv   # just show the plot

"""

import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ------ Fixed order (legend order) and fixed colors ------
ORDERED_LABELS = ["Home", "Work", "Education", "Social", "Shopping", "Accompanying", "Other"]
FIXED_COLORS = {
    "Home": "#9ecae1",         # light blue
    "Work": "#3182bd",         # blue
    "Education": "#31a354",    # green
    "Social": "#756bb1",       # purple
    "Shopping": "#e6550d",     # orange
    "Accompanying": "#fd8d3c", # light orange
    "Other": "#969696",        # grey
}

def load_buffer(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "persid" not in df.columns:
        raise ValueError("Input buffer CSV must contain a 'persid' column.")
    # Identify time columns and sort numerically
    time_cols = [c for c in df.columns if c != "persid"]
    try:
        time_cols_sorted = sorted(time_cols, key=lambda x: int(x))
    except ValueError:
        raise ValueError("Time columns must be integers or strings of integers (e.g., '0','5',...,'1800').")
    return df, time_cols_sorted

def compute_proportions(df: pd.DataFrame, time_cols_sorted: list[str]) -> pd.DataFrame:
    """
    Returns a wide DataFrame indexed by time (int minute) with columns = ORDERED_LABELS
    containing proportions at each time.
    """
    records = []
    for t in time_cols_sorted:
        col = df[t].astype(str)
        # Map any unseen label to "Other"
        col_mapped = col.where(col.isin(ORDERED_LABELS), "Other")
        counts = col_mapped.value_counts()
        total = counts.sum()
        for lab in ORDERED_LABELS:
            prop = float(counts.get(lab, 0)) / total if total > 0 else 0.0
            records.append({"time": int(t), "purpose": lab, "proportion": prop})

    prop_df_long = pd.DataFrame(records)
    prop_wide = (
        prop_df_long
        .pivot(index="time", columns="purpose", values="proportion")
        .reindex(columns=ORDERED_LABELS)
        .fillna(0.0)
        .sort_index()
    )
    return prop_wide

def plot_flipped(prop_wide: pd.DataFrame, out_png: str | None):
    """
    Makes a flipped stacked plot on an hours axis. If out_png is None, shows the plot.
    """
    # Convert index to minutes, drop final terminal point (e.g. 1800) for the area plot
    x_min = prop_wide.index.values
    # If uniform spacing, drop last bin as it's a terminal state
    if len(x_min) >= 2 and (x_min[-1] - x_min[-2]) == (x_min[1] - x_min[0]):
        x_plot_min = x_min[:-1]
        y_plot = prop_wide.iloc[:-1][ORDERED_LABELS].to_numpy().T
    else:
        x_plot_min = x_min
        y_plot = prop_wide[ORDERED_LABELS].to_numpy().T

    # Convert minutes -> hours
    x_plot_hr = x_plot_min / 60.0

    # Flip stack order ONLY for plotting (so the largest "background" sits at the top visually)
    y_plot_flipped = y_plot[::-1]
    colors_flipped = [FIXED_COLORS[l] for l in ORDERED_LABELS[::-1]]

    # Plot
    plt.figure(figsize=(12, 6), dpi=140)
    plt.stackplot(x_plot_hr, y_plot_flipped, colors=colors_flipped)
    plt.title("Stacked Proportional Distribution of Activities (Flipped, Hour Scale)")
    plt.xlabel("Time (hours)")
    plt.ylabel("Proportion")
    plt.ylim(0, 1)
    plt.xlim(float(x_plot_hr.min()), float(x_plot_hr.max()))
    # Ticks every 2 hours for readability
    start_hr = int(np.floor(x_plot_hr.min()))
    end_hr = int(np.ceil(x_plot_hr.max()))
    plt.xticks(range(start_hr, end_hr + 1, 2))

    # Legend in white box with correct color-label mapping (legend order = ORDERED_LABELS)
    legend_handles = [Patch(facecolor=FIXED_COLORS[l], label=l) for l in ORDERED_LABELS]
    plt.legend(handles=legend_handles, loc="upper left", frameon=True, facecolor="white", edgecolor="black")

    plt.tight_layout()

    if out_png:
        plt.savefig(out_png, bbox_inches="tight")
        print(f"Saved plot to: {out_png}")
        plt.close()
    else:
        plt.show()

def main():
    ap = argparse.ArgumentParser(description="Generate flipped stacked proportional plot from buffer CSV.")
    ap.add_argument("buffer_csv", type=str, help="Path to buffer CSV (rows=persid, cols=0..maxtime).")
    ap.add_argument("--out-png", type=str, default=None, help="Optional path to save PNG. If omitted, just shows the plot.")
    ap.add_argument("--out-csv", type=str, default=None, help="Optional path to save the computed proportion table CSV.")
    args = ap.parse_args()

    df, time_cols_sorted = load_buffer(args.buffer_csv)
    prop_wide = compute_proportions(df, time_cols_sorted)

    if args.out_csv:
        prop_wide.to_csv(args.out_csv)
        print(f"Saved proportions to: {args.out_csv}")

    plot_flipped(prop_wide, args.out_png)

if __name__ == "__main__":
    main()
