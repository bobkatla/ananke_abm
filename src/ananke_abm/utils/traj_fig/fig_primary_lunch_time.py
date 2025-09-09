#!/usr/bin/env python3
"""
Generate two zoomed stacked plots (10:00–14:00) from a buffer grid CSV:
  1) Work @ 10:00 & 14:00   (Y-zoom 0..0.05)
  2) Education @ 10:00 & 14:00 (Y-zoom 0..0.005)

Buffer CSV format:
- Rows: one per person (persid)
- Columns: "persid", then time bins in minutes, e.g., "0","5","10",...,"1800"
- Each time cell contains the activity label for [t, t+step).

Usage:
  python make_stacked_zoom_main.py buffer_real_5m.csv --out-dir plots
  # Optional:
  #   --t0 600 --t1 840      # change window (minutes)
  #   --dpi 220              # change DPI
  #   --show                 # also show figures even if --out-dir is set
  #   (If --out-dir is omitted, figures are shown instead of saved)
"""

import os
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ORDERED_LABELS_TOPDOWN = ["Home", "Work", "Education", "Social", "Shopping", "Accompanying", "Other"]

def swap_home_with(main: str) -> list[str]:
    order = ORDERED_LABELS_TOPDOWN.copy()
    if main in order:
        i_home = order.index("Home")
        i_main = order.index(main)
        order[i_home], order[i_main] = order[i_main], order[i_home]
    return order

FIXED_COLORS = {
    "Home": "#9ecae1",
    "Work": "#3182bd",
    "Education": "#31a354",
    "Social": "#756bb1",
    "Shopping": "#e6550d",
    "Accompanying": "#fd8d3c",
    "Other": "#969696",
}

def load_buffer(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "persid" not in df.columns:
        raise ValueError("Buffer CSV must include a 'persid' column.")
    # ensure time columns can be parsed as ints
    _ = [int(c) for c in df.columns if c != "persid"]
    return df

def detect_step(time_cols_int: List[int]) -> int:
    diffs = np.diff(np.sort(time_cols_int))
    if len(diffs) == 0:
        return 5
    # Use the minimum positive step
    step = int(np.min(diffs[diffs > 0])) if np.any(diffs > 0) else int(np.min(diffs))
    return max(step, 1)

def filter_main(df: pd.DataFrame, activity: str, t0: int, t1: int) -> pd.DataFrame:
    """Keep persons where df[str(t0)] == df[str(t1)] == activity, and subset to t0..t1 (inclusive)."""
    mask = (df[str(t0)] == activity) & (df[str(t1)] == activity)
    kept = df.loc[mask].copy()
    return kept

def compute_props(df: pd.DataFrame, tcols: List[int], main: str) -> pd.DataFrame:
    """Return wide proportion table indexed by time with columns ORDERED_LABELS_TOPDOWN."""
    records = []
    for t in tcols:
        col = df[str(t)].astype(str)
        col_mapped = col.where(col.isin(ORDERED_LABELS_TOPDOWN), "Other")
        counts = col_mapped.value_counts()
        total = counts.sum()
        for lab in ORDERED_LABELS_TOPDOWN:
            prop = float(counts.get(lab, 0)) / total if total > 0 else 0.0
            records.append({"time": t, "purpose": lab, "proportion": prop})
    long = pd.DataFrame(records)
    wide = (long.pivot(index="time", columns="purpose", values="proportion")
                 .reindex(columns=swap_home_with(main))
                 .fillna(0.0)
                 .sort_index())
    return wide

def stacked_plot(props_wide: pd.DataFrame,
                 title: str,
                 y_max: float,
                 out_png: str | None,
                 t0: int,
                 t1: int,
                 dpi: int = 200,
                 show: bool = False,
                 main: str = "Work"):
    """Make polished stacked area with last bin included (right edge), top-down order, hours x-axis, Y zoom [0, y_max]."""
    x_min = props_wide.index.values
    steps = np.diff(x_min)
    step_min = float(steps[0]) if len(steps) else 5.0

    # extend right edge to include the final interval
    x_edges_min = np.append(x_min, x_min[-1] + step_min)
    x_edges_hr = x_edges_min / 60.0

    # bottom-up arrays for stackplot; extend with last column
    bottom_up_labels = list(reversed(swap_home_with(main)))
    y_bottom_up = props_wide[bottom_up_labels].to_numpy().T
    y_bottom_up_ext = np.hstack([y_bottom_up, y_bottom_up[:, -1][:, None]])
    colors_bottom_up = [FIXED_COLORS[l] for l in bottom_up_labels]

    # plot
    plt.figure(figsize=(11.5, 6.5), dpi=dpi)
    plt.stackplot(x_edges_hr, y_bottom_up_ext, colors=colors_bottom_up, antialiased=True)
    plt.title(title)
    plt.xlabel("Time (hours)")
    plt.ylabel("Proportion")
    plt.xlim(t0/60.0, (t1)/60.0)
    plt.ylim(0.0, y_max)

    # ticks every 30 minutes
    xmin_hr, xmax_hr = t0/60.0, (t1)/60.0
    plt.xticks(np.arange(np.floor(xmin_hr*2)/2, np.ceil(xmax_hr*2)/2 + 1e-9, 0.5))

    # style
    plt.grid(axis='both', alpha=0.15)
    ax = plt.gca()
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # vertical guides at t0 and t1
    for xline in [t0/60.0, t1/60.0]:
        plt.axvline(x=xline, color="#888888", linestyle="--", linewidth=0.8, alpha=0.6)

    # legend in requested top-down order
    legend_handles = [Patch(facecolor=FIXED_COLORS[l], label=l) for l in swap_home_with(main)]
    plt.legend(handles=legend_handles, loc="upper left", frameon=True, facecolor="white", edgecolor="black")
    plt.tight_layout()

    if out_png:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, bbox_inches="tight")
        print(f"Saved: {out_png}")
        plt.close()
    if show or not out_png:
        plt.show()

def fig_primary_lunch_time(
        buffer_csv: str,
        out_dir: str | None,
        y_work_max: float = 0.5,
        y_edu_max: float = 0.5,
        t0: int = 600,
        t1: int = 840,
        dpi: int = 300):
    print(f"Generating zoomed stacked plots for Work/Education main activity ({t0} & {t1})...")
    df = load_buffer(buffer_csv)
    time_cols_int = sorted([int(c) for c in df.columns if c != "persid"])
    step = detect_step(time_cols_int)
    tcols = list(range(t0, t1 + step, step))

    # Construct time columns for the requested window (inclusive)
    # --- Work cohort ---
    df_work = filter_main(df, "Work", t0, t1)
    n_work = len(df_work)
    props_work = compute_props(df_work, tcols, "Work")
    out_work = None
    if out_dir:
        out_work = os.path.join(out_dir, "stacked_work_zoom.png")
    title_work = f"Stacked Proportions (Y-zoom 0-{y_work_max}, includes last bin) — Work — n={n_work:,}"
    stacked_plot(props_work, title_work, y_max=y_work_max, out_png=out_work, t0=t0, t1=t1, dpi=dpi, show=True if out_dir is None else False, main="Work")

    # --- Education cohort ---
    df_edu = filter_main(df, "Education", t0, t1)
    n_edu = len(df_edu)
    props_edu = compute_props(df_edu, tcols, "Education")
    out_edu = None
    if out_dir:
        out_edu = os.path.join(out_dir, "stacked_education_zoom.png")
    title_edu = f"Stacked Proportions (Y-zoom 0-{y_edu_max}, includes last bin) — Education — n={n_edu:,}"
    stacked_plot(props_edu, title_edu, y_max=y_edu_max, out_png=out_edu, t0=t0, t1=t1, dpi=dpi, show=True if out_dir is None else False, main="Education")

