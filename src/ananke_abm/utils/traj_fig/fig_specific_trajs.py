#!/usr/bin/env python3
"""
Generate TWO zoomed line plots from a buffer grid CSV:
  (1) Work & Education
  (2) Social, Shopping & Accompanying

Behavior:
- X-axis is always fixed to 0..30 hours.
- Auto-zoom is applied to the Y-axis only (so details are clearer, but time stays 0..30).
- If --out-dir is omitted, the script shows both figures instead of saving files.

Usage:
  python make_zoomed_lines.py buffer_real_5m.csv
  python make_zoomed_lines.py buffer_real_5m.csv --out-dir plots --dpi 180
  python make_zoomed_lines.py buffer_real_5m.csv --eps-work 0.002 --eps-ssa 0.0015
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ORDERED_LABELS = ["Home", "Work", "Education", "Social", "Shopping", "Accompanying", "Other"]
COLORS = {
    "Home": "#9ecae1",
    "Work": "#3182bd",
    "Education": "#31a354",
    "Social": "#756bb1",
    "Shopping": "#e6550d",
    "Accompanying": "#fd8d3c",
    "Other": "#969696",
}

def load_buffer(path: str) -> tuple[pd.DataFrame, list[int]]:
    df = pd.read_csv(path)
    if "persid" not in df.columns:
        raise ValueError("Buffer CSV must contain a 'persid' column.")
    time_cols = [c for c in df.columns if c != "persid"]
    time_cols_sorted = sorted([int(c) for c in time_cols])
    return df, time_cols_sorted

def compute_proportions(df: pd.DataFrame, time_cols: list[int]) -> pd.DataFrame:
    """Return wide DataFrame: index=time (min), columns=ORDERED_LABELS with proportions."""
    recs = []
    for t in time_cols:
        s = df[str(t)].astype(str)
        s = s.where(s.isin(ORDERED_LABELS), "Other")
        vc = s.value_counts()
        tot = vc.sum()
        for lab in ORDERED_LABELS:
            recs.append({"time": t, "purpose": lab, "proportion": float(vc.get(lab, 0)) / tot if tot else 0.0})
    long = pd.DataFrame(recs)
    wide = (
        long.pivot(index="time", columns="purpose", values="proportion")
            .reindex(columns=ORDERED_LABELS)
            .fillna(0.0)
            .sort_index()
    )
    return wide

def plot_zoom_yonly(x_hr: np.ndarray,
                    series: dict[str, np.ndarray],
                    title: str,
                    save_path: str | None,
                    eps: float,
                    dpi: int = 160,
                    show: bool = False):
    """Plot multiple lines; X is fixed 0..30, Y auto-zooms to informative range."""
    stack = np.column_stack([series[k] for k in series.keys()])
    mask = (stack > eps).any(axis=1)
    if mask.any():
        idx = np.where(mask)[0]
        y_min = float(stack[idx[0]:idx[-1]+1].min())
        y_max = float(stack[idx[0]:idx[-1]+1].max())
    else:
        y_min = float(stack.min())
        y_max = float(stack.max())
    y_pad = (y_max - y_min) * 0.15 if y_max > y_min else 0.05
    y0 = max(0.0, y_min - y_pad)
    y1 = min(1.0, y_max + y_pad)

    plt.figure(figsize=(12, 5), dpi=dpi)
    for lab, y in series.items():
        plt.plot(x_hr, y, label=lab, linewidth=2.6, color=COLORS[lab])
    plt.title(title)
    plt.xlabel("Time (hours)")
    plt.ylabel("Proportion")
    plt.xlim(0.0, 30.0)                 # << fixed 0..30
    plt.xticks(range(0, 31, 2))
    plt.ylim(y0, y1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="black")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.close()
    if show or not save_path:
        plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("buffer_csv", type=str, help="Path to buffer grid CSV")
    ap.add_argument("--out-dir", type=str, default=None, help="If provided, save PNGs here; otherwise show the figures.")
    ap.add_argument("--eps-work", type=float, default=0.002, help="Y-zoom threshold for Work/Education")
    ap.add_argument("--eps-ssa", type=float, default=0.0015, help="Y-zoom threshold for Social/Shopping/Accompanying")
    ap.add_argument("--dpi", type=int, default=300, help="Figure DPI")
    args = ap.parse_args()

    df, tcols = load_buffer(args.buffer_csv)
    prop = compute_proportions(df, tcols)

    # Drop terminal point from the series (interval convention); X-axis still shown 0..30
    times_min = prop.index.values
    if len(times_min) >= 2 and (times_min[-1] - times_min[-2]) == (times_min[1] - times_min[0]):
        times_min = times_min[:-1]
        prop = prop.iloc[:-1]

    x_hr = times_min / 60.0

    # Ensure directory if saving
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        out1 = os.path.join(args.out_dir, "lines_work_education_zoom.png")
        out2 = os.path.join(args.out_dir, "lines_social_shopping_accompanying_zoom.png")
    else:
        out1 = out2 = None

    # Plot 1: Work & Education
    series_we = {"Work": prop["Work"].to_numpy(),
                 "Education": prop["Education"].to_numpy()}
    plot_zoom_yonly(x_hr, series_we,
                    "Proportion over Time — Work & Education (Y-zoom, 0–30h X-axis)",
                    save_path=out1, eps=args.eps_work, dpi=args.dpi, show=(args.out_dir is None))

    # Plot 2: Social, Shopping & Accompanying
    series_ssa = {"Social": prop["Social"].to_numpy(),
                  "Shopping": prop["Shopping"].to_numpy(),
                  "Accompanying": prop["Accompanying"].to_numpy()}
    plot_zoom_yonly(x_hr, series_ssa,
                    "Proportion over Time — Social, Shopping & Accompanying (Y-zoom, 0–30h X-axis)",
                    save_path=out2, eps=args.eps_ssa, dpi=args.dpi, show=(args.out_dir is None))

if __name__ == "__main__":
    main()
