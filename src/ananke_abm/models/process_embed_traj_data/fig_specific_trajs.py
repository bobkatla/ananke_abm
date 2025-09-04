#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Fixed styling ----------
COLORS = {
    "Work": "#d88c00",          # golden/orange
    "Education": "#51a3ff",     # blue
    "Social": "#f0a93b",        # warm orange
    "Shopping": "#f2d95c",      # yellow
    "Accompanying": "#007f6e",  # teal/green
}

# ---------- Helpers ----------
def coerce_clip_minutes(s: pd.Series, upper: int) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).clip(0, upper).astype(int)

def compute_occupancy_props(
    df: pd.DataFrame,
    purposes: list[str],
    horizon_min: int = 30 * 60,
    step_min: int = 5,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Returns:
      grid_hours: array of time points in HOURS
      props: dict {purpose -> proportion array over time}
    Proportions are relative to *all people engaged in any activity* at each time step.
    """
    req_cols = {"purpose", "startime", "total_duration"}
    missing = req_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Coerce minutes
    df = df.copy()
    df["startime"] = coerce_clip_minutes(df["startime"], horizon_min)
    df["total_duration"] = coerce_clip_minutes(df["total_duration"], horizon_min)

    # Time grid
    grid = np.arange(0, horizon_min + 1, step_min, dtype=int)
    n = len(grid)
    total_counts = np.zeros(n, dtype=np.int64)
    per_counts = {p: np.zeros(n, dtype=np.int64) for p in purposes}

    # Accumulate occupancy on the grid
    for _, r in df.iterrows():
        start = int(r["startime"])
        end = int(min(horizon_min, r["startime"] + r["total_duration"]))
        if end <= start:
            continue
        start_idx = start // step_min
        end_idx = max(start_idx, min(n, (end + step_min - 1) // step_min))  # ceil(end/step)
        total_counts[start_idx:end_idx] += 1
        p = str(r["purpose"])
        if p in per_counts:
            per_counts[p][start_idx:end_idx] += 1

    # Convert to proportions
    with np.errstate(divide="ignore", invalid="ignore"):
        props = {
            p: np.where(total_counts > 0, per_counts[p] / total_counts, 0.0)
            for p in purposes
        }

    grid_hours = grid / 60.0
    return grid_hours, props

def nice_ylim(arrays: list[np.ndarray], pad: float = 1.10, min_top: float = 0.12) -> tuple[float, float]:
    """
    Auto zoom y-limit: upper bound = max(values) * pad,
    but at least min_top to avoid squeezed plots for tiny ranges.
    """
    top = max([float(a.max()) if a.size else 0.0 for a in arrays] + [0.0]) * pad
    return 0.0, max(top, min_top)

# ---------- Plotters ----------
def plot_pair(grid_h, work, edu, out_path=None):
    plt.figure(figsize=(12, 4.6))
    plt.plot(grid_h, work, label="Work", linewidth=2, color=COLORS["Work"])
    plt.plot(grid_h, edu, label="Education", linewidth=2, color=COLORS["Education"])
    y0, y1 = nice_ylim([work, edu], pad=1.10, min_top=0.42)  # leaves room for legend/title
    plt.ylim(y0, min(1.0, y1))
    plt.xlim(0, grid_h.max())
    plt.title("Share of People at Work vs Education Over the Day")
    plt.xlabel("Hour of Day")
    plt.ylabel("Proportion of People")
    plt.legend(loc="upper right")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close()
    else:
        plt.show()

def plot_triple(grid_h, social, shopping, accomp, out_path=None):
    plt.figure(figsize=(12, 4.6))
    plt.plot(grid_h, social, label="Social", linewidth=2, color=COLORS["Social"])
    plt.plot(grid_h, shopping, label="Shopping", linewidth=2, color=COLORS["Shopping"])
    plt.plot(grid_h, accomp, label="Accompanying", linewidth=2, color=COLORS["Accompanying"])
    y0, y1 = nice_ylim([social, shopping, accomp], pad=1.15, min_top=0.1)  # zoomed for small shares
    plt.ylim(y0, min(1.0, y1))
    plt.xlim(0, grid_h.max())
    plt.title("Share of People at Social, Shopping, Accompanying Over the Day")
    plt.xlabel("Hour of Day")
    plt.ylabel("Proportion of People")
    plt.legend(loc="upper right")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close()
    else:
        plt.show()

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Output two occupancy plots: (Work vs Education) and (Social, Shopping, Accompanying).")
    ap.add_argument("activities_csv", help="Path to activities CSV with columns: purpose, startime, total_duration.")
    ap.add_argument("-o1", "--out1", default=None, help="Output path for Work vs Education plot (PNG).")
    ap.add_argument("-o2", "--out2", default=None, help="Output path for Social/Shopping/Accompanying plot (PNG).")
    ap.add_argument("--step-min", type=int, default=5, help="Time resolution in minutes (default: 5).")
    ap.add_argument("--horizon-min", type=int, default=30*60, help="Horizon in minutes (default: 1800 = 30h).")
    args = ap.parse_args()

    df = pd.read_csv(args.activities_csv)

    # Pair: Work vs Education
    grid_h, props_pair = compute_occupancy_props(
        df, purposes=["Work", "Education"],
        horizon_min=args.horizon_min, step_min=args.step_min
    )
    work = props_pair["Work"]
    edu = props_pair["Education"]
    plot_pair(grid_h, work, edu, out_path=args.out1)

    # Triple: Social, Shopping, Accompanying
    grid_h2, props_triple = compute_occupancy_props(
        df, purposes=["Social", "Shopping", "Accompanying"],
        horizon_min=args.horizon_min, step_min=args.step_min
    )
    social = props_triple["Social"]
    shopping = props_triple["Shopping"]
    accomp = props_triple["Accompanying"]
    plot_triple(grid_h2, social, shopping, accomp, out_path=args.out2)

if __name__ == "__main__":
    main()
