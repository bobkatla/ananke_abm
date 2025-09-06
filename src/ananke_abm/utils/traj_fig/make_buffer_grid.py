#!/usr/bin/env python3
"""
Build a buffer-analysis CSV from home-bound trajectories.

- Rows: person id (persid)
- Columns: 0, step, 2*step, ..., maxtime
  * Each column t encodes what the person is doing in [t, t+step).
  * If an activity starts inside a bin, that bin is labeled with the new activity.
  * Otherwise, the bin carries forward the previous activity’s label.
- The last column (== maxtime) is always "Home".

Args:
  input_file: trajectory CSV with [persid, hhid, stopno, purpose, startime, total_duration]
  output_file: path for the buffer grid CSV
  --maxtime: horizon in minutes (default 1800)
  --step: bin size in minutes (default 5)
"""

import argparse
import pandas as pd
from tqdm import tqdm

def build_buffer_grid(df: pd.DataFrame, maxtime: int, step: int) -> pd.DataFrame:
    required_cols = {"persid", "stopno", "purpose", "startime", "total_duration"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input file missing required columns: {required_cols - set(df.columns)}")

    df = df.sort_values(["persid", "stopno"]).reset_index(drop=True)
    bin_starts = list(range(0, maxtime, step)) + [maxtime]

    persons = df["persid"].unique()
    out_rows = {}

    for pid, g in tqdm(df.groupby("persid"), total=len(persons), desc="Building grid"):
        g = g.sort_values("stopno").reset_index(drop=True)
        acts = [
            {"start": float(r["startime"]), "dur": float(r["total_duration"]), "purpose": str(r["purpose"])}
            for _, r in g.iterrows()
        ]

        starts_by_bin = {}
        for a in acts:
            b = int((a["start"] // step) * step)
            if 0 <= b < maxtime:
                starts_by_bin[b] = a["purpose"]

        row_values = {}
        current_label = acts[0]["purpose"] if acts else "Home"

        for b in bin_starts[:-1]:
            if b in starts_by_bin:
                current_label = starts_by_bin[b]
            row_values[b] = current_label

        row_values[maxtime] = "Home"
        out_rows[pid] = row_values

    cols_sorted = sorted(bin_starts)
    out_df = pd.DataFrame.from_dict(out_rows, orient="index")[cols_sorted]
    out_df.index.name = "persid"
    return out_df.reset_index()

def main():
    parser = argparse.ArgumentParser(description="Create buffer-analysis CSV from trajectories.")
    parser.add_argument("input_file", type=str, help="Path to input trajectory CSV")
    parser.add_argument("output_file", type=str, help="Path to output buffer CSV")
    parser.add_argument("--maxtime", type=int, default=1800, help="Total horizon in minutes (default 1800)")
    parser.add_argument("--step", type=int, default=5, help="Bin size in minutes (default 5)")
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)
    grid = build_buffer_grid(df, maxtime=args.maxtime, step=args.step)
    grid.to_csv(args.output_file, index=False)

    print(f"Done. Wrote: {args.output_file}")
    print(f"Shape: {grid.shape[0]} rows x {grid.shape[1]} columns (including 'persid')")

if __name__ == "__main__":
    main()
