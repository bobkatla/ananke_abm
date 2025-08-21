#!/usr/bin/env python3
"""
make_purposes.py

Builds a purposes catalog CSV (purposes.csv) from an activities table like VISTA.
Intended to feed Dual Space AE FiLM/meta.

Inputs
------
- activities CSV with columns similar to:
  persid, hhid, stopno, purpose, startime, total_duration
  * startime and total_duration are in MINUTES (the script handles units).
  * Column names are configurable via CLI flags.

Optional
--------
- A YAML/CSV mapping to override categorical tags like "is_primary".
- Persons CSV to count person-days robustly (if multiple days present).

Outputs
-------
- purposes.csv with one row per distinct purpose label, including:
  * purpose
  * n_occurrences
  * n_person_days_with_purpose
  * person_day_participation_rate
  * mean_duration_min, median_duration_min, p10_duration_min, p90_duration_min, iqr_duration_min, std_duration_min
  * mean_start_min, median_start_min, p10_start_min, p90_start_min, std_start_min
  * start_circ_var (0..1), duration_cv (>=0), flexibility (0..1)
  * is_primary (Y/N)  [overridable; default heuristic]
  * can_open_close_day (Y/N)  [heuristic: Home=True, otherwise False unless overridden]
  * notes (free text from mapping if provided)

Usage
-----
python make_purposes.py \
  --activities_csv /mnt/data/activities_homebound_wd.csv \
  --out_csv /mnt/data/purposes.csv

Optional flags:
  --persons_csv /mnt/data/persons_homebound_wd.csv
  --purpose_col purpose --start_col startime --dur_col total_duration
  --person_col persid --day_col day
  --override_yaml /path/to/purpose_overrides.yaml
  --override_csv  /path/to/purpose_overrides.csv

The overrides file (YAML or CSV) can contain columns:
  - purpose (required key)
  - is_primary (Y/N/True/False/1/0)
  - can_open_close_day (Y/N/True/False/1/0)
  - notes (free text)
Any other columns will be copied through as extra meta.
"""

import argparse
import math
import sys
import warnings
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore
    _YAML_OK = True
except Exception:
    _YAML_OK = False

def _boolify(x: Any) -> Optional[bool]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"y","yes","true","1"}: return True
    if s in {"n","no","false","0"}: return False
    return None

def _safe_col(df: pd.DataFrame, preferred: str, fallbacks: list[str]) -> str:
    """Return an existing column name from df that matches preferred or any fallback, case-insensitive."""
    cols_lower = {c.lower(): c for c in df.columns}
    if preferred and preferred.lower() in cols_lower:
        return cols_lower[preferred.lower()]
    for name in [*fallbacks]:
        if name.lower() in cols_lower:
            return cols_lower[name.lower()]
    raise KeyError(f"Could not find any of columns { [preferred, *fallbacks] } in CSV. Available: {list(df.columns)}")

def _circular_variance_minutes(mins: np.ndarray, period_min: float) -> float:
    """
    Circular variance in [0,1], where 0 = perfectly concentrated, 1 = maximally dispersed on circle.
    mins: times in minutes modulo period_min
    """
    if len(mins) == 0: return float("nan")
    theta = (mins % period_min) / period_min * 2*np.pi
    C = np.mean(np.cos(theta))
    S = np.mean(np.sin(theta))
    R = math.sqrt(C*C + S*S)
    return 1 - R  # 0..1

def _cv(x: np.ndarray) -> float:
    """Coefficient of variation (std/mean); returns NaN if mean ~ 0."""
    if len(x) == 0: return float("nan")
    m = np.mean(x)
    s = np.std(x, ddof=1) if len(x) > 1 else 0.0
    return float("nan") if abs(m) < 1e-9 else float(s / m)

def _minmax01(x: pd.Series) -> pd.Series:
    if x.isna().all():
        return x
    mn, mx = x.min(), x.max()
    if pd.isna(mn) or pd.isna(mx) or abs(mx - mn) < 1e-12:
        return pd.Series(0.5, index=x.index)  # constant -> mid
    return (x - mn) / (mx - mn)

def _load_overrides(override_yaml: Optional[str], override_csv: Optional[str]) -> Optional[pd.DataFrame]:
    df = None
    if override_yaml:
        if not _YAML_OK:
            warnings.warn("pyyaml not installed; cannot read YAML. Install pyyaml or use CSV overrides.")
        else:
            with open(override_yaml, "r", encoding="utf-8") as f:
                y = yaml.safe_load(f) or {}
            # accept either list of dicts or dict keyed by purpose
            if isinstance(y, dict) and "items" not in y and "purpose" not in y:
                rows = []
                for k, v in y.items():
                    row = {"purpose": k}
                    if isinstance(v, dict):
                        row.update(v)
                    else:
                        row["notes"] = str(v)
                    rows.append(row)
                df = pd.DataFrame(rows)
            else:
                df = pd.DataFrame(y)
    if override_csv:
        d2 = pd.read_csv(override_csv)
        df = d2 if df is None else pd.concat([df, d2], axis=0, ignore_index=True)
    if df is not None and "purpose" not in df.columns:
        raise ValueError("Overrides must include a 'purpose' column or be a YAML mapping keyed by purpose.")
    if df is not None:
        # normalize booleans
        for c in list(df.columns):
            if c in {"is_primary","can_open_close_day"}:
                df[c] = df[c].apply(_boolify)
        # de-duplicate by purpose, keep last
        df = df.drop_duplicates(subset=["purpose"], keep="last")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activities_csv", required=True, type=str)
    ap.add_argument("--out_csv", required=True, type=str)
    ap.add_argument("--persons_csv", type=str, default=None)

    ap.add_argument("--purpose_col", type=str, default="purpose")
    ap.add_argument("--start_col", type=str, default="startime")  # VISTA name
    ap.add_argument("--dur_col", type=str, default="total_duration")
    ap.add_argument("--person_col", type=str, default="persid")
    ap.add_argument("--day_col", type=str, default=None)  # if None, infer single-day per person

    ap.add_argument("--day_horizon_minutes", type=float, default=24*60.0,
                    help="Time wrap period in minutes for circular stats (e.g., 1800 for 30h horizon).")

    ap.add_argument("--override_yaml", type=str, default=None)
    ap.add_argument("--override_csv", type=str, default=None)

    args = ap.parse_args()

    # Load activities
    act = pd.read_csv(args.activities_csv)
    # Resolve columns (with forgiving aliases)
    purpose_col = _safe_col(act, args.purpose_col, ["purpose","activity","activity_type","act"])
    start_col   = _safe_col(act, args.start_col,   ["startime","start_time","start","start_min","start_minutes"])
    dur_col     = _safe_col(act, args.dur_col,     ["total_duration","duration","dur","dur_min","duration_minutes"])
    person_col  = _safe_col(act, args.person_col,  ["persid","person_id","pid","person"])
    day_col     = args.day_col
    if day_col:
        day_col = _safe_col(act, day_col, ["day","date","diary_day","day_id"])
    else:
        # create pseudo-day = 1 per person if not present
        day_col = "__day__"
        act[day_col] = 1

    # basic cleaning
    for c in [start_col, dur_col]:
        act[c] = pd.to_numeric(act[c], errors="coerce")
    act = act.dropna(subset=[purpose_col, start_col, dur_col]).copy()

    # standardize purpose strings
    act["_purpose_norm"] = (act[purpose_col].astype(str)
                            .str.strip()
                            .str.replace(r"\s+", " ", regex=True)
                            .str.title())  # Title Case
    # keep explicit "Home" capitalization
    act["_purpose_norm"] = act["_purpose_norm"].replace({"Home": "Home"})

    # robust person-day identifier
    act["_person_day"] = act[person_col].astype(str) + "||" + act[day_col].astype(str)

    # Duration and start in minutes
    dur_min = act[dur_col].to_numpy()
    start_min = act[start_col].to_numpy()

    # Aggregate stats per purpose
    gb = act.groupby("_purpose_norm", sort=False)
    rows = []
    all_person_days = act["_person_day"].nunique()

    # precompute participation per purpose
    purpose_to_person_days = act.groupby("_purpose_norm")["_person_day"].nunique()

    # compute stats safely
    for purpose, g in gb:
        d = {}
        d["purpose"] = purpose
        d["n_occurrences"] = int(len(g))
        n_pd = int(g["_person_day"].nunique())
        d["n_person_days_with_purpose"] = n_pd
        d["person_day_participation_rate"] = n_pd / all_person_days if all_person_days > 0 else np.nan

        # durations
        dur = pd.to_numeric(g[dur_col], errors="coerce").dropna().to_numpy()
        start = pd.to_numeric(g[start_col], errors="coerce").dropna().to_numpy()

        def _quant(a, q):
            return float(np.quantile(a, q)) if a.size else np.nan

        d["mean_duration_min"] = float(np.mean(dur)) if dur.size else np.nan
        d["median_duration_min"] = _quant(dur, 0.5)
        d["p10_duration_min"] = _quant(dur, 0.10)
        d["p90_duration_min"] = _quant(dur, 0.90)
        d["iqr_duration_min"] = (_quant(dur, 0.75) - _quant(dur, 0.25)) if dur.size else np.nan
        d["std_duration_min"] = float(np.std(dur, ddof=1)) if dur.size > 1 else 0.0
        d["duration_cv"] = _cv(dur)

        # starts
        d["mean_start_min"] = float(np.mean(start)) if start.size else np.nan
        d["median_start_min"] = _quant(start, 0.5)
        d["p10_start_min"] = _quant(start, 0.10)
        d["p90_start_min"] = _quant(start, 0.90)
        d["std_start_min"] = float(np.std(start, ddof=1)) if start.size > 1 else 0.0

        # circular variance of start times (0..1) on the provided horizon
        d["start_circ_var"] = _circular_variance_minutes(start, args.day_horizon_minutes) if start.size else np.nan

        rows.append(d)

    out = pd.DataFrame(rows)

    # Flexibility (0..1): combine normalized circular variance of starts and CV of durations
    out["flexibility"] = _minmax01(out["start_circ_var"].astype(float)) * 0.6 + _minmax01(out["duration_cv"].astype(float)) * 0.4

    # Heuristic categorical tags
    # Defaults: Home/Work/Education primary; visit/shop/medical/escort/other non-primary.
    # can_open_close_day: Home True; others False unless overridden.
    def _default_primary(p: str) -> bool:
        p_l = p.lower()
        if p_l in {"home","work","education","school","uni","university"}:
            return True
        return False

    def _default_open_close(p: str) -> bool:
        return p.lower() == "home"

    out["is_primary"] = out["purpose"].apply(_default_primary)
    out["can_open_close_day"] = out["purpose"].apply(_default_open_close)

    # Participation-based "skip probability" (proxy): 1 - normalized participation rate
    # Normalize across purposes to [0,1] for interpretability.
    out["skip_probability"] = 1.0 - _minmax01(out["person_day_participation_rate"].astype(float))

    # Merge overrides if provided
    ov = _load_overrides(args.override_yaml, args.override_csv)
    if ov is not None:
        # copy through any extra columns
        for c in ov.columns:
            if c not in {"purpose","is_primary","can_open_close_day","notes"}:
                # will be copied as extra meta
                pass
        out = out.merge(ov, on="purpose", how="left", suffixes=("","__ov"))
        # apply overrides if present
        for c in ["is_primary","can_open_close_day","notes"]:
            c_ov = c + "__ov"
            if c_ov in out.columns:
                out[c] = np.where(out[c_ov].notna(), out[c_ov], out[c])
                out = out.drop(columns=[c_ov])
        # preserve extra override columns
        extra_cols = [c for c in ov.columns if c not in {"purpose","is_primary","can_open_close_day","notes"}]
        # these may already be present; if duplicates existed, pandas merge handled with suffixes
        # we'll rename any suffixed columns back to original if needed
        for c in extra_cols:
            c_left = c
            c_right = c + "__ov"
            if c_right in out.columns and c_left in out.columns:
                out[c_left] = np.where(out[c_right].notna(), out[c_right], out[c_left])
                out = out.drop(columns=[c_right])

    # Order columns nicely
    lead = [
        "purpose",
        "is_primary",
        "can_open_close_day",
        "person_day_participation_rate",
        "skip_probability",
        "n_occurrences",
        "n_person_days_with_purpose",
        "mean_duration_min","median_duration_min","p10_duration_min","p90_duration_min","iqr_duration_min","std_duration_min","duration_cv",
        "mean_start_min","median_start_min","p10_start_min","p90_start_min","std_start_min","start_circ_var",
        "flexibility",
        "notes"
    ]
    # include any other columns at the end
    cols = [c for c in lead if c in out.columns] + [c for c in out.columns if c not in lead]
    out = out[cols]

    # Sort by participation descending then by purpose
    out = out.sort_values(["person_day_participation_rate","purpose"], ascending=[False, True]).reset_index(drop=True)

    # Cast booleans to Y/N
    for c in ["is_primary","can_open_close_day"]:
        if c in out.columns:
            out[c] = out[c].map({True: "Y", False: "N"}).fillna("")

    # Write
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[make_purposes] wrote {out_path} with {len(out)} purposes.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[make_purposes] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
