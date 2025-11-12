# duration_jsd.py
import os
import csv
import numpy as np
from typing import Dict, List, Tuple
from ananke_abm.models.gen_schedule.losses.jsd import jsd
from ananke_abm.models.gen_schedule.compare.utils import ensure_dir

# -------- helpers --------

def _segments_from_Y(Y: np.ndarray) -> List[List[Tuple[int, int, int]]]:
    """
    Convert each row of Y (T-long labels) into segments:
      returns per-person list of (label, start_bin, length_bins).
    """
    out = []
    for row in Y:
        segs = []
        T = row.shape[0]
        s = 0
        cur = row[0]
        for t in range(1, T):
            if row[t] != cur:
                segs.append((int(cur), s, t - s))
                cur = row[t]
                s = t
        segs.append((int(cur), s, T - s))
        out.append(segs)
    return out

def _collect_durations_activity(segs_all: List[List[Tuple[int, int, int]]], P: int, grid_min: int) -> Dict[int, List[int]]:
    """
    For n=1: durations per activity from contiguous runs.
    Returns: dict p -> list of durations in minutes.
    """
    d = {p: [] for p in range(P)}
    for segs in segs_all:
        for p, _, ln in segs:
            d[p].append(int(ln * grid_min))
    return d

def _collect_durations_ngram_segments(
    segs_all: List[List[Tuple[int, int, int]]],
    n: int,
    grid_min: int
) -> Dict[Tuple[int, ...], List[int]]:
    """
    For n>=2: scan OVER SEGMENTS, not bins.
    Collect total duration (sum over the n matched segment lengths) in minutes
    for each ordered segment n-gram key = (p1,...,pn).
    """
    d: Dict[Tuple[int, ...], List[int]] = {}
    for segs in segs_all:
        S = len(segs)
        if S < n:
            continue
        # slide window over segments
        for i in range(S - n + 1):
            key = tuple(segs[i + k][0] for k in range(n))
            tot_bins = sum(segs[i + k][2] for k in range(n))
            dur_min = int(tot_bins * grid_min)
            d.setdefault(key, []).append(dur_min)
    return d

def _hist_prob(values: List[int], bin_edges: np.ndarray) -> np.ndarray:
    """
    Turn raw duration samples (minutes) into a probability vector via fixed bins.
    """
    if len(values) == 0:
        return np.zeros(len(bin_edges) - 1, dtype=np.float64)
    hist, _ = np.histogram(np.asarray(values, dtype=np.float64), bins=bin_edges)
    hist = hist.astype(np.float64)
    s = hist.sum()
    return hist / s if s > 0 else np.zeros_like(hist)

# -------- metric entry point --------

def metric_duration_jsd_ngram(ref: Dict, models: List[Dict], outdir: str, n: int = 1,
                              max_minutes: int = 1440, bin_width: int = 10):
    """
    Duration JSD for n-grams (n=1..4).
    n=1 -> per-activity duration distributions from contiguous runs.
    n>=2 -> ordered segment n-grams; duration = sum of the n segment lengths (minutes).
    Writes:
      duration_jsd_n{n}.csv: model, key, count_ref, count_model, jsd
    Note: keys are ints for n=1 and tuples for n>=2; counts are sample counts used to form histograms.
    """
    ensure_dir(outdir)

    Y_ref = ref["Y"]
    grid_min = int(ref["grid_min"])
    P = len(ref["purpose_map"])

    segs_ref = _segments_from_Y(Y_ref)
    if n == 1:
        dur_ref = _collect_durations_activity(segs_ref, P, grid_min)               # dict p -> [mins]
        keys_ref = sorted(dur_ref.keys())
    else:
        dur_ref = _collect_durations_ngram_segments(segs_ref, n, grid_min)         # dict (p1..pn) -> [mins]
        keys_ref = sorted(dur_ref.keys())

    # fixed duration bins
    bin_edges = np.arange(0, max_minutes + bin_width, bin_width, dtype=np.float64)

    rows = []
    # reference self-row (optional summary per key)
    # We will only write model rows for comparatives + one summary "ref" row per key with jsd=0
    for key in keys_ref:
        rows.append({
            "model": "ref",
            "key": key if isinstance(key, tuple) else int(key),
            "count_ref": len(dur_ref[key]),
            "count_model": 0,
            "jsd": 0.0
        })

    for m in models:
        Y_m = m["Y"]
        segs_m = _segments_from_Y(Y_m)

        if n == 1:
            dur_m = _collect_durations_activity(segs_m, P, grid_min)
            all_keys = sorted(set(keys_ref) | set(dur_m.keys()))
        else:
            dur_m = _collect_durations_ngram_segments(segs_m, n, grid_min)
            all_keys = sorted(set(keys_ref) | set(dur_m.keys()))

        for key in all_keys:
            ref_vals = dur_ref.get(key, [])
            mod_vals = dur_m.get(key, [])
            p = _hist_prob(ref_vals, bin_edges)
            q = _hist_prob(mod_vals, bin_edges)
            val = jsd(p, q)
            rows.append({
                "model": m["name"],
                "key": key if isinstance(key, tuple) else int(key),
                "count_ref": len(ref_vals),
                "count_model": len(mod_vals),
                "jsd": float(val)
            })

    # write CSV
    path = os.path.join(outdir, f"duration_jsd_n{n}.csv")
    # normalize key to string for CSV
    for r in rows:
        if isinstance(r["key"], tuple):
            r["key"] = "|".join(str(x) for x in r["key"])
    fieldnames = ["model", "key", "count_ref", "count_model", "jsd"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

# ---------- registry ----------
DURATION_FUNCS = {
    "duration_jsd_ngram": metric_duration_jsd_ngram,
}
