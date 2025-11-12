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

def metric_duration_jsd_ngram_specific(ref: Dict, models: List[Dict], outdir: str, n: int = 1,
                              max_minutes: int = 1440, bin_width: int = 5, output_details: bool = False):
    """
    Duration JSD for n-grams (n=1..4).
    n=1 -> per-activity duration distributions from contiguous runs.
    n>=2 -> ordered segment n-grams; duration = sum of the n segment lengths (minutes).

    Writes:
      duration_jsd_macro_n{n}.csv: model, macro_jsd, weighted_jsd, K_keys
      duration_jsd_n{n}.csv      : model, key, count_ref, count_model, jsd   (ONLY for n==1)
    """
    ensure_dir(outdir)

    Y_ref = ref["Y"]
    grid_min = int(ref["grid_min"])
    P = len(ref["purpose_map"])

    segs_ref = _segments_from_Y(Y_ref)
    if n == 1:
        dur_ref = _collect_durations_activity(segs_ref, P, grid_min)      # p -> [mins]
        keys_ref = sorted(dur_ref.keys())
    else:
        dur_ref = _collect_durations_ngram_segments(segs_ref, n, grid_min)  # (p1..pn) -> [mins]
        keys_ref = sorted(dur_ref.keys())

    # fixed duration bins
    bin_edges = np.arange(0, max_minutes + bin_width, bin_width, dtype=np.float64)

    # ---- per-key reference histograms and counts (for weighting) ----
    ref_hist = {}
    ref_counts = {}
    for key in keys_ref:
        vals = dur_ref.get(key, [])
        ref_hist[key] = _hist_prob(vals, bin_edges)
        ref_counts[key] = len(vals)

    # ---- macro summary rows (always) ----
    macro_rows = []
    # reference summary row
    K = len(keys_ref)
    macro_rows.append({"model": "ref", "macro_jsd": 0.0, "weighted_jsd": 0.0, "K_keys": K})

    # optional detailed rows (only for n==1)
    detail_rows = []
    if output_details:
        for key in keys_ref:
            detail_rows.append({
                "model": "ref",
                "key": int(key),
                "count_ref": ref_counts[key],
                "count_model": 0,
                "jsd": 0.0
            })

    # ---- models ----
    for m in models:
        Y_m = m["Y"]
        segs_m = _segments_from_Y(Y_m)
        if output_details:
            dur_m = _collect_durations_activity(segs_m, P, grid_min)
            all_keys = sorted(set(keys_ref) | set(dur_m.keys()))
        else:
            dur_m = _collect_durations_ngram_segments(segs_m, n, grid_min)
            all_keys = sorted(set(keys_ref) | set(dur_m.keys()))

        jsd_vals = []
        weights = []
        for key in all_keys:
            p = ref_hist.get(key, np.zeros(len(bin_edges) - 1, dtype=np.float64))
            q = _hist_prob(dur_m.get(key, []), bin_edges)
            val = jsd(p, q)
            jsd_vals.append(val)
            # weight by ref sample count for this key (0 if unseen in ref)
            weights.append(float(ref_counts.get(key, 0)))

            if n == 1:
                detail_rows.append({
                    "model": m["name"],
                    "key": int(key),
                    "count_ref": ref_counts.get(key, 0),
                    "count_model": len(dur_m.get(key, [])),
                    "jsd": float(val)
                })

        # macro = unweighted mean over union keys
        macro = float(np.mean(jsd_vals)) if jsd_vals else 0.0
        # weighted by ref counts over keys that appear in ref
        w = np.asarray(weights, dtype=np.float64)
        v = np.asarray(jsd_vals, dtype=np.float64)
        if w.sum() > 0:
            wmacro = float((w * v).sum() / w.sum())
        else:
            wmacro = 0.0

        macro_rows.append({
            "model": m["name"],
            "macro_jsd": macro,
            "weighted_jsd": wmacro,
            "K_keys": int(len(all_keys))
        })

    # ---- write macro summary (always) ----
    macro_path = os.path.join(outdir, f"duration_jsd_macro_n{n}.csv")
    with open(macro_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "macro_jsd", "weighted_jsd", "K_keys"])
        w.writeheader()
        for r in macro_rows:
            w.writerow(r)

    # ---- write details (only n==1) ----
    if output_details:
        detail_path = os.path.join(outdir, f"duration_jsd_n{n}.csv")
        with open(detail_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["model", "key", "count_ref", "count_model", "jsd"])
            w.writeheader()
            for r in detail_rows:
                w.writerow(r)


def metric_duration_jsd_ngram(ref: Dict, models: List[Dict], outdir: str):
    """
    Detailed I want
    """
    for n in [1, 2, 3, 4]:
        metric_duration_jsd_ngram_specific(
            ref=ref,
            models=models,
            outdir=outdir,
            n=n,
            max_minutes=1440,
            bin_width=5,
            output_details=(n == 1)
        )
        


# ---------- registry ----------
DURATION_FUNCS = {
    "duration_jsd_ngram": metric_duration_jsd_ngram,
}
