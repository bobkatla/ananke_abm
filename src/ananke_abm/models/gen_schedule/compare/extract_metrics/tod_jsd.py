import os
import numpy as np
import csv
from typing import Dict, List, Tuple
from ananke_abm.models.gen_schedule.losses.jsd import jsd
from ananke_abm.models.gen_schedule.compare.utils import ensure_dir


def _extract_ngrams(Y: np.ndarray, n: int) -> List[Tuple[Tuple[int, ...], int]]:
    """
    Extract all overlapping n-grams and their start bin index.
    Returns list of (ngram_tuple, start_bin)
    """
    N, T = Y.shape
    res = []
    for seq in Y:
        for t in range(T - n + 1):
            ngram = tuple(seq[t:t+n])
            res.append((ngram, t))
    return res


def _tod_histograms_from_ngrams(ngrams, grid_min, horizon_min, binsize_min=10):
    """
    Build normalized histogram per n-gram of start times.
    Returns dict: {ngram_tuple: (hist, bin_edges)}
    """
    if not ngrams:
        return {}

    L = horizon_min // binsize_min
    hist_dict = {}
    for ng, t in ngrams:
        start_min = t * grid_min
        bucket = int(start_min // binsize_min)
        if ng not in hist_dict:
            hist_dict[ng] = np.zeros(L, dtype=np.float64)
        hist_dict[ng][bucket] += 1

    # normalize to probabilities
    for ng in hist_dict.keys():
        s = hist_dict[ng].sum()
        if s > 0:
            hist_dict[ng] /= s
    return hist_dict


def _tod_jsd_core(ref_Y: np.ndarray, syn_Y: np.ndarray,
                  grid_min: int, horizon_min: int,
                  n: int, outdir: str,
                  purpose_map: Dict[str, int],
                  detail: bool = False):
    """
    Compute JSD between time-of-day distributions for all n-grams.
    Returns macro and weighted averages.
    """
    ensure_dir(outdir)
    inv_map = {v: k for k, v in purpose_map.items()}

    # 1. Extract n-grams
    ref_ngrams = _extract_ngrams(ref_Y, n)
    syn_ngrams = _extract_ngrams(syn_Y, n)

    # 2. Histograms of start times
    ref_hist = _tod_histograms_from_ngrams(ref_ngrams, grid_min, horizon_min)
    syn_hist = _tod_histograms_from_ngrams(syn_ngrams, grid_min, horizon_min)

    # 3. Collect union of n-grams
    all_ngrams = set(ref_hist.keys()) | set(syn_hist.keys())
    jsd_list = []
    weighted_sum = 0.0
    total_ref_count = sum(v.sum() for v in ref_hist.values()) + 1e-9

    # Optional detailed table for n=1
    if detail:
        det_rows = []

    for ng in all_ngrams:
        p = ref_hist.get(ng, np.zeros(horizon_min // grid_min, dtype=np.float64))
        q = syn_hist.get(ng, np.zeros_like(p))
        val = jsd(p, q)
        jsd_list.append(val)

        w = ref_hist.get(ng, np.zeros_like(p)).sum()
        weighted_sum += w * val

        if detail:
            label = "→".join(inv_map[i] for i in ng)
            ref_count = ref_hist.get(ng, np.zeros_like(p)).sum()
            syn_count = syn_hist.get(ng, np.zeros_like(p)).sum()
            ref_mean = float(np.argmax(p) * grid_min) if ref_count > 0 else np.nan
            syn_mean = float(np.argmax(q) * grid_min) if syn_count > 0 else np.nan
            det_rows.append({
                "ngram_label": label,
                "ref_count": ref_count,
                "syn_count": syn_count,
                "jsd": val,
                "ref_mean_start_min": ref_mean,
                "syn_mean_start_min": syn_mean,
            })

    macro_jsd = float(np.mean(jsd_list)) if jsd_list else 0.0
    weighted_jsd = float(weighted_sum / total_ref_count) if total_ref_count > 0 else 0.0

    if detail:
        detail_path = os.path.join(outdir, f"tod_jsd_detail_n{n}.csv")
        with open(detail_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["ngram_label", "ref_count", "syn_count", "jsd",
                            "ref_mean_start_min", "syn_mean_start_min"]
            )
            writer.writeheader()
            for row in det_rows:
                writer.writerow(row)

    return macro_jsd, weighted_jsd


def metric_tod_jsd_ngram(ref: Dict, models: List[Dict], outdir: str):
    """
    Compute ToD JSD for n=1..4.  Only n=1 produces detail file.
    Writes one summary CSV for each n.
    """
    ensure_dir(outdir)
    grid_min = ref["grid_min"]
    horizon_min = ref["horizon_min"]
    purpose_map = ref["purpose_map"]

    ref_Y = ref["Y"]

    for n in [1, 2, 3, 4]:
        rows = [{"model": "ref", "macro_jsd": 0.0, "weighted_jsd": 0.0}]
        for m in models:
            macro, weighted = _tod_jsd_core(
                ref_Y,
                m["Y"],
                grid_min,
                horizon_min,
                n,
                outdir,
                purpose_map,
                detail=(n == 1)
            )
            rows.append({
                "model": m["name"],
                "macro_jsd": macro,
                "weighted_jsd": weighted
            })

        csv_path = os.path.join(outdir, f"tod_jsd_n{n}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "macro_jsd", "weighted_jsd"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

TOD_FUNCS = {
    "tod_jsd_ngram": metric_tod_jsd_ngram,
}
