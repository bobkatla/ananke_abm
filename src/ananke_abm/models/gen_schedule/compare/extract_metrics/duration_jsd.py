import os
import numpy as np
import csv
from typing import Dict, List, Tuple
from ananke_abm.models.gen_schedule.losses.jsd import jsd
from ananke_abm.models.gen_schedule.compare.utils import ensure_dir


# ---------- helpers ----------

def _extract_segments(Y: np.ndarray, grid_min: int) -> List[Tuple[int, int]]:
    """
    Extract (purpose, duration_min) for all contiguous segments.
    """
    N, T = Y.shape
    out = []
    for seq in Y:
        curr = seq[0]
        length = 1
        for t in range(1, T):
            if seq[t] == curr:
                length += 1
            else:
                out.append((curr, length * grid_min))
                curr = seq[t]
                length = 1
        out.append((curr, length * grid_min))
    return out


def _extract_ngrams_with_duration(Y: np.ndarray, n: int, grid_min: int) -> List[Tuple[Tuple[int, ...], float]]:
    """
    Extract (ngram_tuple, duration_min) where duration spans n consecutive bins.
    For n>1, duration = n * grid_min.
    """
    N, T = Y.shape
    res = []
    for seq in Y:
        for t in range(T - n + 1):
            ng = tuple(seq[t:t + n])
            res.append((ng, n * grid_min))
    return res


def _duration_histograms(items: List[Tuple[Tuple[int, ...], float]],
                         horizon_min: int, binsize_min: int = 10):
    """
    items: list of (ngram_tuple, duration_min)
    Returns: dict {ngram_tuple: hist (normalized), bin_edges}
    """
    if not items:
        return {}
    L = horizon_min // binsize_min + 1
    hists = {}
    for ng, dur in items:
        bucket = int(min(L - 1, dur // binsize_min))
        if ng not in hists:
            hists[ng] = np.zeros(L, dtype=np.float64)
        hists[ng][bucket] += 1
    for ng in hists:
        s = hists[ng].sum()
        if s > 0:
            hists[ng] /= s
    return hists


# ---------- main metric ----------

def _duration_jsd_core(ref_Y: np.ndarray, syn_Y: np.ndarray,
                       grid_min: int, horizon_min: int,
                       n: int, outdir: str,
                       purpose_map: Dict[str, int],
                       detail: bool = False):
    ensure_dir(outdir)
    inv_map = {v: k for k, v in purpose_map.items()}

    # 1. Extract n-grams with durations
    ref_items = _extract_ngrams_with_duration(ref_Y, n, grid_min)
    syn_items = _extract_ngrams_with_duration(syn_Y, n, grid_min)

    # 2. Build duration histograms
    ref_hist = _duration_histograms(ref_items, horizon_min)
    syn_hist = _duration_histograms(syn_items, horizon_min)

    # 3. Compute JSD for each ngram
    all_ngrams = set(ref_hist.keys()) | set(syn_hist.keys())
    jsd_list = []
    weighted_sum = 0.0
    total_ref_count = sum(v.sum() for v in ref_hist.values()) + 1e-9

    if detail:
        det_rows = []

    for ng in all_ngrams:
        p = ref_hist.get(ng, np.zeros(horizon_min // grid_min + 1, dtype=np.float64))
        q = syn_hist.get(ng, np.zeros_like(p))
        val = jsd(p, q)
        jsd_list.append(val)
        w = ref_hist.get(ng, np.zeros_like(p)).sum()
        weighted_sum += w * val

        if detail:
            label = "→".join(inv_map[i] for i in ng)
            ref_count = ref_hist.get(ng, np.zeros_like(p)).sum()
            syn_count = syn_hist.get(ng, np.zeros_like(p)).sum()
            det_rows.append({
                "ngram_label": label,
                "ref_count": ref_count,
                "syn_count": syn_count,
                "jsd": val
            })

    macro_jsd = float(np.mean(jsd_list)) if jsd_list else 0.0
    weighted_jsd = float(weighted_sum / total_ref_count) if total_ref_count > 0 else 0.0

    if detail:
        detail_path = os.path.join(outdir, f"duration_jsd_detail_n{n}.csv")
        with open(detail_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["ngram_label", "ref_count", "syn_count", "jsd"])
            writer.writeheader()
            for row in det_rows:
                writer.writerow(row)

    return macro_jsd, weighted_jsd


def metric_duration_jsd_ngram(ref: Dict, models: List[Dict], outdir: str):
    """
    Compute Duration JSD for n=1..4.  Only n=1 produces detail file.
    """
    ensure_dir(outdir)
    grid_min = ref["grid_min"]
    horizon_min = ref["horizon_min"]
    purpose_map = ref["purpose_map"]
    ref_Y = ref["Y"]

    for n in [1, 2, 3, 4]:
        rows = [{"model": "ref", "macro_jsd": 0.0, "weighted_jsd": 0.0}]
        for m in models:
            macro, weighted = _duration_jsd_core(
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

        csv_path = os.path.join(outdir, f"duration_jsd_n{n}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "macro_jsd", "weighted_jsd"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)


# ---------- registry ----------
DURATION_FUNCS = {
    "duration_jsd_ngram": metric_duration_jsd_ngram,
}
