import os
import csv
import numpy as np
from typing import Dict, List, Tuple
from ananke_abm.models.gen_schedule.compare.utils import ensure_dir
from ananke_abm.models.gen_schedule.losses.jsd import jsd


def _ngram_start_histograms(Y: np.ndarray, n: int) -> Tuple[Dict[Tuple[int, ...], np.ndarray], int]:
    """
    For each n-gram key, build a histogram over *start bins* t=0..T-n
    counting how often that n-gram starts at t across all sequences.

    Returns:
      hists: dict key -> np.ndarray shape (Tn,) with start-time counts
      Tn   : number of valid start positions (T - n + 1)
    """
    N, T = Y.shape
    if n < 1 or n > T:
        return {}, max(0, T - n + 1)

    Tn = T - n + 1
    hists: Dict[Tuple[int, ...], np.ndarray] = {}

    for i in range(N):
        seq = Y[i]
        # sliding windows of length n
        for t in range(Tn):
            key = tuple(int(x) for x in seq[t:t+n])
            h = hists.get(key)
            if h is None:
                h = np.zeros(Tn, dtype=np.float64)
                hists[key] = h
            h[t] += 1.0

    return hists, Tn


def _normalize_hist(h: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = float(h.sum())
    if s < eps:
        return np.zeros_like(h, dtype=np.float64)
    return (h.astype(np.float64) / s)


def _tod_jsd_core(Y_ref: np.ndarray, Y_syn: np.ndarray, n: int) -> Tuple[float, float]:
    """
    Compute macro-averaged and reference-weighted JSD between *start-time*
    distributions of n-grams in ref vs syn.

    Critical: both sides use the same Tn = (T_ref - n + 1) across all models.
    """
    # Build reference histograms and fix Tn from REF (authoritative)
    h_ref, Tn = _ngram_start_histograms(Y_ref, n)

    # Build synthetic histograms; if a key exists only on one side,
    # we still include it (the missing side becomes all-zeros over Tn).
    h_syn, _ = _ngram_start_histograms(Y_syn, n)

    keys = sorted(set(h_ref.keys()) | set(h_syn.keys()))
    if not keys:
        return 0.0, 0.0

    # Precompute ref weights (support) for weighted averaging
    ref_support = {k: float(h_ref.get(k, np.zeros(Tn)).sum()) for k in keys}
    total_support = sum(ref_support.values())
    # Avoid zero total support; if zero, fall back to macro
    use_weights = total_support > 0.0

    jsd_vals = []
    weights = []

    for k in keys:
        # Get histograms; if missing on a side, use zeros of shape (Tn,)
        h_r = h_ref.get(k)
        if h_r is None:
            h_r = np.zeros(Tn, dtype=np.float64)
        else:
            # Guard against shape drift from earlier bugs: coerce to Tn
            if h_r.shape[0] != Tn:
                # pad or trim deterministically; but this should not happen after fix
                tmp = np.zeros(Tn, dtype=np.float64)
                tmp[:min(Tn, h_r.shape[0])] = h_r[:min(Tn, h_r.shape[0])]
                h_r = tmp

        h_s = h_syn.get(k)
        if h_s is None:
            h_s = np.zeros(Tn, dtype=np.float64)
        else:
            if h_s.shape[0] != Tn:
                tmp = np.zeros(Tn, dtype=np.float64)
                tmp[:min(Tn, h_s.shape[0])] = h_s[:min(Tn, h_s.shape[0])]
                h_s = tmp

        p = _normalize_hist(h_r)  # (Tn,)
        q = _normalize_hist(h_s)  # (Tn,)

        # Now p and q ALWAYS have the same length (Tn)
        jsd_val = float(jsd(p, q))
        jsd_vals.append(jsd_val)
        weights.append(ref_support[k])

    macro = float(np.mean(jsd_vals)) if jsd_vals else 0.0
    if use_weights:
        w = np.array(weights, dtype=np.float64)
        w /= (w.sum() if w.sum() > 0 else 1.0)
        weighted = float((w * np.array(jsd_vals, dtype=np.float64)).sum())
    else:
        weighted = macro

    return macro, weighted


def metric_tod_jsd_ngram(ref: Dict, models: List[Dict], outdir: str):
    """
    Writes:
      - tod_jsd_macro.csv        : columns [n, model, tod_jsd_macro]
      - tod_jsd_weighted.csv     : columns [n, model, tod_jsd_weighted]
      - tod_jsd_detail_n1.csv    : optional detailed per-activity (n=1) JSD by start bin
                                   (kept small; higher n omitted due to size)
    """
    ensure_dir(outdir)

    Y_ref = ref["Y"]
    # authoritative T from ref; models have been grid-checked elsewhere
    T_ref = Y_ref.shape[1]

    # n in {1,2,3,4} but do not exceed T_ref
    ns = [n for n in (1, 2, 3, 4) if n <= T_ref]

    macro_rows = []
    weighted_rows = []

    for n in ns:
        # Reference vs each model
        for m in models:
            name = m["name"]
            macro, weighted = _tod_jsd_core(Y_ref, m["Y"], n)
            macro_rows.append({"n": n, "model": name, "tod_jsd_macro": macro})
            weighted_rows.append({"n": n, "model": name, "tod_jsd_weighted": weighted})

    # Save macro table
    with open(os.path.join(outdir, "tod_jsd_macro.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "model", "tod_jsd_macro"])
        writer.writeheader()
        for r in macro_rows:
            writer.writerow(r)

    # Save weighted table
    with open(os.path.join(outdir, "tod_jsd_weighted.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "model", "tod_jsd_weighted"])
        writer.writeheader()
        for r in weighted_rows:
            writer.writerow(r)

    # Optional detail for n=1 only (kept compact)
    # We export per-purpose start-time JSD between ref and each model for n=1.
    # For n=1, keys are single labels (tuples of length 1).
    n = 1
    if n <= T_ref:
        # Build per-key per-time distributions once for ref
        h_ref, Tn = _ngram_start_histograms(Y_ref, n)
        # Normalize to distributions
        p_ref = {k: _normalize_hist(h) for k, h in h_ref.items()}

        # union of keys across all models with ref
        keys_union = set(p_ref.keys())
        for m in models:
            h_syn_m, _ = _ngram_start_histograms(m["Y"], n)
            keys_union |= set(h_syn_m.keys())
        keys_sorted = sorted(keys_union)

        # write rows: key (as tuple), model, JSD
        detail_path = os.path.join(outdir, "tod_jsd_detail_n1.csv")
        with open(detail_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "model", "jsd"])
            for m in models:
                name = m["name"]
                h_syn_m, _ = _ngram_start_histograms(m["Y"], n)
                for k in keys_sorted:
                    pr = p_ref.get(k, np.zeros(Tn, dtype=np.float64))
                    ps = _normalize_hist(h_syn_m.get(k, np.zeros(Tn, dtype=np.float64)))
                    # Both pr and ps are length Tn by construction
                    writer.writerow([str(k), name, float(jsd(pr, ps))])

TOD_FUNCS = {
    "tod_jsd_ngram": metric_tod_jsd_ngram,
}
