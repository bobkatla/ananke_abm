from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Basic structures
# -----------------------------
def build_clock_grid(step_minutes: int) -> np.ndarray:
    """
    Returns clock boundaries [0, step, 2*step, ..., < 1440].
    """
    step = int(step_minutes)
    if step <= 0:
        raise ValueError("step_minutes must be positive")
    return np.arange(0, 1440, step, dtype=np.int32)


def activities_csv_to_segments(csv_path: str) -> List[List[dict]]:
    """
    Returns list over people; each is a list of dicts with keys:
    {'purpose': str, 'startime': int_minutes, 'total_duration': int_minutes}
    Assumes the CSV has columns: persid, stopno, purpose, startime, total_duration
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        return []
    segments: List[List[dict]] = []
    for _, g in df.sort_values(["persid", "stopno"]).groupby("persid"):
        person: List[dict] = []
        for _idx, r in g.iterrows():
            person.append({
                "purpose": str(r["purpose"]),
                "startime": int(r["startime"]),
                "total_duration": int(r["total_duration"]),
            })
        segments.append(person)
    return segments


# -----------------------------
# Conversions (if needed by validate path)
# -----------------------------
def labels_to_segments(labels: np.ndarray, idx2purpose: List[str], step_minutes: int) -> List[List[dict]]:
    """
    Convert a batch of label sequences on a uniform evaluation grid into segments.
    labels: shape [B, L] of integer purpose indices
    step_minutes: number of minutes per grid step
    Returns list over batch; each list contains dicts with purpose/startime/total_duration (minutes)
    """
    if labels.size == 0:
        return []
    B, L = labels.shape
    out: List[List[dict]] = []
    for b in range(B):
        y = labels[b]
        if L == 0:
            out.append([])
            continue
        start = 0
        person: List[dict] = []
        for i in range(1, L):
            if y[i] != y[i - 1]:
                pidx = int(y[i - 1])
                t0_m = start * step_minutes
                dur_m = (i - start) * step_minutes
                person.append({
                    "purpose": idx2purpose[pidx],
                    "startime": int(t0_m),
                    "total_duration": int(dur_m),
                })
                start = i
        pidx = int(y[-1])
        t0_m = start * step_minutes
        dur_m = (L - start) * step_minutes
        person.append({
            "purpose": idx2purpose[pidx],
            "startime": int(t0_m),
            "total_duration": int(dur_m),
        })
        out.append(person)
    return out


# -----------------------------
# Aggregations
# -----------------------------
def _accumulate_presence_for_segment(acc: np.ndarray, start_m: int, dur_m: int, bin_m: int):
    if dur_m <= 0:
        return
    end_m = start_m + dur_m
    if end_m <= 0 or start_m >= 1440:
        return
    s = max(0, int(start_m))
    e = min(1440, int(end_m))
    if e <= s:
        return
    b0 = s // bin_m
    b1 = (e - 1) // bin_m
    for b in range(b0, b1 + 1):
        bin_start = b * bin_m
        bin_end = min(1440, bin_start + bin_m)
        overlap = max(0, min(e, bin_end) - max(s, bin_start))
        if overlap > 0:
            acc[b] += overlap


def presence_hist_by_purpose(segments, step_minutes: int, clock_bin_minutes: int = 10) -> Dict[str, np.ndarray]:
    """
    Returns total minutes present per clock bin for each purpose.
    The result is NOT normalized; caller can normalize to probabilities if desired.
    """
    bin_m = int(clock_bin_minutes)
    nbins = int(np.ceil(1440 / bin_m))
    purpose_to_hist: Dict[str, np.ndarray] = {}
    for person in segments:
        for seg in person:
            purpose = str(seg["purpose"])
            start_m = int(seg["startime"])
            dur_m = int(seg["total_duration"])
            if purpose not in purpose_to_hist:
                purpose_to_hist[purpose] = np.zeros(nbins, dtype=np.float64)
            _accumulate_presence_for_segment(purpose_to_hist[purpose], start_m, dur_m, bin_m)
    return purpose_to_hist


def duration_samples_by_purpose(segments) -> Dict[str, np.ndarray]:
    out: Dict[str, List[int]] = {}
    for person in segments:
        for seg in person:
            p = str(seg["purpose"])
            out.setdefault(p, []).append(int(seg["total_duration"]))
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def bigram_matrix(segments, purposes: List[str]) -> np.ndarray:
    P = len(purposes)
    idx = {p: i for i, p in enumerate(purposes)}
    M = np.zeros((P, P), dtype=np.float64)
    for person in segments:
        if len(person) <= 1:
            continue
        seq = [str(s["purpose"]) for s in person]
        for a, b in zip(seq[:-1], seq[1:]):
            if a in idx and b in idx:
                M[idx[a], idx[b]] += 1.0
    return M


def percent_start_end_home(segments) -> Tuple[float, float, float]:
    """Returns (start_home_pct, end_home_pct, all_home_day_pct)."""
    if not segments:
        return 0.0, 0.0, 0.0
    n = len(segments)
    start, end, allhome = 0, 0, 0
    for person in segments:
        if not person:
            continue
        if str(person[0]["purpose"]) == "Home":
            start += 1
        if str(person[-1]["purpose"]) == "Home":
            end += 1
        if all(str(seg["purpose"]) == "Home" for seg in person):
            allhome += 1
    return 100.0 * start / n, 100.0 * end / n, 100.0 * allhome / n


def work_start_window_coverage(segments, start_hour: int = 7, end_hour: int = 10) -> float:
    """
    Percent of people with at least one 'Work' segment starting within [start_hour, end_hour) local time.
    """
    if not segments:
        return 0.0
    start_m = int(start_hour * 60)
    end_m = int(end_hour * 60)
    count = 0
    for person in segments:
        found = False
        for seg in person:
            if str(seg["purpose"]) == "Work":
                t0 = int(seg["startime"]) % 1440
                if start_m <= t0 < end_m:
                    found = True
                    break
        if found:
            count += 1
    return 100.0 * count / len(segments)


# -----------------------------
# Distances
# -----------------------------
def _normalize_prob(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    s = x.sum()
    if s <= 0:
        return np.full_like(x, 1.0 / len(x))
    return np.maximum(x / s, eps)


def jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = _normalize_prob(p, eps)
    q = _normalize_prob(q, eps)
    m = 0.5 * (p + q)
    def kl(a, b):
        return float(np.sum(a * (np.log(a + eps) - np.log(b + eps))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def wasserstein1(x_real: np.ndarray, x_syn: np.ndarray) -> float:
    r = np.asarray(x_real, dtype=np.float64)
    s = np.asarray(x_syn, dtype=np.float64)
    if r.size == 0 and s.size == 0:
        return 0.0
    if r.size == 0:
        return float(np.mean(np.abs(s)))
    if s.size == 0:
        return float(np.mean(np.abs(r)))
    qs = np.linspace(0.0, 1.0, 101)
    rq = np.quantile(r, qs)
    sq = np.quantile(s, qs)
    return float(np.mean(np.abs(rq - sq)))


def bigram_l1(B_real: np.ndarray, B_syn: np.ndarray) -> float:
    Pr = _normalize_prob(B_real)
    Ps = _normalize_prob(B_syn)
    return float(np.sum(np.abs(Pr - Ps)))


# -----------------------------
# Top-level summarizer
# -----------------------------
def summarize(real_segments, syn_segments, purposes: List[str], step_minutes: int) -> Dict[str, Any]:
    clock_bin = 10  # fixed for comparability

    # Presence histograms
    H_real = presence_hist_by_purpose(real_segments, step_minutes, clock_bin_minutes=clock_bin)
    H_syn = presence_hist_by_purpose(syn_segments, step_minutes, clock_bin_minutes=clock_bin)

    # JSD per purpose (only when both have non-empty)
    by_purpose_jsd: Dict[str, float] = {}
    macro_vals: List[float] = []
    for p in purposes:
        hr = H_real.get(p, np.zeros(int(np.ceil(1440 / clock_bin)), dtype=np.float64))
        hs = H_syn.get(p, np.zeros_like(hr))
        d = jsd(hr, hs)
        by_purpose_jsd[p] = float(d)
        macro_vals.append(d)

    # Durations per purpose: Wasserstein-1 distance (in minutes)
    D_real = duration_samples_by_purpose(real_segments)
    D_syn = duration_samples_by_purpose(syn_segments)
    dur_w1: Dict[str, float] = {}
    for p in purposes:
        dur_w1[p] = float(wasserstein1(D_real.get(p, np.array([])), D_syn.get(p, np.array([]))))

    # Bigrams
    B_r = bigram_matrix(real_segments, purposes)
    B_s = bigram_matrix(syn_segments, purposes)
    bigram_dist = bigram_l1(B_r, B_s)

    # Endpoints
    start_home, end_home, all_home = percent_start_end_home(real_segments)
    start_home_s, end_home_s, all_home_s = percent_start_end_home(syn_segments)

    # Work window coverage
    work_cov = work_start_window_coverage(syn_segments, 7, 10)

    # Minutes by purpose
    def mins_by_purpose(segments):
        m: Dict[str, int] = {}
        for person in segments:
            for seg in person:
                p = str(seg["purpose"]) 
                m[p] = m.get(p, 0) + int(seg["total_duration"]) 
        return m

    mins_r = mins_by_purpose(real_segments)
    mins_s = mins_by_purpose(syn_segments)

    summary = {
        "dataset": {
            "n_people_real": int(len(real_segments)),
            "n_people_syn": int(len(syn_segments)),
            "step_minutes": int(step_minutes),
        },
        "endpoint": {
            "start_home_pct": float(start_home_s),
            "end_home_pct": float(end_home_s),
            "all_home_day_pct": float(all_home_s),
        },
        "time_of_day_jsd": {
            "macro_avg": float(np.mean(macro_vals) if len(macro_vals) else 0.0),
            "by_purpose": {k: float(v) for k, v in by_purpose_jsd.items()},
        },
        "duration_w1": {k: float(v) for k, v in dur_w1.items()},
        "bigram": {"l1_distance": float(bigram_dist)},
        "coverage": {"work_start_7_10_pct": float(work_cov)},
        "mins_by_purpose": {
            "real": {k: int(v) for k, v in mins_r.items()},
            "syn": {k: int(v) for k, v in mins_s.items()},
        },
    }
    return summary


