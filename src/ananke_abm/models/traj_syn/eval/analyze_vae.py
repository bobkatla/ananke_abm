# ananke_abm/models/traj_syn/eval/analyze_vae.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import click
import numpy as np
import pandas as pd


# ---------------------------
# Helpers
# ---------------------------
def safe_read_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    return pd.read_csv(p)

def softmax_stable(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    np.exp(x, out=(x := np.exp(x)))
    x_sum = np.sum(x, axis=axis, keepdims=True) + 1e-12
    return x / x_sum

def entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=axis)

def kl_div(p: np.ndarray, q: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return np.sum(p * (np.log(p) - np.log(q)), axis=axis)

def js_div(p: np.ndarray, q: np.ndarray, axis: int = -1) -> np.ndarray:
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m, axis=axis) + 0.5 * kl_div(q, m, axis=axis)

def minutes_by_purpose(df: pd.DataFrame) -> Dict[str, int]:
    grp = df.groupby("purpose")["total_duration"].sum()
    return {k: int(v) for k, v in grp.items()}

def endpoint_stats(gen_df: pd.DataFrame, home_label: str = "Home") -> Dict[str, float]:
    rows = []
    for pid, g in gen_df.sort_values(["persid", "startime", "stopno"]).groupby("persid"):
        seq = g["purpose"].tolist()
        start_home = len(seq) > 0 and seq[0] == home_label
        end_home = len(seq) > 0 and seq[-1] == home_label
        all_home = all(p == home_label for p in seq) if seq else True
        rows.append((start_home, end_home, all_home))
    if not rows:
        return {"start_home_pct": 0.0, "end_home_pct": 0.0, "all_home_day_pct": 0.0}
    a = np.asarray(rows, dtype=np.bool_)
    n = a.shape[0]
    return {
        "start_home_pct": 100.0 * a[:, 0].mean(),
        "end_home_pct": 100.0 * a[:, 1].mean(),
        "all_home_day_pct": 100.0 * a[:, 2].mean(),
    }

def sequences_and_bigrams(df: pd.DataFrame) -> Tuple[List[Tuple[str, ...]], Dict[Tuple[str, str], int]]:
    seqs = []
    bigram_counts: Dict[Tuple[str, str], int] = {}
    for _, g in df.sort_values(["persid", "startime", "stopno"]).groupby("persid"):
        s = tuple(g["purpose"].tolist())
        seqs.append(s)
        for a, b in zip(s[:-1], s[1:]):
            bigram_counts[(a, b)] = bigram_counts.get((a, b), 0) + 1
    return seqs, bigram_counts

def l1_bigram_distance(b1: Dict[Tuple[str, str], int], b2: Dict[Tuple[str, str], int]) -> float:
    keys = set(b1.keys()) | set(b2.keys())
    n1 = sum(b1.values()) or 1
    n2 = sum(b2.values()) or 1
    d = 0.0
    for k in keys:
        p = b1.get(k, 0) / n1
        q = b2.get(k, 0) / n2
        d += abs(p - q)
    return d

def build_time_of_day_hist(
    df: pd.DataFrame,
    purposes: List[str],
    T_alloc_minutes: int,
    step_minutes: int,
) -> np.ndarray:
    """Return [P, L] histogram of minutes per bin normalized to a distribution per purpose."""
    L = int(np.ceil(T_alloc_minutes / step_minutes))
    P = len(purposes)
    purp_to_idx = {p: i for i, p in enumerate(purposes)}
    hist = np.zeros((P, L), dtype=np.float64)

    for _, g in df.groupby("persid"):
        # explode each stop into bins
        for _, r in g.iterrows():
            p = purp_to_idx.get(str(r["purpose"]))
            if p is None:
                continue
            start = int(r["startime"])
            dur = int(r["total_duration"])
            if dur <= 0:
                continue
            end = start + dur
            # bin indices
            b0 = start // step_minutes
            b1 = min(L * step_minutes, end) // step_minutes
            if b1 <= b0:
                b1 = min(b0 + 1, L)
            hist[p, b0:b1] += 1.0  # simple count proxy per bin

    # Normalize to distributions per purpose over the day
    with np.errstate(invalid="ignore", divide="ignore"):
        row_sums = hist.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        dist = hist / row_sums
    return dist  # [P, L]


# ---------------------------
# CLI
# ---------------------------
@click.command()
@click.option("--gen_csv", type=click.Path(exists=True), required=True,
              help="VAE-only generated CSV (persid, stopno, purpose, startime, total_duration).")
@click.option("--mean_probs_npy", type=click.Path(exists=True), required=True,
              help="[P,L] numpy array of mean per-purpose probabilities on eval grid.")
@click.option("--purposes", type=str, required=True,
              help="Comma-separated purpose list in the exact P-order used by the model.")
@click.option("--step_minutes", type=int, required=True,
              help="Eval grid minutes used to produce mean_probs and to bin gen_csv.")
@click.option("--t_alloc_minutes", type=int, required=True,
              help="Allocation horizon minutes (e.g., 1800 for 30h).")
@click.option("--history_csv", type=click.Path(exists=False), default=None,
              help="Optional VAE training history CSV to summarize.")
@click.option("--real_csv", type=click.Path(exists=False), default=None,
              help="Optional real activities CSV for distributional comparisons.")
@click.option("--out_json", type=click.Path(), required=True,
              help="Where to write the summary JSON.")
def main(
    gen_csv: str,
    mean_probs_npy: str,
    purposes: str,
    step_minutes: int,
    t_alloc_minutes: int,
    history_csv: Optional[str],
    real_csv: Optional[str],
    out_json: str,
):
    # ---------- Load ----------
    purposes_list = [p.strip() for p in purposes.split(",") if p.strip()]
    P = len(purposes_list)

    gen_df = safe_read_csv(gen_csv)
    if gen_df is None:
        raise ValueError("gen_csv is required.")

    mean_probs = np.load(mean_probs_npy)  # [P, L]
    if mean_probs.ndim != 2 or mean_probs.shape[0] != P:
        raise ValueError(f"mean_probs shape {mean_probs.shape} incompatible with P={P}.")

    L_eval = mean_probs.shape[1]

    hist_df = safe_read_csv(history_csv) if history_csv else None
    real_df = safe_read_csv(real_csv) if real_csv else None

    # ---------- Training summary (optional) ----------
    history_summary = None
    if hist_df is not None and not hist_df.empty:
        cols = [c for c in hist_df.columns if c in ("epoch", "train_total", "val_total", "nll", "kl")]
        hmini = hist_df[cols].copy()
        last = hmini.iloc[-1].to_dict()
        best_val = float(hmini["val_total"].min())
        history_summary = {
            "last": {k: float(last[k]) for k in hmini.columns if k != "epoch"},
            "best_val_total": best_val,
            "epochs": int(hmini["epoch"].iloc[-1]) if "epoch" in hmini.columns else len(hmini),
        }

    # ---------- Entropy over day (from mean probs) ----------
    ent_per_purpose = entropy(mean_probs, axis=1).tolist()  # entropy across L per purpose
    row_sums = mean_probs.sum(axis=1, keepdims=True) + 1e-12
    mean_probs_norm = mean_probs / row_sums
    ent_per_purpose = entropy(mean_probs_norm, axis=1).tolist()
    max_ent = float(np.log(mean_probs_norm.shape[1]))
    peakiness = [(1.0 - (e / max_ent)) for e in ent_per_purpose]
    
    # ---------- Minutes & shares ----------
    gen_minutes = minutes_by_purpose(gen_df)
    total_gen_minutes = sum(gen_minutes.values()) or 1
    gen_share = {k: (v / total_gen_minutes) for k, v in gen_minutes.items()}

    real_minutes = minutes_by_purpose(real_df) if real_df is not None else None
    real_share = None
    if real_minutes:
        total_real_minutes = sum(real_minutes.values()) or 1
        real_share = {k: (v / total_real_minutes) for k, v in real_minutes.items()}

    # ---------- Endpoint stats ----------
    endpoint = endpoint_stats(gen_df, home_label="Home")

    # ---------- Sequence diversity & bigram L1 (optional vs real) ----------
    seq_diversity = {}
    gen_seqs, gen_bigrams = sequences_and_bigrams(gen_df)
    uniq = len({s for s in gen_seqs})
    seq_diversity["unique_seq_ratio"] = float(uniq / max(len(gen_seqs), 1))
    bigram_l1 = None
    if real_df is not None:
        real_seqs, real_bigrams = sequences_and_bigrams(real_df)
        bigram_l1 = l1_bigram_distance(real_bigrams, gen_bigrams)

    # ---------- Time-of-day JSD per purpose (optional vs real) ----------
    tod_jsd = None
    if real_df is not None:
        real_dist = build_time_of_day_hist(real_df, purposes_list, t_alloc_minutes, step_minutes)  # [P,L]
        gen_dist = build_time_of_day_hist(gen_df, purposes_list, t_alloc_minutes, step_minutes)    # [P,L]
        if real_dist.shape[1] != gen_dist.shape[1]:
            raise ValueError(f"L mismatch for time-of-day hist: real {real_dist.shape[1]} vs gen {gen_dist.shape[1]}")
        by_purpose = {}
        vals = []
        for i, p in enumerate(purposes_list):
            v = float(js_div(real_dist[i], gen_dist[i], axis=0))
            by_purpose[p] = v
            vals.append(v)
        tod_jsd = {
            "macro_avg": float(np.mean(vals) if vals else 0.0),
            "by_purpose": by_purpose,
        }

    # ---------- Package & save ----------
    out = {
        "dataset": {
            "n_people_syn": int(gen_df["persid"].nunique()),
            "n_rows_syn": int(len(gen_df)),
            "P": P,
            "L_eval": L_eval,
            "step_minutes": int(step_minutes),
            "t_alloc_minutes": int(t_alloc_minutes),
        },
        "training": history_summary,
        "entropy": {
            "per_purpose": {purposes_list[i]: float(ent_per_purpose[i]) for i in range(P)},
            "peakiness": {purposes_list[i]: float(peakiness[i]) for i in range(P)},
        },
        "endpoint": endpoint,
        "minutes_by_purpose": {
            "syn": gen_minutes,
            "syn_share": {k: float(v) for k, v in gen_share.items()},
            "real": real_minutes if real_minutes is not None else None,
            "real_share": {k: float(v) for k, v in real_share.items()} if real_share is not None else None,
        },
        "sequence_diversity": seq_diversity,
        "bigram": {"l1_distance": float(bigram_l1)} if bigram_l1 is not None else None,
        "time_of_day_jsd": tod_jsd,
    }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    click.echo(f"Wrote VAE diagnostics to: {out_json}")

if __name__ == "__main__":
    main()
