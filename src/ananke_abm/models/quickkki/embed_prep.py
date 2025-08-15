import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ---------- Helpers ----------

def zscore_cols(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Tuple[float,float]]]:
    stats = {}
    out = df.copy()
    for c in cols:
        mu = float(out[c].mean())
        sd = float(out[c].std(ddof=0)) or 1.0
        out[c] = (out[c] - mu) / sd
        stats[c] = (mu, sd)
    return out, stats

def periodic_time_enc(starts: np.ndarray, K: int = 2) -> np.ndarray:
    # starts shape: (T,), returns (T, 2K)
    s = starts.reshape(-1,1)
    outs = []
    for k in range(1, K+1):
        outs.append(np.sin(2*math.pi*k*s/24.0))
        outs.append(np.cos(2*math.pi*k*s/24.0))
    return np.concatenate(outs, axis=1)

def compute_deltas(starts: np.ndarray, durs: np.ndarray) -> np.ndarray:
    T = starts.shape[0]
    delta = np.zeros_like(starts)
    if T == 0:
        return delta
    delta[0] = starts[0]
    if T > 1:
        prev_end = starts[:-1] + durs[:-1]
        delta[1:] = starts[1:] - prev_end
    return np.maximum(delta, 0.0)

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Embedding prep & validation for mock weekday activity data.")
    ap.add_argument("--persons", required=True, help="Path to persons.csv")
    ap.add_argument("--schedules", required=True, help="Path to schedules.csv")
    ap.add_argument("--purposes", required=True, help="Path to purposes.csv")
    ap.add_argument("--out_dir", required=True, help="Directory to save artifacts and figures")
    ap.add_argument("--harmonics", type=int, default=2, help="#harmonics for periodic time encoding")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fig_dir = os.path.join(args.out_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)
    art_dir = os.path.join(args.out_dir, "artifacts")
    os.makedirs(art_dir, exist_ok=True)

    persons = pd.read_csv(args.persons)
    sched = pd.read_csv(args.schedules)
    purpose_tbl = pd.read_csv(args.purposes)

    # ---------- Vocab ----------
    # Keep the order from purposes.csv to ensure stability
    purposes = purpose_tbl["purpose"].tolist()
    p2i = {p:i for i,p in enumerate(purposes)}
    P = len(purposes)

    # ---------- Normalize purpose meta ----------
    NUMERIC = ["importance","flexibility","start_mu","start_std","dur_mu","dur_std","skip_prob"]
    CATEG = ["category"]
    norm_meta, meta_stats = zscore_cols(purpose_tbl.copy(), NUMERIC)
    cats = pd.Categorical(norm_meta["category"])
    norm_meta["category_idx"] = cats.codes
    cat_classes = list(cats.categories)
    n_cats = len(cat_classes)

    # ---------- Basic validations on schedules ----------
    report = {}
    required_cols = {"person_id","day","seq_id","purpose","start_time","duration"}
    assert required_cols.issubset(set(sched.columns)), "schedules.csv missing required columns"

    # 0) All purposes referenced exist in purposes.csv
    unknown = sorted(set(sched["purpose"].unique()) - set(purposes))
    report["unknown_purposes_in_schedules"] = unknown
    report["all_purposes_known"] = (len(unknown) == 0)

    # 1) Per (person, day): sorted by seq_id; start[0]==0, last end==24; non-overlapping
    group = sched.groupby(["person_id","day"])
    bad_sort = 0
    bad_start0 = 0
    bad_end24 = 0
    overlaps = 0
    negative_delta = 0

    # also build co-occurrence counts: purpose x purpose, co-appear in the same day (unordered pairs)
    cooc = np.zeros((P,P), dtype=np.float64)

    for (pid, day), df in group:
        df = df.sort_values("seq_id").reset_index(drop=True)
        # sort by seq_id must imply non-decreasing start_time after sanitize from generator
        if not (df["seq_id"].values == np.arange(len(df))).all():
            bad_sort += 1

        starts = df["start_time"].values.astype(float)
        durs   = df["duration"].values.astype(float)

        if len(starts) == 0 or abs(starts[0] - 0.0) > 1e-6:
            bad_start0 += 1

        ends = starts + durs
        if len(ends) == 0 or abs(ends[-1] - 24.0) > 1e-6:
            bad_end24 += 1

        # deltas >= 0
        deltas = compute_deltas(starts, durs)
        if np.any(deltas < -1e-8):
            negative_delta += 1

        # overlaps check: start_t >= end_{t-1}
        if len(starts) > 1:
            if np.any(starts[1:] < ends[:-1] - 1e-8):
                overlaps += 1

        # co-appearance
        day_purposes = sorted(set(df["purpose"].tolist()))
        idxs = [p2i[p] for p in day_purposes if p in p2i]
        for i in idxs:
            for j in idxs:
                cooc[i,j] += 1.0

    report["bad_sort_groups"] = int(bad_sort)
    report["bad_start0_groups"] = int(bad_start0)
    report["bad_end24_groups"] = int(bad_end24)
    report["overlap_groups"] = int(overlaps)
    report["negative_delta_groups"] = int(negative_delta)

    # ---------- Figures ----------
    # 1) Co-occurrence heatmap
    fig = plt.figure(figsize=(6,5))
    im = plt.imshow(cooc, interpolation='nearest', aspect='auto')
    plt.title("Purpose Co-occurrence (same day)")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(P), purposes, rotation=45, ha='right')
    plt.yticks(range(P), purposes)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "cooccurrence_heatmap.png"))
    plt.close(fig)

    # PMI (Pointwise Mutual Information), with smoothing
    total_days = len(group)
    row_sum = cooc.sum(axis=1, keepdims=True) + 1e-8
    col_sum = cooc.sum(axis=0, keepdims=True) + 1e-8
    p_ij = cooc / max(total_days, 1)
    p_i = row_sum / max(total_days, 1)
    p_j = col_sum / max(total_days, 1)
    pmi = np.log((p_ij + 1e-8) / (p_i @ p_j + 1e-12))
    # PCA to 2D for a light-weight embedding
    pca = PCA(n_components=2, random_state=0)
    emb2 = pca.fit_transform(pmi)

    fig = plt.figure(figsize=(6,5))
    plt.scatter(emb2[:,0], emb2[:,1])
    for i,p in enumerate(purposes):
        plt.text(emb2[i,0], emb2[i,1], p, fontsize=9)
    plt.title("Purpose PMI-PCA (2D)")
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "purpose_pmi_pca.png"))
    plt.close(fig)

    # 2) Start/duration distributions per purpose (simple scatter)
    # scatter of start vs duration, colored by purpose index
    fig = plt.figure(figsize=(7,5))
    colors = np.linspace(0,1,P)
    for i,p in enumerate(purposes):
        sub = sched[sched["purpose"] == p]
        if len(sub) == 0: 
            continue
        plt.scatter(sub["start_time"].values, sub["duration"].values, s=6, alpha=0.5, label=p)
    plt.xlabel("Start time (h)")
    plt.ylabel("Duration (h)")
    plt.title("Start vs Duration by Purpose")
    plt.legend(markerscale=3, fontsize=7, ncol=min(P,3))
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "start_vs_duration_scatter.png"))
    plt.close(fig)

    # ---------- Save artifacts ----------
    # Build meta feature matrix aligned with vocab order
    meta_numeric = norm_meta.loc[:, NUMERIC].values.astype(np.float32)
    # category one-hot
    cat_idx = norm_meta["category_idx"].values.astype(int)
    cat_oh = np.eye(len(cat_classes), dtype=np.float32)[cat_idx]
    meta_mat = np.concatenate([meta_numeric, cat_oh], axis=1)

    artifacts = {
        "purposes": purposes,
        "p2i": p2i,
        "numeric_cols": NUMERIC,
        "category_classes": cat_classes,
        "meta_stats": meta_stats,  # means/stds for inverse-transform later if needed
        "meta_matrix": meta_mat,   # (P, Dmeta_zscored + n_cats)
        "pmi_embedding_2d": emb2,  # (P,2) for visualization sanity checks
    }
    np.savez(os.path.join(art_dir, "embedding_artifacts.npz"), **artifacts)

    # ---------- Validation report ----------
    report["n_days"] = int(len(group))
    report["n_persons"] = int(persons.shape[0])
    report["n_purposes"] = int(P)
    with open(os.path.join(args.out_dir, "validation_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("Validation summary:")
    print(json.dumps(report, indent=2))
    print(f"Artifacts -> {art_dir}")
    print(f"Figures   -> {fig_dir}")

if __name__ == "__main__":
    main()