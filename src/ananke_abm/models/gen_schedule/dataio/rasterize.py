
from __future__ import annotations
import pandas as pd
import numpy as np
import json
import os

PURPOSE_COL = "purpose"

def build_purpose_map(df: pd.DataFrame):
    uniq = sorted(df[PURPOSE_COL].unique().tolist())
    return {p:i for i,p in enumerate(uniq)}

def rasterize_person(person_df, purpose_map, grid_min: int, horizon_min: int=1800):
    L = horizon_min // grid_min
    arr = np.zeros(L, dtype=np.int64)
    for _,r in person_df.iterrows():
        s = int(r["starttime"])
        d = int(r["total_duration"])
        p = purpose_map[r["purpose"]]
        a = max(0, s)//grid_min
        b = min(L, (s+d + grid_min - 1)//grid_min)
        arr[a:b] = p
    return arr

def compute_empirical_tod(Y, P):
    N,L = Y.shape
    m = np.zeros((L,P), dtype=np.float64)
    for t in range(L):
        col = Y[:,t]
        for p in range(P):
            m[t,p] = np.mean(col==p)
    return m

def prepare_from_csv(csv_path: str, out_path: str, grid_min: int=10, horizon_min: int=1800):
    df = pd.read_csv(csv_path)
    if "startime" in df.columns and "starttime" not in df.columns:
        df = df.rename(columns={"startime":"starttime"})
    purpose_map = build_purpose_map(df)
    inv_map = {v:k for k,v in purpose_map.items()}
    L = horizon_min // grid_min

    seqs, pers = [], []
    for pid, grp in df.groupby("persid"):
        grp = grp.sort_values("stopno")
        y = rasterize_person(grp, purpose_map, grid_min, horizon_min)
        seqs.append(y); pers.append(pid)
    Y = np.stack(seqs, axis=0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, Y=Y.astype(np.int64))
    meta = {"grid_min": grid_min, "horizon_min": horizon_min, "L": int(L),
            "purpose_map": purpose_map, "inv_purpose_map": inv_map, "N": int(Y.shape[0])}
    with open(out_path.replace(".npz", "_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    m = compute_empirical_tod(Y, P=len(purpose_map))
    np.save(out_path.replace(".npz", "_tod.npy"), m)
    return out_path, meta
