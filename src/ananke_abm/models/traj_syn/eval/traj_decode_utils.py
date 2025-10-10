import torch
import pandas as pd
import numpy as np
from typing import List, Tuple


def labels_to_segments(y_hat: torch.Tensor, t_alloc01: torch.Tensor) -> List[List[Tuple[int, float, float]]]:
    """
    y_hat: [B, L] long labels
    t_alloc01: [L] normalized grid positions in [0,1]
    Returns: list over batch; each item is list[(p_idx, t0_norm, d_norm)]
    """
    B, L = y_hat.shape
    t = t_alloc01.detach().cpu().numpy()
    out: List[List[Tuple[int, float, float]]] = []
    for b in range(B):
        yb = y_hat[b].detach().cpu().numpy()
        if L == 0:
            out.append([])
            continue
        segs = []
        start = 0
        for i in range(1, L):
            if yb[i] != yb[i - 1]:
                p = int(yb[i - 1])
                t0 = float(t[start])
                t1 = float(t[i])
                segs.append((p, t0, max(t1 - t0, 0.0)))
                start = i
        p = int(yb[-1])
        t0 = float(t[start])
        t1 = float(t[-1])
        segs.append((p, t0, max(t1 - t0, 0.0)))
        out.append(segs)
    return out


def decoded_to_activities_df(decoded, purposes, T_minutes: int,
                             start_persid: int = 0, prefix: str = "gen") -> pd.DataFrame:
    """
    Turn a list[List[(p_idx, t0, d)]] into a long DataFrame with columns:
    persid, stopno, purpose, startime, total_duration.
    Durations sum to T_minutes per persid; start times are cumulative.
    """
    rows = []
    for s_idx, segs in enumerate(decoded):
        persid = f"{prefix}_{start_persid + s_idx:06d}"
        if not segs:
            rows.append({"persid": persid, "stopno": 1, "purpose": "Home",
                         "startime": 0, "total_duration": int(T_minutes)})
            continue

        d = np.array([max(0.0, float(dur)) for (_, _, dur) in segs], dtype=np.float64)
        if d.sum() <= 0:
            d = np.ones_like(d) / len(d)
        d = d / d.sum()

        dur_m = np.rint(d * T_minutes).astype(int)
        delta = int(T_minutes - dur_m.sum())
        dur_m[-1] += delta
        start_m = np.concatenate([[0], np.cumsum(dur_m[:-1])])

        for stopno, ((p_idx, _t0, _d), st, du) in enumerate(zip(segs, start_m, dur_m), start=1):
            if du <= 0:
                continue
            rows.append({
                "persid": persid,
                "stopno": stopno,
                "purpose": purposes[p_idx],
                "startime": int(st),
                "total_duration": int(du),
            })

    return pd.DataFrame(rows, columns=["persid", "stopno", "purpose", "startime", "total_duration"])


def build_truth_sets(activities_csv: str):
    acts = pd.read_csv(activities_csv)
    acts = acts.sort_values(["persid", "startime", "stopno"])
    seqs = acts.groupby("persid")["purpose"].apply(list)

    full_seqs = set(tuple(s) for s in seqs.values)

    bigrams = set()
    for s in seqs.values:
        if len(s) >= 2:
            bigrams.update(zip(s[:-1], s[1:]))
    return full_seqs, bigrams


def validate_sequences(gen_df: pd.DataFrame,
                       full_seqs: set,
                       bigrams: set,
                       home_label: str = "Home") -> pd.DataFrame:
    out = []
    for pid, g in gen_df.sort_values(["persid", "stopno"]).groupby("persid"):
        seq = g["purpose"].tolist()
        start_home = (len(seq) > 0 and seq[0] == home_label)
        end_home   = (len(seq) > 0 and seq[-1] == home_label)
        seq_exists = tuple(seq) in full_seqs
        if len(seq) <= 1:
            all_pairs_ok = True
        else:
            pairs = list(zip(seq[:-1], seq[1:]))
            all_pairs_ok = all(pair in bigrams for pair in pairs)

        if not (start_home and end_home):
            confidence = "NO"
        elif seq_exists:
            confidence = "OK"
        elif all_pairs_ok:
            confidence = "MODERATE"
        else:
            confidence = "MAYBE"

        out.append({
            "persid": pid,
            "n_stops": len(seq),
            "start_is_home": start_home,
            "end_is_home": end_home,
            "sequence_exists_in_seed": seq_exists,
            "all_adjacent_pairs_in_seed": all_pairs_ok,
            "confidence": confidence,
            "sequence": " > ".join(seq),
        })

    cols = ["persid","n_stops","start_is_home","end_is_home",
            "sequence_exists_in_seed","all_adjacent_pairs_in_seed","confidence","sequence"]
    return pd.DataFrame(out, columns=cols)