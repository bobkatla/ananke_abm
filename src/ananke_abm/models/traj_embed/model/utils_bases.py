import torch
import math
import numpy as np
import pandas as pd

def fourier_time_features(t, K:int):
    """
    Compute Fourier time features for continuous t in [0,1].
    Args:
        t: (...,) tensor of times in [0,1]
        K: number of cosine/sine pairs (positive integer)
    Returns:
        feat: (..., 2K+1) with [1, cos(2πt), sin(2πt), cos(4πt), sin(4πt), ...]
    """
    twopi = 2.0 * math.pi
    t = t.unsqueeze(-1)  # (...,1)
    if K <= 0:
        return torch.ones_like(t)
    orders = torch.arange(1, K+1, device=t.device, dtype=t.dtype)
    base = torch.ones_like(t)  # (...,1)
    angles = twopi * t * orders  # (...,K)
    cos_part = torch.cos(angles)
    sin_part = torch.sin(angles)
    feat = torch.cat([base, cos_part, sin_part], dim=-1)  # (..., 1+K+K)
    return feat  # (..., 2K+1)

def gauss_legendre_nodes(Q:int, dtype=torch.float32, device="cpu"):
    """
    Return Gauss-Legendre nodes and weights on [0,1].
    """
    xs, ws = np.polynomial.legendre.leggauss(Q)  # nodes on [-1,1]
    # map to [0,1]
    t = (xs + 1.0) / 2.0
    w = ws / 2.0
    t = torch.tensor(t, dtype=dtype, device=device)  # (Q,)
    w = torch.tensor(w, dtype=dtype, device=device)  # (Q,)
    return t, w

def build_time_binned_transition_costs(activities_csv, purposes, nbins=24, eps=1.0, device="cpu"):
    df = pd.read_csv(activities_csv).sort_values(["persid","startime","stopno"])
    mp = {p:i for i,p in enumerate(purposes)}
    P = len(purposes); Tm = int(df["total_duration"].groupby(df["persid"]).sum().mode().iloc[0])

    cnt = np.full((nbins, P, P), eps, dtype=np.float64)  # Laplace smoothing
    for _, g in df.groupby("persid"):
        g = g.sort_values(["startime","stopno"])
        ids = [mp[p] for p in g["purpose"].tolist()]
        starts = g["startime"].to_numpy(dtype=np.float64)
        for i in range(len(ids)-1):
            a, b = ids[i], ids[i+1]
            tnext = starts[i+1] / max(1.0, Tm)      # normalized [0,1)
            bin_idx = min(int(tnext * nbins), nbins-1)
            cnt[bin_idx, a, b] += 1.0

    prob = cnt / cnt.sum(axis=2, keepdims=True)     # normalize over b
    cost = -np.log(prob + 1e-12)                    # (nbins,P,P)
    return torch.tensor(cost, dtype=torch.float32, device=device)

def viterbi_timecost_decode(u, t_nodes, C_t, switch_cost=0.0):
    """
    u: (B,P,L) utilities (higher=better)
    t_nodes: (L,)
    C_t: (Tbin,P,P) time-binned transition costs (lower=better)
    switch_cost: optional constant added to all transitions (fewer switches)
    returns: list of segments [(p, t0_norm, dur_norm)] per batch item
    """
    B, P, L = u.shape
    nbins = C_t.shape[0]
    t_mid = 0.5 * (t_nodes[:-1] + t_nodes[1:])                  # (L-1,)
    bins  = torch.clamp((t_mid * nbins).long(), 0, nbins-1)     # (L-1,)

    unary = -u                                                   # cost = -utility
    paths = []
    for b in range(B):
        dp = unary[b, :, 0].clone()                              # (P,)
        bp = torch.full((P, L), -1, dtype=torch.long, device=u.device)
        for t in range(1, L):
            A = C_t[bins[t-1]] + switch_cost                     # (P,P)
            prev = dp.view(P,1) + A                              # (P,P)
            best_prev_cost, best_prev = prev.min(dim=0)          # (P,), (P,)
            dp = best_prev_cost + unary[b, :, t]
            bp[:, t] = best_prev
        last = int(dp.argmin().item())
        labels = [last]
        for t in range(L-1, 0, -1):
            last = int(bp[last, t].item())
            labels.append(last)
        labels = labels[::-1]

        # merge labels -> segments (on [0,1])
        segs = []
        t0 = 0
        for t in range(1, L+1):
            if t == L or labels[t] != labels[t-1]:
                p = labels[t-1]
                dur = t - t0
                segs.append((p, t0 / L, dur / L))
                t0 = t
        paths.append(segs)
    return paths

