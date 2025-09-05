import math
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch


# ---------------------------
# Fourier features (kept)
# ---------------------------
def fourier_time_features(t: torch.Tensor, K: int) -> torch.Tensor:
    """
    Compute Fourier features for t ∈ [0,1].
    Returns (..., 2K+1) with [1, cos(2πt), sin(2πt), cos(4πt), sin(4πt), ...].
    """
    twopi = 2.0 * math.pi
    t = t.unsqueeze(-1)  # (...,1)
    if K <= 0:
        return torch.ones_like(t)
    orders = torch.arange(1, K + 1, device=t.device, dtype=t.dtype)  # (K,)
    base = torch.ones_like(t)                                        # (...,1)
    angles = twopi * t * orders                                      # (...,K)
    cos_part = torch.cos(angles)
    sin_part = torch.sin(angles)
    feat = torch.cat([base, cos_part, sin_part], dim=-1)             # (..., 1+K+K)
    return feat


# ------------------------------------------------------
# Allocation grid helpers (minutes + normalized [0,1])
# ------------------------------------------------------
def make_alloc_grid(
    T_alloc_minutes: int,
    step_minutes: int,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a uniform allocation grid.

    Args:
        T_alloc_minutes: horizon in minutes (e.g., 1800 for 30h).
        step_minutes: grid step (e.g., 5 for train, 1 for eval).
        device/dtype: torch placement.

    Returns:
        t_alloc_minutes: [L] int/float minute indices {0, step, 2*step, ...}
        t_alloc01:       [L] normalized positions in [0,1] over allocation axis.
    """
    if device is None:
        device = "cpu"
    L = int(math.ceil(T_alloc_minutes / step_minutes))
    # Use exact minute marks; ensure last point is within horizon
    t_alloc_minutes = torch.arange(0, L * step_minutes, step_minutes, device=device, dtype=dtype)  # [L]
    t_alloc01 = t_alloc_minutes / float(T_alloc_minutes)
    t_alloc01 = torch.clamp(t_alloc01, 0.0, 1.0)
    return t_alloc_minutes, t_alloc01


# -------------------------------------------------------------------
# 24h clock–binned transition costs (optional, for structured decode)
# -------------------------------------------------------------------
def build_clock_binned_transition_costs(
    activities_csv: str,
    purposes: list[str],
    nbins: int = 24,
    eps: float = 1.0,
    T_clock_minutes: int = 1440,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Estimate time-of-day (24h)–binned transition costs A[p->q | clock_bin].

    We count transitions a→b and bucket them by the *clock time* (in 24h) at
    which the *next* activity starts. Laplace smoothing on counts, then
    negative log to make costs.

    Args:
        activities_csv: path to activities CSV with columns:
            ['persid','hhid','stopno','purpose','startime','total_duration']
            startime in minutes (absolute).
        purposes: list of purpose names in the model's order.
        nbins: number of 24h clock bins (24 = hourly; 288 = 5-min).
        eps: Laplace smoothing pseudo-count.
        T_clock_minutes: 1440 by default.
        device: torch device.

    Returns:
        cost: [nbins, P, P] float32 tensor with per-bin transition costs.
              Lower is better. Can be added to pairwise terms.
    """
    df = pd.read_csv(activities_csv).sort_values(["persid", "startime", "stopno"])
    mp = {p: i for i, p in enumerate(purposes)}
    P = len(purposes)

    cnt = np.full((nbins, P, P), eps, dtype=np.float64)  # Laplace smoothing

    for _, g in df.groupby("persid"):
        g = g.sort_values(["startime", "stopno"])
        ids = [mp.get(p, None) for p in g["purpose"].tolist()]
        starts_min = g["startime"].to_numpy(dtype=np.float64)
        # Filter unknown purposes
        valid_rows = [i for i, pid in enumerate(ids) if pid is not None]
        if len(valid_rows) < 2:
            continue
        ids = [ids[i] for i in valid_rows]
        starts_min = starts_min[valid_rows]

        for i in range(len(ids) - 1):
            a, b = ids[i], ids[i + 1]
            # Bin by clock-of-day of the *next* start time
            tnext_clock = (starts_min[i + 1] % float(T_clock_minutes)) / float(T_clock_minutes)  # in [0,1)
            bin_idx = min(int(tnext_clock * nbins), nbins - 1)
            cnt[bin_idx, a, b] += 1.0

    prob = cnt / cnt.sum(axis=2, keepdims=True)   # normalize over destination b
    cost = -np.log(prob + 1e-12)                  # [nbins, P, P]
    return torch.tensor(cost, dtype=torch.float32, device=device)


# -------------------------------------------------------
# Legacy Viterbi with time-binned costs (optional/legacy)
# -------------------------------------------------------
def viterbi_timecost_decode(u, t_nodes, C_t, switch_cost: float = 0.0):
    """
    LEGACY helper (pre-CRF). Dynamic program that adds time-binned transition
    costs C_t and a constant switch_cost to -utilities and finds a best path.

    Args:
        u: (B,P,L) utilities (higher=better)
        t_nodes: (L,) in [0,1] (allocation-normalized)
        C_t: (Tbin,P,P) time-binned transition costs (lower=better)
        switch_cost: optional constant added to all transitions

    Returns:
        list of segments [(p, t0_norm, dur_norm)] per batch item
    """
    B, P, L = u.shape
    nbins = C_t.shape[0]
    t_mid = 0.5 * (t_nodes[:-1] + t_nodes[1:])              # (L-1,)
    bins = torch.clamp((t_mid * nbins).long(), 0, nbins - 1)  # (L-1,)

    unary = -u                                               # cost = -utility
    paths = []
    for b in range(B):
        dp = unary[b, :, 0].clone()                          # (P,)
        bp = torch.full((P, L), -1, dtype=torch.long, device=u.device)
        for t in range(1, L):
            A = C_t[bins[t - 1]] + switch_cost               # (P,P)
            prev = dp.view(P, 1) + A                         # (P,P)
            best_prev_cost, best_prev = prev.min(dim=0)      # (P,), (P,)
            dp = best_prev_cost + unary[b, :, t]
            bp[:, t] = best_prev
        last = int(dp.argmin().item())
        labels = [last]
        for t in range(L - 1, 0, -1):
            last = int(bp[last, t].item())
            labels.append(last)
        labels = labels[::-1]

        # merge labels -> segments (on [0,1])
        segs = []
        t0 = 0
        for t in range(1, L + 1):
            if t == L or labels[t] != labels[t - 1]:
                p = labels[t - 1]
                dur = t - t0
                segs.append((p, t0 / L, dur / L))
                t0 = t
        paths.append(segs)
    return paths
