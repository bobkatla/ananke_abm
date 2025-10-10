import math
from typing import Tuple, Optional, List

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


def merge_primary_slivers(y_grid: torch.Tensor,
                          is_primary: torch.Tensor,   # [P] bool
                          tau_bins: int) -> torch.Tensor:
    """
    In-place merges any run of length < tau_bins that is non-primary and
    is flanked by the same primary label on both sides: A ... B ... A  ->  A ... A ... A
    y_grid: [B,L] long, is_primary: [P] bool, tau_bins: sliver threshold in bins.
    """
    B, L = y_grid.shape
    y = y_grid.clone()
    for b in range(B):
        seq = y[b]
        # run-length scan
        start = 0
        while start < L:
            end = start
            lab = int(seq[start].item())
            while end < L and int(seq[end].item()) == lab:
                end += 1
            run_len = end - start
            # Check pattern A (primary) ... B (non-primary short) ... A (same primary)
            # Look at the *next* run if exists
            if end < L and not is_primary[lab]:
                # previous run: [prev_start, start), current run: [start,end)
                # need prev and next runs to be same primary
                # Find previous run bounds
                prev_end = start
                prev_start = prev_end - 1
                if prev_start >= 0:
                    prev_lab = int(seq[prev_start].item())
                    while prev_start >= 0 and int(seq[prev_start].item()) == prev_lab:
                        prev_start -= 1
                    prev_start += 1
                    # Next run bounds
                    next_start = end
                    if next_start < L:
                        next_lab = int(seq[next_start].item())
                        next_end = next_start
                        while next_end < L and int(seq[next_end].item()) == next_lab:
                            next_end += 1
                        # Conditions: prev and next labels identical, primary; current run short
                        if (run_len < tau_bins and
                            prev_lab == next_lab and
                            is_primary[prev_lab]):
                            seq[start:end] = prev_lab  # merge sliver into primary
                            # continue from end of merged region to avoid infinite loop
                            start = end
                            continue
            start = end
    return y


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


def segments_from_padded_to_grid(
    p_pad: torch.Tensor,    # [B,Lmax] long
    t_pad: torch.Tensor,    # [B,Lmax] float in [0,1]
    d_pad: torch.Tensor,    # [B,Lmax] float in [0,1]
    lengths: List[int],
    L: int,                 # number of grid bins
) -> List[List[Tuple[int,int,int]]]:
    """
    Returns per-batch list of segments as (p, s, d) in grid bins, merged for identical adjacents.
    """
    B, Lmax = p_pad.shape
    out: List[List[Tuple[int,int,int]]] = []
    for b in range(B):
        Lb = lengths[b]
        segs: List[Tuple[int,int,int]] = []
        for i in range(Lb):
            p = int(p_pad[b,i].item())
            s = max(0, min(L-1, int(round(t_pad[b,i].item() * L))))
            d = max(1, int(round(d_pad[b,i].item() * L)))
            t_end = min(L, s + d)
            d = t_end - s
            if d <= 0: continue
            if segs and segs[-1][0] == p and segs[-1][1] + segs[-1][2] == s:
                # merge
                prev_p, prev_s, prev_d = segs[-1]
                segs[-1] = (prev_p, prev_s, prev_d + d)
            else:
                segs.append((p, s, d))
        out.append(segs)
    return out
