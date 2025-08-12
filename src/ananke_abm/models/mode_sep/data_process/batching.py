"""
Batching utilities: build a union dense grid across people in the batch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

from ananke_abm.models.mode_sep.config import ModeSepConfig
from ananke_abm.models.mode_sep.data_process.data import PersonData


@dataclass
class UnionBatch:
    times_union: torch.Tensor                 # (T,) float32 on device
    is_gt_union: torch.Tensor                 # (B, T) bool
    snap_indices: torch.Tensor                # (B, T) long (index into person loc_ids) or -1
    stay_mask: torch.Tensor                   # (B, T) bool
    gt_interior_mask: torch.Tensor     # (B, T) bool — GT snaps excluding first/last
    stay_non_gt_mask: torch.Tensor     # (B, T) bool — inside stays but not at snaps
    stay_loc_ids: torch.Tensor
    min_dt: float


def _insert_internal_points(sorted_times: torch.Tensor, K: int) -> torch.Tensor:
    # sorted_times: (N,) strictly increasing
    if sorted_times.numel() <= 1:
        return sorted_times
    gaps = sorted_times[1:] - sorted_times[:-1]
    pieces: List[torch.Tensor] = []
    for i in range(sorted_times.numel() - 1):
        t0 = sorted_times[i]
        t1 = sorted_times[i + 1]
        pieces.append(t0.unsqueeze(0))
        if K > 0:
            # Internal points strictly inside (t0, t1)
            internal = torch.linspace(float(t0), float(t1), steps=K + 2, dtype=sorted_times.dtype, device=sorted_times.device)[1:-1]
            if internal.numel() > 0:
                pieces.append(internal)
    pieces.append(sorted_times[-1:].clone())
    return torch.unique(torch.cat(pieces), sorted=True)


def _times_to_union_mapping(times_union: torch.Tensor, times_snap: torch.Tensor, tol: float) -> Tuple[torch.Tensor, torch.Tensor]:
    # For each t in union, find if it matches a snap time within tol, and the snap index or -1
    # Vectorized via broadcasting
    if times_snap.numel() == 0:
        B = times_union.shape[0]
        return torch.zeros(B, dtype=torch.bool, device=times_union.device), torch.full((B,), -1, dtype=torch.long, device=times_union.device)
    du = times_union.unsqueeze(1)            # (T,1)
    ds = times_snap.unsqueeze(0)             # (1,S)
    eq = torch.isclose(du, ds, atol=tol, rtol=0.0)  # (T,S)
    is_gt = eq.any(dim=1)
    # For indices, choose the first matching snap index if any, else -1
    snap_idx = torch.where(is_gt, eq.float().argmax(dim=1).long(), torch.full((times_union.shape[0],), -1, device=times_union.device, dtype=torch.long))
    return is_gt, snap_idx


def _stay_mask_for_union(times_union: torch.Tensor, stay_intervals: List[Tuple[float, float]]) -> torch.Tensor:
    if not stay_intervals:
        return torch.zeros_like(times_union, dtype=torch.bool)
    mask = torch.zeros_like(times_union, dtype=torch.bool)
    for t0, t1 in stay_intervals:
        t0_t = torch.tensor(float(t0), dtype=times_union.dtype, device=times_union.device)
        t1_t = torch.tensor(float(t1), dtype=times_union.dtype, device=times_union.device)
        mask = mask | ((times_union >= t0_t) & (times_union <= t1_t))
    return mask


def build_union_batch(persons: List[PersonData], config: ModeSepConfig, device: torch.device) -> UnionBatch:
    # 1) Union of all snap times
    all_times = [p.times_snap for p in persons if p.times_snap.numel() > 0]
    if not all_times:
        raise ValueError("No snap times found for any person in the batch.")
    times_union = torch.unique(torch.cat(all_times), sorted=True)
    # Insert K_internal internal points inside each gap
    times_union = _insert_internal_points(times_union, config.K_internal)

    times_union = times_union.to(device)
    B = len(persons)
    T = times_union.shape[0]

    is_gt_union = torch.zeros((B, T), dtype=torch.bool, device=device)
    snap_indices = torch.full((B, T), -1, dtype=torch.long, device=device)
    stay_mask = torch.zeros((B, T), dtype=torch.bool, device=device)
    gt_interior_mask = torch.zeros((B, T), dtype=torch.bool, device=device)
    stay_non_gt_mask = torch.zeros((B, T), dtype=torch.bool, device=device)
    stay_loc_ids = torch.full((B, T), -1, dtype=torch.long, device=device)

    for i, p in enumerate(persons):
        is_gt, sidx = _times_to_union_mapping(times_union, p.times_snap, config.time_match_tol)
        is_gt_union[i] = is_gt
        snap_indices[i] = sidx
        stay_mask[i] = _stay_mask_for_union(times_union, p.stay_intervals)

        for (t0, t1, loc_idx) in p.stay_segments:
            t0_t = torch.tensor(float(t0), dtype=times_union.dtype, device=times_union.device)
            t1_t = torch.tensor(float(t1), dtype=times_union.dtype, device=times_union.device)
            in_seg = (times_union >= t0_t) & (times_union <= t1_t)
            stay_loc_ids[i, in_seg] = loc_idx
        # Compute interior GT-snap mask (exclude first/last)
        gt_interior = torch.zeros_like(is_gt)
        if is_gt.any():
            gt_idx = torch.nonzero(is_gt, as_tuple=False).squeeze(-1)  # indices in union
            if gt_idx.numel() >= 3:
                interior_idx = gt_idx[1:-1]   # drop first and last
                gt_interior[interior_idx] = True
        gt_interior_mask[i] = gt_interior
        stay_non_gt_mask[i] = stay_mask[i] & (~is_gt) # stay points that are not snaps

    diffs = times_union[1:] - times_union[:-1]
    min_dt = float(diffs.min().item()) if diffs.numel() > 0 else 1.0

    return UnionBatch(
        times_union=times_union,
        is_gt_union=is_gt_union,
        snap_indices=snap_indices,
        stay_mask=stay_mask,
        gt_interior_mask=gt_interior_mask,
        stay_non_gt_mask=stay_non_gt_mask,
        stay_loc_ids=stay_loc_ids,
        min_dt=min_dt,
    )

