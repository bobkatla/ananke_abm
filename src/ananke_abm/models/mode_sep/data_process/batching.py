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
    travel_mask: torch.Tensor            # (B,T) bool — imputed points inside travel segments
    prev_zone_idx: torch.Tensor          # (B,T) long — zone at segment start (valid where travel_mask)
    dest_zone_idx: torch.Tensor          # (B,T) long — zone at segment end   (valid where travel_mask)
    progress_s: torch.Tensor             # (B,T) float in [0,1] — normalized progress (valid where travel_mask)
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
    """
    Build a shared union time grid across the batch, with masks/indices for:
      - GT snaps alignment
      - stay regions (incl. per-time stay location id)
      - interior GT snaps (exclude first & last)
      - non-GT points inside stays
      - travel interior points (between distinct consecutive GT snaps), plus:
          * prev_zone_idx (start zone of that travel segment)
          * dest_zone_idx (end zone of that travel segment)
          * progress_s \in [0,1] normalized time within that travel segment
    """
    # --- 1) Union time grid (GT snaps from all people) + K internal points per gap
    all_times = [p.times_snap for p in persons if p.times_snap.numel() > 0]
    if not all_times:
        raise ValueError("No snap times found for any person in the batch.")
    times_union = torch.unique(torch.cat(all_times), sorted=True)
    times_union = _insert_internal_points(times_union, config.K_internal).to(device)

    B = len(persons)
    T = int(times_union.shape[0])

    # --- 2) Allocate batch tensors
    is_gt_union      = torch.zeros((B, T), dtype=torch.bool,  device=device)
    snap_indices     = torch.full((B, T), -1, dtype=torch.long, device=device)
    stay_mask        = torch.zeros((B, T), dtype=torch.bool,  device=device)
    gt_interior_mask = torch.zeros((B, T), dtype=torch.bool,  device=device)
    stay_non_gt_mask = torch.zeros((B, T), dtype=torch.bool,  device=device)
    stay_loc_ids     = torch.full((B, T), -1, dtype=torch.long, device=device)

    # Travel metadata
    travel_mask    = torch.zeros((B, T), dtype=torch.bool,   device=device)
    prev_zone_idx  = torch.full((B, T), -1, dtype=torch.long, device=device)
    dest_zone_idx  = torch.full((B, T), -1, dtype=torch.long, device=device)
    progress_s     = torch.zeros((B, T), dtype=torch.float32, device=device)

    # --- 3) Per-person masks & indices
    for i, p in enumerate(persons):
        # (a) GT alignment
        is_gt, sidx = _times_to_union_mapping(times_union, p.times_snap, config.time_match_tol)
        is_gt_union[i]  = is_gt
        snap_indices[i] = sidx

        # (b) Stay mask over union & per-time stay location ids
        stay_mask[i] = _stay_mask_for_union(times_union, p.stay_intervals)
        # p.stay_segments: iterable of (t0, t1, loc_idx)
        for (t0, t1, loc_idx) in p.stay_segments:
            t0_t = torch.tensor(float(t0), dtype=times_union.dtype, device=device)
            t1_t = torch.tensor(float(t1), dtype=times_union.dtype, device=device)
            in_seg = (times_union >= t0_t) & (times_union <= t1_t)
            stay_loc_ids[i, in_seg] = int(loc_idx)

        # (c) Interior GT snaps (exclude first/last)
        if is_gt.any():
            gt_u_idx = torch.nonzero(is_gt, as_tuple=False).squeeze(-1)  # union indices of GT snaps
            if gt_u_idx.numel() >= 3:
                gt_interior_mask[i, gt_u_idx[1:-1]] = True

        # (d) Non-GT points inside stays
        stay_non_gt_mask[i] = stay_mask[i] & (~is_gt)

        # (e) Travel metadata from consecutive GT snaps with zone changes
        #     Build zone sequence at GT snaps on union axis
        if is_gt.any():
            gt_u_idx = torch.nonzero(is_gt, as_tuple=False).squeeze(-1)  # [S_u]
            if gt_u_idx.numel() >= 2:
                # Map those union-GT indices back to original snap indices, then to zone ids
                orig_snap_idx = sidx[gt_u_idx]           # [S_u] indices into p.loc_ids
                z_seq = p.loc_ids[orig_snap_idx]         # [S_u] long

                for a in range(len(gt_u_idx) - 1):
                    j0 = int(gt_u_idx[a].item())
                    j1 = int(gt_u_idx[a + 1].item())
                    z0 = int(z_seq[a].item())
                    z1 = int(z_seq[a + 1].item())

                    # Travel segment only if zone changes, and there are interior union points
                    if z0 != z1 and (j1 - j0) > 1:
                        interior = torch.arange(j0 + 1, j1, device=device)
                        travel_mask[i, interior]   = True
                        prev_zone_idx[i, interior] = z0
                        dest_zone_idx[i, interior] = z1

                        t0 = times_union[j0]
                        t1 = times_union[j1]
                        denom = (t1 - t0).clamp(min=1e-8)
                        progress_s[i, interior] = ((times_union[interior] - t0) / denom).clamp(0.0, 1.0)

    # --- 4) Union grid spacing diagnostics
    diffs = times_union[1:] - times_union[:-1]
    min_dt = float(diffs.min().item()) if diffs.numel() > 0 else 1.0

    # --- 5) Pack
    return UnionBatch(
        times_union=times_union,
        is_gt_union=is_gt_union,
        snap_indices=snap_indices,
        stay_mask=stay_mask,
        gt_interior_mask=gt_interior_mask,
        stay_non_gt_mask=stay_non_gt_mask,
        stay_loc_ids=stay_loc_ids,
        travel_mask=travel_mask,
        prev_zone_idx=prev_zone_idx,
        dest_zone_idx=dest_zone_idx,
        progress_s=progress_s,
        min_dt=min_dt,
    )
