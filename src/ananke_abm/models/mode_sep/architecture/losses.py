"""
Losses for mode_sep model.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from ananke_abm.models.mode_sep.config import ModeSepConfig


def ce_at_snaps(logits: torch.Tensor, y_union: torch.Tensor, is_gt_mask: torch.Tensor) -> torch.Tensor:
    # logits: [B, T, Z]; y_union: [B, T] (long with -1 where not GT); is_gt_mask: [B, T]
    mask = is_gt_mask
    if mask.sum() == 0:
        return torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
    logits_flat = logits[mask]         # [N, Z]
    targets_flat = y_union[mask]       # [N]
    return F.cross_entropy(logits_flat, targets_flat, reduction="mean")


def mse_at_snaps(pred_emb: torch.Tensor, y_union: torch.Tensor, class_table: torch.Tensor, is_gt_mask: torch.Tensor) -> torch.Tensor:
    # pred_emb: [B, T, E]
    mask = is_gt_mask
    if mask.sum() == 0:
        return torch.tensor(0.0, dtype=pred_emb.dtype, device=pred_emb.device)
    targets = class_table[y_union.clamp(min=0)]  # [B, T, E] but with junk rows for -1 that will be masked away
    diff2 = (pred_emb - targets).pow(2).sum(dim=-1)  # [B, T]
    return diff2[mask].mean()


def expected_distance_at_snaps(logits: torch.Tensor, y_union: torch.Tensor, dist_mat: torch.Tensor, is_gt_mask: torch.Tensor) -> torch.Tensor:
    # logits: [B, T, Z]; dist_mat: [Z, Z]
    mask = is_gt_mask
    if mask.sum() == 0:
        return torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
    probs = torch.softmax(logits, dim=-1)  # [B, T, Z]
    # Gather distance rows corresponding to ground truth
    gt_rows = dist_mat[y_union.clamp(min=0)]  # [B, T, Z]
    exp_dist = (gt_rows * probs).sum(dim=-1)  # [B, T]
    return exp_dist[mask].mean()


def _dist_to_classes(pred_emb: torch.Tensor, table: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    pred_emb: [B,T,E], table: [Z,E], idx: [B,T] long
    returns per-(B,T) Euclidean distance to the class at 'idx'.
    """
    # Gather class vectors
    target = table[idx.clamp_min(0)]                  # [B,T,E]
    return (pred_emb - target).pow(2).sum(dim=-1).sqrt()  # [B,T]


def travel_margin_loss(
    pred_emb: torch.Tensor,
    class_table: torch.Tensor,
    travel_mask: torch.Tensor,
    prev_idx: torch.Tensor,
    dest_idx: torch.Tensor,
    m_travel: float,
) -> torch.Tensor:
    if not travel_mask.any():
        return pred_emb.new_zeros(())
    d_prev = _dist_to_classes(pred_emb, class_table, prev_idx)  # [B,T]
    d_dest = _dist_to_classes(pred_emb, class_table, dest_idx)  # [B,T]
    # Encourage separation: d_prev - d_dest >= m_travel
    sep = d_prev - d_dest
    hinge = (m_travel - sep)[travel_mask].clamp(min=0.0)
    return hinge.mean()


def travel_monotonicity_loss(
    pred_emb: torch.Tensor,
    class_table: torch.Tensor,
    travel_mask: torch.Tensor,
    prev_idx: torch.Tensor,
    dest_idx: torch.Tensor,
    epsilon_mono: float,
) -> torch.Tensor:
    """
    Finite-difference monotonicity inside travel segments:
      d_prev(t+Δ) >= d_prev(t) - epsilon_mono   (moving away from prev)
      d_dest(t+Δ) <= d_dest(t) + epsilon_mono   (moving toward dest)
    Both applied where consecutive points belong to the same travel segment.
    """
    B, T, E = pred_emb.shape
    if T < 2:
        return pred_emb.new_zeros(())

    d_prev = _dist_to_classes(pred_emb, class_table, prev_idx)  # [B,T]
    d_dest = _dist_to_classes(pred_emb, class_table, dest_idx)  # [B,T]

    # Build "pair mask" for (t, t+1) that are both travel AND same segment (same prev/dest)
    mask_t   = travel_mask[:, :-1]
    mask_t1  = travel_mask[:, 1:]
    same_prev = (prev_idx[:, :-1] == prev_idx[:, 1:])
    same_dest = (dest_idx[:, :-1] == dest_idx[:, 1:])
    pair_mask = (mask_t & mask_t1 & same_prev & same_dest)

    if not pair_mask.any():
        return pred_emb.new_zeros(())

    # Differences on those pairs
    dprev_t   = d_prev[:, :-1][pair_mask]
    dprev_t1  = d_prev[:,  1:][pair_mask]
    ddest_t   = d_dest[:, :-1][pair_mask]
    ddest_t1  = d_dest[:,  1:][pair_mask]

    # Hinge penalties
    away_prev  = (dprev_t - dprev_t1 + epsilon_mono).clamp(min=0.0)  # penalize decreases
    toward_dest = (ddest_t1 - ddest_t + epsilon_mono).clamp(min=0.0) # penalize increases
    return (away_prev.mean() + toward_dest.mean()) * 0.5


def total_loss(
    config: ModeSepConfig,
    logits: torch.Tensor,
    pred_emb: torch.Tensor,
    y_union: torch.Tensor,
    is_gt_mask: torch.Tensor,
    dist_mat: torch.Tensor,
    class_table: torch.Tensor,
    travel_mask: torch.Tensor,
    prev_idx: torch.Tensor,
    dest_idx: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:

    loss_ce = ce_at_snaps(logits, y_union, is_gt_mask)
    loss_mse = mse_at_snaps(pred_emb, y_union, class_table, is_gt_mask)
    loss_dist = expected_distance_at_snaps(logits, y_union, dist_mat, is_gt_mask)
    loss_travel_margin = travel_margin_loss(
        pred_emb, class_table, travel_mask, prev_idx, dest_idx, config.m_travel
    )
    loss_travel_mono = travel_monotonicity_loss(
        pred_emb, class_table, travel_mask, prev_idx, dest_idx, config.epsilon_mono
    )

    # Weighted sum
    weighted = (
        config.w_ce * loss_ce +
        config.w_mse * loss_mse +
        config.w_dist * loss_dist +
        config.w_travel_margin * loss_travel_margin +
        config.w_travel_mono * loss_travel_mono
    )

    metrics = {
        "ce": float(loss_ce.detach().item()),
        "mse": float(loss_mse.detach().item()),
        "dist": float(loss_dist.detach().item()),
        "travel_margin": float(loss_travel_margin.detach().item()),
        "travel_mono": float(loss_travel_mono.detach().item()),
    }
    return weighted, metrics


