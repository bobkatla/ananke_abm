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


def stay_velocity(v: torch.Tensor, stay_mask: torch.Tensor) -> torch.Tensor:
    # v: [B, T, E]
    vel2 = (v.pow(2).sum(dim=-1))  # [B, T]
    if stay_mask.sum() == 0:
        return torch.tensor(0.0, dtype=v.dtype, device=v.device)
    return vel2[stay_mask].mean()


def total_loss(
    config: ModeSepConfig,
    logits: torch.Tensor,
    pred_emb: torch.Tensor,
    v: torch.Tensor,
    y_union: torch.Tensor,
    is_gt_mask: torch.Tensor,
    dist_mat: torch.Tensor,
    class_table: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss_ce = ce_at_snaps(logits, y_union, is_gt_mask)
    loss_mse = mse_at_snaps(pred_emb, y_union, class_table, is_gt_mask)
    loss_dist = expected_distance_at_snaps(logits, y_union, dist_mat, is_gt_mask)
    loss_stay = stay_velocity(v, stay_mask=is_gt_mask.new_zeros(is_gt_mask.shape).bool())  # placeholder
    # NOTE: stay loss is computed outside with true stay_mask and supplied in metrics; combine here properly below.
    # To keep API simple, compute with provided stay_mask separately.
    # We'll override loss_stay in the caller.

    # Weighted sum (caller can replace stay term)
    weighted = (
        config.w_ce * loss_ce +
        config.w_mse * loss_mse +
        config.w_dist * loss_dist +
        config.w_stay_vel * 0.0
    )

    metrics = {
        "ce": float(loss_ce.detach().item()),
        "mse": float(loss_mse.detach().item()),
        "dist": float(loss_dist.detach().item()),
        "stay_vel": float(0.0),
    }
    return weighted, metrics


