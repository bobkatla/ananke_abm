import warnings
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


# =========================
# NEW: CRF grid rasterizers
# =========================

@torch.no_grad()
def rasterize_to_grid(
    segments_batch: List[List[Tuple[str, float, float]]],
    purpose_to_idx: Dict[str, int],
    L: int,
    fallback_purpose: Optional[str] = "Home",
) -> torch.Tensor:
    """
    Rasterize variable-length segment lists into integer class labels on a uniform grid.

    Args:
        segments_batch: list over batch; each item is a list of (purpose_str, t0_norm, d_norm),
                        with times normalized to [0,1] over the *allocation* horizon (e.g., 30h).
        purpose_to_idx: mapping from purpose string to class index [0..P-1].
        L: number of uniform grid steps (e.g., 360 for 5-min over 30h, 1800 for 1-min).
        fallback_purpose: if a grid step is uncovered by any segment, fill with this purpose
                          (default "Home" if present; else class 0).

    Returns:
        y_idx: LongTensor [B, L] with integer purpose labels per grid step.
               Grid step k represents interval [k/L, (k+1)/L) on the allocation axis.
    """
    B = len(segments_batch)
    P = len(purpose_to_idx)
    device = torch.device("cpu")  # caller can .to(device) afterwards

    # Choose fallback idx
    if fallback_purpose is not None and fallback_purpose in purpose_to_idx:
        fallback_idx = purpose_to_idx[fallback_purpose]
    else:
        fallback_idx = 0

    # Build grid in normalized allocation coordinates
    grid = torch.arange(L, dtype=torch.float32, device=device) / float(L)  # [L] left edges

    y_idx = torch.full((B, L), fill_value=fallback_idx, dtype=torch.long, device=device)

    for b, segs in enumerate(segments_batch):
        if not segs:
            continue
        # Build tensors for this sample
        t0 = torch.tensor([max(0.0, float(t0)) for _, t0, _ in segs], dtype=torch.float32, device=device)  # [S]
        d  = torch.tensor([max(0.0, float(d))  for _, _, d  in segs], dtype=torch.float32, device=device)
        t1 = torch.clamp(t0 + d, max=1.0)  # right-open; clip to [0,1]
        p  = torch.tensor([int(purpose_to_idx[str(pp)]) for pp, _, _ in segs], dtype=torch.long, device=device)  # [S]

        # Mask of coverage: [S, L] where step k is covered by seg s iff k/L ∈ [t0_s, t1_s)
        cover = (grid.unsqueeze(0) >= t0.unsqueeze(1)) & (grid.unsqueeze(0) < t1.unsqueeze(1))  # [S,L]

        if cover.any():
            # Assign class per covered step.
            # If multiple segments cover (shouldn't happen), the one with the largest class index wins (arbitrary but stable).
            assign = torch.where(cover, p.unsqueeze(1).expand_as(cover), torch.full_like(cover, fill_value=-1, dtype=torch.long))
            # Reduce across segments
            chosen = torch.max(assign, dim=0).values  # [L], values in {-1, 0..P-1}
            # Fill uncovered with fallback
            mask_cov = chosen >= 0
            y_idx[b, mask_cov] = chosen[mask_cov]

    return y_idx


@torch.no_grad()
def rasterize_from_padded_to_grid(
    p_idx_pad: torch.Tensor,   # [B, Lmax] long
    t0_pad: torch.Tensor,      # [B, Lmax] float in [0,1]
    d_pad: torch.Tensor,       # [B, Lmax] float in [0,1]
    lengths: List[int],
    L: int,
    fallback_idx: int = 0,
) -> torch.Tensor:
    """
    Vectorized rasterizer from padded tensors to integer labels on a uniform grid.

    Args:
        p_idx_pad: [B, Lmax] class indices per segment.
        t0_pad:    [B, Lmax] normalized start times in [0,1].
        d_pad:     [B, Lmax] normalized durations in [0,1].
        lengths:   list of valid segment counts per batch item.
        L:         number of grid steps (e.g., 360 or 1800) over the allocation window.
        fallback_idx: class index to use for uncovered steps.

    Returns:
        y_idx: LongTensor [B, L] with integer labels per grid step.
    """
    device = t0_pad.device
    B, Lmax = p_idx_pad.shape
    grid = torch.arange(L, device=device, dtype=torch.float32) / float(L)  # [L]

    lengths_t = torch.as_tensor(lengths, device=device)
    valid = (torch.arange(Lmax, device=device).unsqueeze(0) < lengths_t.unsqueeze(1))  # [B, Lmax]

    # Compute segment coverage mask on the grid
    t1_pad = torch.clamp(t0_pad + d_pad, max=1.0)
    # Broadcast to [B, Lmax, L]
    cover = (grid.view(1, 1, L) >= t0_pad.unsqueeze(-1)) & (grid.view(1, 1, L) < t1_pad.unsqueeze(-1))
    cover = cover & valid.unsqueeze(-1)

    # Assign per covered cell the class index; else -1
    assign = torch.where(
        cover,
        p_idx_pad.unsqueeze(-1).expand_as(cover),
        torch.full_like(cover, fill_value=-1, dtype=torch.long),
    )  # [B, Lmax, L]

    # Reduce across segments: for each (b, k), pick the (unique) covering seg's class (or -1 if none)
    chosen = torch.max(assign, dim=1).values  # [B, L]

    # Fill uncovered with fallback_idx
    y_idx = torch.where(chosen >= 0, chosen, torch.full_like(chosen, fallback_idx))

    return y_idx


# ======================================
# (DEPRECATED) Quadrature-based helpers
# ======================================

@torch.no_grad()
def rasterize_from_padded(
    p_idx_pad: torch.Tensor,
    t0_pad: torch.Tensor,
    d_pad: torch.Tensor,
    lengths,
    P: int,
    t_nodes: torch.Tensor,
) -> torch.Tensor:
    """
    DEPRECATED for training. Kept for back-compat with older validation code.
    Vectorized rasterizer: (B,L), (B,L), (B,L), lengths, P, (Q,) -> (B,P,Q).
    """
    warnings.warn(
        "rasterize_from_padded is deprecated in the CRF-VAE pipeline. "
        "Use rasterize_from_padded_to_grid for grid labels.",
        DeprecationWarning,
    )
    device = t_nodes.device
    B, L = p_idx_pad.shape
    lengths_t = torch.as_tensor(lengths, device=p_idx_pad.device)
    valid = (torch.arange(L, device=p_idx_pad.device).unsqueeze(0) < lengths_t.unsqueeze(1))  # (B,L)
    t1_pad = t0_pad + d_pad
    tq = t_nodes.view(1, 1, -1)  # (1,1,Q)
    seg_mask = (tq >= t0_pad.unsqueeze(-1)) & (tq < t1_pad.unsqueeze(-1))  # (B,L,Q)
    seg_mask = seg_mask & valid.unsqueeze(-1)
    seg_mask = seg_mask.to(torch.float32)
    onehot = F.one_hot(p_idx_pad.clamp_min(0), num_classes=P).to(seg_mask.dtype)  # (B,L,P)
    y = torch.einsum('blp,blq->bpq', onehot, seg_mask)
    return y.clamp_(0, 1)


@torch.no_grad()
def rasterize_batch(
    segments_batch,
    purpose_to_idx,
    t_nodes,
):
    """
    DEPRECATED for training. Kept for back-compat with older validation code.
    Convert batch of variable-length segments into one-hot over arbitrary nodes.
    """
    warnings.warn(
        "rasterize_batch is deprecated in the CRF-VAE pipeline. "
        "Use rasterize_to_grid for integer labels on a uniform grid.",
        DeprecationWarning,
    )
    B = len(segments_batch)
    Q = t_nodes.shape[0]
    P = len(purpose_to_idx)
    y = torch.zeros((B, P, Q), dtype=torch.float32, device=t_nodes.device)
    for b, segs in enumerate(segments_batch):
        for (p, t0, d) in segs:
            p_idx = purpose_to_idx[p]
            t1 = t0 + d
            mask = (t_nodes >= t0) & (t_nodes < t1)
            y[b, p_idx, mask] = 1.0
    return y
