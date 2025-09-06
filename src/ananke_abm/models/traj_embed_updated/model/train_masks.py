from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import torch


def _parse_bool(x) -> bool:
    """Robust bool parser for CSV fields."""
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    if isinstance(x, (int, float)):
        return bool(int(x))
    s = str(x).strip().lower()
    return s in {"1", "y", "yes", "true", "t"}


@dataclass
class EndpointMasks:
    """Container for endpoint constraints."""
    open_allowed: torch.Tensor  # [P] bool; True if purpose can appear at the very first/last step
    close_allowed: torch.Tensor  # [P] bool; True if purpose can appear at the very first/last step

    def to(self, device: torch.device) -> "EndpointMasks":
        return EndpointMasks(
            open_allowed=self.open_allowed.to(device),
            close_allowed=self.close_allowed.to(device),
        )


def build_endpoint_mask(
    purposes_df: pd.DataFrame,
    purposes_order: list[str],
    can_open_col: str = "can_open_day",
    can_close_col: str = "can_close_day",
    device: str | torch.device = "cpu",
) -> EndpointMasks:
    """
    Build a unified endpoint mask for start/end of the allocation window.

    Rules:
      - If purposes_df has `can_open_col` and `can_close_col`, allow only those marked True at ends.
      - Else allow all purposes at ends.

    Args:
        purposes_df: CSV as DataFrame with at least a 'purpose' column and optional `can_open_col` and `can_close_col`.
        purposes_order: list of purpose names defining index order used by the model.
        can_open_col: column name indicating permission to open the day.
        can_close_col: column name indicating permission to close the day.
        device: target device.
    Returns:
        EndpointMasks(open_allowed=[P] bool, close_allowed=[P] bool)
    """
    name_to_idx = {p: i for i, p in enumerate(purposes_order)}
    P = len(purposes_order)

    open_allowed = torch.ones(P, dtype=torch.bool)
    close_allowed = torch.ones(P, dtype=torch.bool)

    if can_open_col in purposes_df.columns:
        open_allowed[:] = False
        for _, r in purposes_df.iterrows():
            p = str(r.get("purpose", "")).strip()
            if p in name_to_idx and _parse_bool(r.get(can_open_col, False)):
                open_allowed[name_to_idx[p]] = True

    if can_close_col in purposes_df.columns:
        close_allowed[:] = False
        for _, r in purposes_df.iterrows():
            p = str(r.get("purpose", "")).strip()
            if p in name_to_idx and _parse_bool(r.get(can_close_col, False)):
                close_allowed[name_to_idx[p]] = True

    return EndpointMasks(open_allowed, close_allowed).to(device)


def endpoint_time_mask(
    open_allowed: torch.Tensor,
    close_allowed: torch.Tensor,
    L: int,
    step_mins: int,
    head_open_mins: int = 60,
    tail_close_mins: int = 30,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Expand a [P]-bool endpoint permission vector into a [L, P]-bool time mask
    that only constrains the first and last time steps.

    Args:
        open_allowed: [P] bool; True if purpose allowed at endpoints.
        close_allowed: [P] bool; True if purpose allowed at endpoints.
        L: length of the time grid (e.g., 360 for 5-min over 30h).
        device: target device.

    Returns:
        M: [L, P] bool; rows 0 and L-1 copy open_allowed and close_allowed; others are True.
    """
    P = open_allowed.numel()
    mask = torch.ones(L, P, dtype=torch.bool, device=device)

    head_bins = max(1, int(round(head_open_mins / max(1, step_mins)))) if head_open_mins > 0 else 1
    tail_bins = max(1, int(round(tail_close_mins / max(1, step_mins)))) if tail_close_mins > 0 else 1
    head_bins = min(head_bins, L)
    tail_bins = min(tail_bins, L)
    
    # Head window
    if head_open_mins > 0:
        mask[:head_bins, :] = False
        mask[:head_bins, open_allowed] = True
    
    # Tail window
    if tail_close_mins > 0:
        mask[-tail_bins:, :] = False
        mask[-tail_bins:, close_allowed] = True
    
    # safety constraints
    mask[0, :] = False
    mask[0, open_allowed] = True
    mask[-1, :] = False
    mask[-1, close_allowed] = True
    return mask


def apply_endpoint_mask_inplace(theta: torch.Tensor, endpoint_mask: torch.Tensor, neg_large: float = -1e4) -> None:
    """
    In-place apply endpoint mask to unaries (log-potentials) for CRF/argmax decoders.

    Args:
        theta: [B, P, L] log-unaries (will be modified in place).
        endpoint_mask: [L, P] bool (True = allowed). Typically from `endpoint_time_mask`.
        neg_large: value to add to forbidden entries (kept finite for numerical stability).

    Effect:
        For t in {0, L-1}, theta[:, ~allowed, t] += neg_large  (i.e., ~∞ negative).
    """
    assert theta.dim() == 3, "theta must be [B, P, L]"
    B, P, L = theta.shape
    assert endpoint_mask.shape == (L, P), f"endpoint_mask must be [L, P], got {endpoint_mask.shape}"

    # Identify forbidden at endpoints
    forbid_first = ~endpoint_mask[0]   # [P]
    forbid_last  = ~endpoint_mask[-1]  # [P]

    if forbid_first.any():
        theta[:, forbid_first, 0] += neg_large
    if forbid_last.any():
        theta[:, forbid_last, -1] += neg_large
