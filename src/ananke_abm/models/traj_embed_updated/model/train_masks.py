from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import warnings

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
    endpoint_allowed: torch.Tensor  # [P] bool; True if purpose can appear at the very first/last step
    home_idx: Optional[int]         # index of 'Home' purpose if present
    force_home_ends: bool           # if True, only Home is allowed at ends

    def to(self, device: torch.device) -> "EndpointMasks":
        return EndpointMasks(
            endpoint_allowed=self.endpoint_allowed.to(device),
            home_idx=self.home_idx,
            force_home_ends=self.force_home_ends,
        )


def build_endpoint_mask(
    purposes_df: pd.DataFrame,
    purposes_order: list[str],
    force_home_ends: bool = True,
    can_open_close_col: str = "can_open_close_day",
) -> EndpointMasks:
    """
    Build a unified endpoint mask for start/end of the allocation window.

    Rules:
      - If force_home_ends == True and 'Home' exists -> only Home is allowed at t=0 and t=L-1.
      - Else if purposes_df has `can_open_close_col`, allow only those marked True at ends.
      - Else allow all purposes at ends.

    Args:
        purposes_df: CSV as DataFrame with at least a 'purpose' column and optional `can_open_close_col`.
        purposes_order: list of purpose names defining index order used by the model.
        force_home_ends: if True, strictly enforce Home at both ends (when present).
        can_open_close_col: column name indicating permission to open/close the day.

    Returns:
        EndpointMasks(endpoint_allowed=[P] bool, home_idx, force_home_ends)
    """
    name_to_idx = {p: i for i, p in enumerate(purposes_order)}
    P = len(purposes_order)

    endpoint_allowed = torch.ones(P, dtype=torch.bool)
    home_idx = name_to_idx.get("Home", None)

    if force_home_ends:
        if home_idx is None:
            warnings.warn(
                "force_home_ends=True but 'Home' not found in purposes; "
                "falling back to CSV/open-close permissions.",
                RuntimeWarning,
            )
        else:
            endpoint_allowed[:] = False
            endpoint_allowed[home_idx] = True
            return EndpointMasks(endpoint_allowed, home_idx, True)

    # If we reach here, either force_home_ends=False or Home missing
    if can_open_close_col in purposes_df.columns:
        endpoint_allowed[:] = False
        for _, r in purposes_df.iterrows():
            p = str(r.get("purpose", "")).strip()
            if p in name_to_idx and _parse_bool(r.get(can_open_close_col, False)):
                endpoint_allowed[name_to_idx[p]] = True
    else:
        # No column: leave all True (allow all at ends)
        pass

    return EndpointMasks(endpoint_allowed, home_idx, False)


def endpoint_time_mask(
    endpoint_allowed: torch.Tensor,
    L: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Expand a [P]-bool endpoint permission vector into a [L, P]-bool time mask
    that only constrains the first and last time steps.

    Args:
        endpoint_allowed: [P] bool; True if purpose allowed at endpoints.
        L: length of the time grid (e.g., 360 for 5-min over 30h).
        device: target device.

    Returns:
        M: [L, P] bool; rows 0 and L-1 copy endpoint_allowed; others are True.
    """
    if device is None:
        device = endpoint_allowed.device
    P = int(endpoint_allowed.numel())
    M = torch.ones(L, P, dtype=torch.bool, device=device)
    M[0, :] = endpoint_allowed.to(device)
    M[-1, :] = endpoint_allowed.to(device)
    return M


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
