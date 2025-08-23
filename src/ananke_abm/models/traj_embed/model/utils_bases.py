import torch
import math
import numpy as np

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
