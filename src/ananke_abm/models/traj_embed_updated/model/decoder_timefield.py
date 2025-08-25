import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from ananke_abm.models.traj_embed_updated.model.utils_bases import fourier_time_features


class TimeFieldDecoder(nn.Module):
    """
    Continuous time-field decoder.

    Core idea:
        u_p(t; z) = C_p(z, e_p) · Φ_alloc(t)  +  α_p * log λ_p(clock(t))  + masks

    Two usage modes:
      1) Grid-based unaries for CRF (preferred in new pipeline):
           theta = utilities_on_grid(z, e_p, t_alloc01, loglam_alloc, endpoint_mask)
         where:
           - t_alloc01: allocation grid in [0,1] (e.g., 30h normalized)
           - loglam_alloc: precomputed log λ_p(clock(t)) on that grid, shape [P, L]
           - endpoint_mask: [L, P] bool (True=allowed) only constraining first/last steps

      2) Legacy continuous nodes (kept for back-compat and previews):
           u = utilities(z, e_p, t_nodes01, log_lambda_p_t, masks=None, Phi=None)

    Notes:
      - Φ_alloc is a Fourier basis over the ALLOCATION axis (not forced to be 24h periodic).
      - The 24h periodicity is injected through log λ_p(clock(t)), supplied by PDS.
    """

    def __init__(self, P: int, m_latent: int, d_p: int, K_decoder_time: int, alpha_prior: float = 1.0):
        super().__init__()
        self.P = P
        self.m = m_latent
        self.d_p = d_p
        self.K = K_decoder_time

        # Per-purpose scale for the prior bias (learnable)
        self.alpha_per_p = nn.Parameter(torch.ones(P) * alpha_prior)

        in_dim = m_latent + d_p
        out_dim = 1 + 2 * K_decoder_time  # DC + K cos + K sin
        self.coeff_mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    # ---- coefficient head ----
    def time_coeff(self, z: torch.Tensor, e_p: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z:  [B, m]
            e_p:[P, d_p]
        Returns:
            C:  [B, P, 2K+1] time-basis coefficients per purpose
        """
        B = z.shape[0]
        z_tiled = z.unsqueeze(1).expand(B, self.P, self.m)         # [B,P,m]
        e_tiled = e_p.unsqueeze(0).expand(B, self.P, self.d_p)     # [B,P,d_p]
        inp = torch.cat([z_tiled, e_tiled], dim=-1)                # [B,P,m+d_p]
        C = self.coeff_mlp(inp)                                    # [B,P,2K+1]
        return C

    # ---- NEW: grid-based unaries for CRF ----
    @staticmethod
    def _basis_from_alloc_t01(t_alloc01: torch.Tensor, K: int) -> torch.Tensor:
        """
        Build Fourier basis Φ on the ALLOCATION axis.
        Args:
            t_alloc01: [L] tensor in [0,1] (allocation-normalized time)
            K: number of harmonic pairs
        Returns:
            Φ: [L, 2K+1]
        """
        return fourier_time_features(t_alloc01, K)  # [L, 2K+1]

    def utilities_on_grid(
        self,
        z: torch.Tensor,                 # [B, m]
        e_p: torch.Tensor,               # [P, d_p]
        t_alloc01: torch.Tensor,         # [L] allocation-normalized grid (e.g., 30h / T_alloc)
        loglam_alloc: torch.Tensor,      # [P, L] log λ_p(clock(t)) evaluated on this grid
        endpoint_mask: Optional[torch.Tensor] = None,  # [L, P] bool; only first/last rows matter
        Phi: Optional[torch.Tensor] = None,            # [L, 2K+1] optional precomputed allocation basis
        neg_large: float = -1e4,
    ) -> torch.Tensor:
        """
        Produce unaries θ for CRF on a uniform allocation grid.

        Returns:
            theta: [B, P, L] log-unaries (no softmax)
        """
        if Phi is None:
            Phi = self._basis_from_alloc_t01(t_alloc01, self.K)    # [L, 2K+1]
        else:
            # ensure device match
            Phi = Phi.to(t_alloc01.device)

        C = self.time_coeff(z, e_p)                                # [B,P,2K+1]
        # u = C · Φ^T
        theta = torch.einsum("bpk,lk->bpl", C, Phi)                # [B,P,L]

        # add prior bias (broadcast over batch)
        theta = theta + self.alpha_per_p.view(1, -1, 1) * loglam_alloc.unsqueeze(0)  # [B,P,L]

        # unified endpoint constraint (apply only at first/last positions)
        if endpoint_mask is not None:
            assert endpoint_mask.dim() == 2 and endpoint_mask.shape[0] == t_alloc01.shape[0], \
                "endpoint_mask must be [L, P]"
            L = t_alloc01.shape[0]
            forbid_first = ~endpoint_mask[0]    # [P]
            forbid_last  = ~endpoint_mask[-1]   # [P]
            if forbid_first.any():
                theta[:, forbid_first, 0] += neg_large
            if forbid_last.any():
                theta[:, forbid_last, -1] += neg_large

        return theta

    # ---- LEGACY: arbitrary nodes path (kept for back-compat / previews) ----
    def utilities(
        self,
        z: torch.Tensor,                     # [B,m]
        e_p: torch.Tensor,                   # [P,d_p]
        t: torch.Tensor,                     # [Q] normalized nodes in [0,1] (allocation axis)
        log_lambda_p_t: torch.Tensor,        # [P,Q] prior log-densities at these nodes
        masks: Optional[Dict[str, torch.Tensor]] = None,
        Phi: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute utilities u_p(t; z) on arbitrary (normalized) nodes.
        NOTE: This is DEPRECATED for training; prefer `utilities_on_grid` for CRF.

        Legacy masks supported keys (will be removed):
          - 'open_allowed'    : [P] bool (allowed at t=0)
          - 'close_allowed'   : [P] bool (allowed at t=1)
          - 'home_idx'        : int (force Home at t=0)
          - 'home_idx_end'    : int (force Home at t=1)
        """
        if Phi is None:
            Phi = fourier_time_features(t, self.K)                 # [Q, 2K+1]
        else:
            Phi = Phi.to(t.device)

        C = self.time_coeff(z, e_p)                                # [B,P,2K+1]
        u = torch.matmul(C, Phi.T)                                 # [B,P,Q]
        u = u + self.alpha_per_p.view(1, -1, 1) * log_lambda_p_t.unsqueeze(0)

        # Legacy endpoint logic (separate open/close/home). Prefer unified mask in CRF path.
        if masks is not None:
            big_neg = -1e9
            big_pos = 1e9
            if "open_allowed" in masks and masks["open_allowed"] is not None:
                allowed = masks["open_allowed"].view(1, -1)
                block = ~allowed
                u[:, block[0], 0] = big_neg
            if "close_allowed" in masks and masks["close_allowed"] is not None:
                allowed = masks["close_allowed"].view(1, -1)
                block = ~allowed
                u[:, block[0], -1] = big_neg
            if "home_idx" in masks and masks["home_idx"] is not None:
                idx = int(masks["home_idx"])
                u[:, :, 0] = big_neg
                u[:, idx, 0] = big_pos
            if "home_idx_end" in masks and masks["home_idx_end"] is not None:
                idx = int(masks["home_idx_end"])
                u[:, :, -1] = big_neg
                u[:, idx, -1] = big_pos
        return u

    # ---- Back-compat helpers (not used in CRF training) ----
    def soft_assign(self, u: torch.Tensor):
        """Softmax over purposes: (B,P,Q) -> (B,P,Q)."""
        return F.softmax(u, dim=1)

    @staticmethod
    def tv_loss(q: torch.Tensor, w: torch.Tensor):
        """Total-variation surrogate along time: sum_p Σ |Δ q_p|."""
        diff = q[:, :, 1:] - q[:, :, :-1]     # [B,P,Q-1]
        tv = diff.abs().sum(dim=(1, 2))       # [B]
        return tv.mean()

    @staticmethod
    def ce_loss(
        q: torch.Tensor,           # [B,P,Q]
        y: torch.Tensor,           # [B,P,Q]
        w: torch.Tensor,           # [Q]
        class_weights: torch.Tensor | None = None,   # [P]
    ):
        """Deprecated: time-integrated cross-entropy with quadrature weights."""
        eps = 1e-8
        logq = torch.log(q + eps)
        ce = -(y * logq)
        if class_weights is not None:
            ce = ce * class_weights.view(1, -1, 1)
        ce_w = ce * w.view(1, 1, -1)
        return ce_w.sum(dim=(1, 2)).mean()

    @staticmethod
    def emd1d_loss(q: torch.Tensor, y: torch.Tensor, w: torch.Tensor):
        """Deprecated: 1D Wasserstein-1 between time distributions per purpose."""
        qw = q * w.view(1, 1, -1)
        yw = y * w.view(1, 1, -1)
        qsum = qw.sum(dim=-1, keepdim=True) + 1e-8
        ysum = yw.sum(dim=-1, keepdim=True) + 1e-8
        qn = qw / qsum
        yn = yw / ysum
        qcdf = torch.cumsum(qn, dim=-1)
        ycdf = torch.cumsum(yn, dim=-1)
        emd = (qcdf - ycdf).abs().sum(dim=-1)
        return emd.mean()

    @staticmethod
    def durlen_loss(q: torch.Tensor, mu_d: torch.Tensor, sigma_d: torch.Tensor, w: torch.Tensor):
        """Deprecated: duration-length surrogate using TV."""
        qw = q * w.view(1, 1, -1)
        M = qw.sum(dim=-1)                               # [B,P]
        TV = (q[:, :, 1:] - q[:, :, :-1]).abs().sum(dim=-1)  # [B,P]
        barL = M / torch.clamp(TV / 2.0, min=1e-6)
        mu = mu_d.view(1, -1)
        sig2 = (sigma_d.view(1, -1) ** 2 + 1e-6)
        loss = ((barL - mu) ** 2 / sig2).mean()
        return loss

    @staticmethod
    def argmax_decode(u: torch.Tensor, t_dense: torch.Tensor):
        """
        Deterministic decoding on a dense grid (legacy argmax).
        Returns list of length B; each element is a list of (p_idx, t0, d).
        """
        q = torch.softmax(u, dim=1)           # [B,P,L]
        p_star = q.argmax(dim=1)              # [B,L]
        B, L = p_star.shape
        out = []
        for b in range(B):
            idxs = p_star[b].cpu().numpy()
            ts = t_dense.cpu().numpy()
            segs = []
            start = 0
            for i in range(1, L):
                if idxs[i] != idxs[i - 1]:
                    p = int(idxs[i - 1])
                    t0 = float(ts[start])
                    t1 = float(ts[i])
                    segs.append((p, t0, max(t1 - t0, 0.0)))
                    start = i
            p = int(idxs[-1])
            t0 = float(ts[start])
            t1 = float(ts[-1])
            segs.append((p, t0, max(t1 - t0, 0.0)))
            out.append(segs)
        return out
