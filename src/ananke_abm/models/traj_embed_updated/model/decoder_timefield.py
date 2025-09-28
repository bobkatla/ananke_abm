import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List

from ananke_abm.models.traj_embed_updated.model.utils_bases import (
    make_alloc_grid,
    fourier_time_features,
)


class TimeFieldDecoder(nn.Module):
    """
    Continuous time-field decoder.

    Core idea:
        u_p(t; z) = C_p(z, e_p) · Φ_alloc(t)  +  α_p * log λ_p(clock(t))  + masks

    Two usage modes:
      1) Grid-based unaries for CRF (preferred in new pipeline):
           theta = utilities_on_grid(z, e_p, loglam_alloc, grid_type="train")
         where:
           - loglam_alloc: precomputed log λ_p(clock(t)) on the specified grid, shape [P, L]
           - grid_type: "train" or "eval" to select the precomputed basis

      2) Legacy continuous nodes (kept for back-compat and previews):
           u = utilities(z, e_p, t_nodes01, log_lambda_p_t, masks=None, Phi=None)

    Notes:
      - Φ_alloc is a Fourier basis over the ALLOCATION axis (not forced to be 24h periodic).
      - The 24h periodicity is injected through log λ_p(clock(t)), supplied by PDS.

    Phase 1 changes:
      - Per-purpose learnable α vector with initialization from names (idx2purpose mapping).
      - L2 pullback regularizer: L2(α - α_init) scaled by alpha_l2.
      - Backward-compatible: if no per-purpose init is provided, falls back to scalar alpha_prior.
    """

    def __init__(
        self,
        P: int,
        m_latent: int,
        d_p: int,
        K_decoder_time: int,
        alpha_prior: float = 1.0,
        time_cfg: Optional[Dict] = None,
        # --- Phase 1 optional additions (kept optional for back-compat) ---
        idx2purpose: Optional[List[str]] = None,
        alpha_init_per_purpose: Optional[Dict[str, float]] = None,
        alpha_l2: float = 1e-3,
        alpha_clamp_min: float = -3.0,
        alpha_clamp_max: float = 3.0,
    ):
        super().__init__()
        self.P = P
        self.m = m_latent
        self.d_p = d_p
        self.K = K_decoder_time

        # --- Decoder coefficient head (latent+purpose -> allocation Fourier coeffs) ---
        in_dim = m_latent + d_p
        out_dim = 1 + 2 * K_decoder_time  # DC + K cos + K sin
        self.coeff_mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

        # --- Cache Fourier bases for train/eval grids (stored on CPU; moved to device at call) ---
        if time_cfg is not None:
            for grid_type, step_mins in [
                ("train", time_cfg["TRAIN_GRID_MINS"]),
                ("eval",  time_cfg["VALID_GRID_MINS"]),
            ]:
                _, t_alloc01 = make_alloc_grid(
                    T_alloc_minutes=time_cfg["ALLOCATION_HORIZON_MINS"],
                    step_minutes=step_mins,
                    device="cpu",  # store on cpu, move to device in forward
                )
                basis = fourier_time_features(t_alloc01, self.K)   # [L, 2K+1]
                self.register_buffer(f"Phi_{grid_type}", basis)

        # -------------------------------------------------------------------------
        # Phase 1: Per-purpose α with pullback to initialization
        # -------------------------------------------------------------------------
        # Determine initialization vector for α in purpose index order.
        self.alpha_l2 = float(alpha_l2)
        self._alpha_min = float(alpha_clamp_min)
        self._alpha_max = float(alpha_clamp_max)

        if (alpha_init_per_purpose is not None) and (idx2purpose is not None):
            # Build α_init from provided map and index-to-name list
            alpha_init_vec = torch.tensor(
                [float(alpha_init_per_purpose.get(name, alpha_prior)) for name in idx2purpose],
                dtype=torch.float32,
            )
        else:
            # Back-compat: uniform init from scalar alpha_prior
            alpha_init_vec = torch.full((P,), float(alpha_prior), dtype=torch.float32)

        self.register_buffer("alpha_init", alpha_init_vec)         # [P], non-trainable reference
        self.alpha = nn.Parameter(alpha_init_vec.clone())          # [P], trainable
        # Track for logging (not used in loss directly)
        self.register_buffer("_last_alpha_reg", torch.tensor(0.0))

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

    # ---- Phase 1: regularizer for α ----
    def regularization_loss(self) -> torch.Tensor:
        """
        L2 pullback: ||alpha - alpha_init||_2^2 scaled by alpha_l2.
        Call from the trainer and add to total loss.
        """
        reg = F.mse_loss(self.alpha, self.alpha_init, reduction="sum")
        # store for optional logging
        self._last_alpha_reg = reg.detach()
        return self.alpha_l2 * reg

    # ---- internal: apply per-purpose α to clock prior grid ----
    def _apply_clock_bias(self, log_lambda_clock: torch.Tensor) -> torch.Tensor:
        """
        log_lambda_clock: [P,L] or [B,P,L]
        Returns α * log_lambda_clock with α broadcast over time (and batch if present).
        """
        alpha = torch.clamp(self.alpha, self._alpha_min, self._alpha_max)  # [P]
        if log_lambda_clock.dim() == 2:        # [P, L]
            return alpha[:, None] * log_lambda_clock
        elif log_lambda_clock.dim() == 3:      # [B, P, L]
            return alpha[None, :, None] * log_lambda_clock
        else:
            raise ValueError("log_lambda_clock must be [P,L] or [B,P,L]")

    # ---- grid-based unaries for CRF ----
    def utilities_on_grid(
        self,
        z: torch.Tensor,                 # [B, m]
        e_p: torch.Tensor,               # [P, d_p]
        loglam_alloc: torch.Tensor,      # [P, L] or [B, P, L] log λ_p(clock(t)) evaluated on this grid
        grid_type: str = "train",        # "train" or "eval"
        endpoint_mask: Optional[torch.Tensor] = None,  # [L, P] bool; only first/last rows matter
        neg_large: float = -1e4,
    ) -> torch.Tensor:
        """
        Produce unaries θ for CRF on a uniform allocation grid.

        Returns:
            theta: [B, P, L] log-unaries (no softmax)
        """
        Phi = getattr(self, f"Phi_{grid_type}").to(z.device)       # [L, 2K+1]
        L = Phi.shape[0]

        C = self.time_coeff(z, e_p)                                # [B,P,2K+1]
        theta_latent = torch.einsum("bpk,lk->bpl", C, Phi)         # [B,P,L]

        # add prior bias (broadcast over batch or already [B,P,L])
        clock_bias = self._apply_clock_bias(loglam_alloc)          # [B,P,L] or [P,L]
        if clock_bias.dim() == 2:                                  # [P,L] -> [B,P,L]
            clock_bias = clock_bias.unsqueeze(0).expand_as(theta_latent)

        theta = theta_latent + clock_bias                          # [B,P,L]

        # unified endpoint constraint (apply only at first/last positions)
        if endpoint_mask is not None:
            assert endpoint_mask.dim() == 2 and endpoint_mask.shape[0] == L, \
                "endpoint_mask must be [L, P]"
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

        # per-purpose α on arbitrary nodes
        alpha = torch.clamp(self.alpha, self._alpha_min, self._alpha_max)  # [P]
        u = u + alpha.view(1, -1, 1) * log_lambda_p_t.unsqueeze(0)

        # Legacy endpoint logic (prefer CRF path)
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
        class_weights: Optional[torch.Tensor] = None,   # [P]
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
