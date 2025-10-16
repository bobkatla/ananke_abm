import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List

from ananke_abm.models.traj_syn.core.utils_bases import (
    make_alloc_grid,
    fourier_time_features,
)


class TimeFieldDecoder(nn.Module):
    """
    Time-field decoder:
      u_p(t; z) = C_p(z, e_p) · Φ_alloc(t) + α_p * log λ_p(clock(t)) + masks

    Phase 1: per-purpose α with L2 pullback to init.
    Phase 3: L2 on latent allocation-Fourier coefficients (prefer prior for shape).
    """

    def __init__(
        self,
        P: int,
        m_latent: int,
        d_p: int,
        K_decoder_time: int,
        alpha_prior: float = 1.0,
        time_cfg: Optional[Dict] = None,
        idx2purpose: Optional[List[str]] = None,
        alpha_init_per_purpose: Optional[Dict[str, float]] = None,
        alpha_l2: float = 1e-3,
        alpha_clamp_min: float = -3.0,
        alpha_clamp_max: float = 3.0,
        coeff_l2_global: float = 0.0,
        coeff_l2_per_purpose: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.P = P
        self.m = m_latent
        self.d_p = d_p
        self.K = K_decoder_time

        # ----- coefficient head (latent + purpose -> allocation Fourier coeffs) -----
        in_dim = m_latent + d_p
        out_dim = 1 + 2 * K_decoder_time  # DC + K cos + K sin
        self.coeff_mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

        # ----- cache Fourier bases for train/eval grids -----
        if time_cfg is not None:
            for grid_type, step_mins in [
                ("train", time_cfg["TRAIN_GRID_MINS"]),
                ("eval",  time_cfg["VALID_GRID_MINS"]),
            ]:
                _, t_alloc01 = make_alloc_grid(
                    T_alloc_minutes=time_cfg["ALLOCATION_HORIZON_MINS"],
                    step_minutes=step_mins,
                    device="cpu",
                )
                basis = fourier_time_features(t_alloc01, self.K)   # [L, 2K+1]
                self.register_buffer(f"Phi_{grid_type}", basis)

        # -------------------------------------------------------------------------
        # Phase 1: per-purpose α with pullback to initialization
        # -------------------------------------------------------------------------
        self.alpha_l2 = float(alpha_l2)
        self._alpha_min = float(alpha_clamp_min)
        self._alpha_max = float(alpha_clamp_max)

        if (alpha_init_per_purpose is not None) and (idx2purpose is not None):
            alpha_init_vec = torch.tensor(
                [float(alpha_init_per_purpose.get(name, alpha_prior)) for name in idx2purpose],
                dtype=torch.float32,
            )
        else:
            alpha_init_vec = torch.full((P,), float(alpha_prior), dtype=torch.float32)

        self.register_buffer("alpha_init", alpha_init_vec)         # [P], non-trainable
        self.alpha = nn.Parameter(alpha_init_vec.clone())          # [P], trainable

        # -------------------------------------------------------------------------
        # Phase 3: coefficient L2 regularization
        # -------------------------------------------------------------------------
        # Build per-purpose weights for coeff L2 (default = coeff_l2_global)
        if coeff_l2_per_purpose is not None and idx2purpose is not None:
            coeff_w = torch.tensor(
                [float(coeff_l2_per_purpose.get(name, coeff_l2_global)) for name in idx2purpose],
                dtype=torch.float32,
            )
        else:
            coeff_w = torch.full((P,), float(coeff_l2_global), dtype=torch.float32)

        self.register_buffer("coeff_l2_per_p", coeff_w)            # [P], non-trainable
        # Stash last reg terms for logging
        self.register_buffer("_last_alpha_reg", torch.tensor(0.0))
        self.register_buffer("_last_coeff_reg", torch.tensor(0.0))
        # Pending coeff reg tensor for current forward (consumed by regularization_loss)
        self._pending_coeff_reg: Optional[torch.Tensor] = None

    # ---- coefficient head ----
    def time_coeff(self, z: torch.Tensor, e_p: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z:  [B, m]
            e_p:[P, d_p]
        Returns:
            C:  [B, P, 2K+1] allocation time-basis coefficients per purpose
        """
        B = z.shape[0]
        z_tiled = z.unsqueeze(1).expand(B, self.P, self.m)         # [B,P,m]
        e_tiled = e_p.unsqueeze(0).expand(B, self.P, self.d_p)     # [B,P,d_p]
        inp = torch.cat([z_tiled, e_tiled], dim=-1)                # [B,P,m+d_p]
        C = self.coeff_mlp(inp)                                    # [B,P,2K+1]
        return C

    # ---- Phase 1: regularizer for α ----
    def _alpha_pullback(self) -> torch.Tensor:
        reg = F.mse_loss(self.alpha, self.alpha_init, reduction="sum")
        self._last_alpha_reg = reg.detach()
        return self.alpha_l2 * reg

    def regularization_loss(self) -> torch.Tensor:
        """
        Returns Phase-1 alpha pullback + Phase-3 coeff L2 (if a batch computed it).
        Call this AFTER a forward that computed utilities_on_grid (sets _pending_coeff_reg).
        """
        reg = self._alpha_pullback()
        if self._pending_coeff_reg is not None:
            reg = reg + self._pending_coeff_reg
            # clear after consumption to avoid accidental reuse
            self._pending_coeff_reg = None
        return reg

    # ---- internal: apply per-purpose α to clock prior grid ----
    def _apply_clock_bias(self, log_lambda_clock: torch.Tensor) -> torch.Tensor:
        """
        log_lambda_clock: [P,L] or [B,P,L] -> returns α * log_lambda_clock
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
        loglam_alloc: torch.Tensor,      # [P, L] or [B, P, L] log λ_p(clock(t)) on this grid
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

        # Phase 3: compute coeff L2 reg for this batch (exclude DC term to avoid biasing level)
        # per-purpose penalty = mean_{B,K} (C[...,1:]^2)  * coeff_l2_per_p[p]
        C_no_dc = C[..., 1:]                                       # [B,P,2K]
        per_p_mean = (C_no_dc ** 2).mean(dim=(0, 2))               # [P]
        coeff_reg = (per_p_mean * self.coeff_l2_per_p).sum()       # scalar
        self._pending_coeff_reg = coeff_reg
        self._last_coeff_reg = coeff_reg.detach()

        # add clock prior bias (broadcast)
        clock_bias = self._apply_clock_bias(loglam_alloc)          # [B,P,L] or [P,L]
        if clock_bias.dim() == 2:
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

    # ---- LEGACY path kept intact (unchanged below) ----
    def utilities(
        self,
        z: torch.Tensor,
        e_p: torch.Tensor,
        t: torch.Tensor,
        log_lambda_p_t: torch.Tensor,
        masks: Optional[Dict[str, torch.Tensor]] = None,
        Phi: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if Phi is None:
            Phi = fourier_time_features(t, self.K)                 # [Q, 2K+1]
        else:
            Phi = Phi.to(t.device)

        C = self.time_coeff(z, e_p)                                # [B,P,2K+1]
        u = torch.matmul(C, Phi.T)                                 # [B,P,Q]

        alpha = torch.clamp(self.alpha, self._alpha_min, self._alpha_max)
        u = u + alpha.view(1, -1, 1) * log_lambda_p_t.unsqueeze(0)

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

    @staticmethod
    def soft_assign(u: torch.Tensor):
        return F.softmax(u, dim=1)

    @staticmethod
    def tv_loss(q: torch.Tensor, w: torch.Tensor):
        diff = q[:, :, 1:] - q[:, :, :-1]
        tv = diff.abs().sum(dim=(1, 2))
        return tv.mean()

    @staticmethod
    def ce_loss(q: torch.Tensor, y: torch.Tensor, w: torch.Tensor, class_weights: Optional[torch.Tensor] = None):
        eps = 1e-8
        logq = torch.log(q + eps)
        ce = -(y * logq)
        if class_weights is not None:
            ce = ce * class_weights.view(1, -1, 1)
        ce_w = ce * w.view(1, 1, -1)
        return ce_w.sum(dim=(1, 2)).mean()

    @staticmethod
    def emd1d_loss(q: torch.Tensor, y: torch.Tensor, w: torch.Tensor):
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
        qw = q * w.view(1, 1, -1)
        M = qw.sum(dim=-1)
        TV = (q[:, :, 1:] - q[:, :, :-1]).abs().sum(dim=-1)
        barL = M / torch.clamp(TV / 2.0, min=1e-6)
        mu = mu_d.view(1, -1)
        sig2 = (sigma_d.view(1, -1) ** 2 + 1e-6)
        loss = ((barL - mu) ** 2 / sig2).mean()
        return loss

    @staticmethod
    def argmax_decode(u: torch.Tensor, t_dense: torch.Tensor):
        q = torch.softmax(u, dim=1)
        p_star = q.argmax(dim=1)
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
