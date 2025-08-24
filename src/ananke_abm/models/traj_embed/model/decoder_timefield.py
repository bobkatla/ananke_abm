import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from ananke_abm.models.traj_embed.model.utils_bases import fourier_time_features

class TimeFieldDecoder(nn.Module):
    """
    Continuous time-field decoder:
        u_p(t; z) = w_p(z) · phi_time(t)  + alpha * log(lambda_p(t)) + masks
        q_p(t; z) = softmax_p( u_p(t; z) )
    """
    def __init__(self, P:int, m_latent:int, d_p:int, K_decoder_time:int, alpha_prior:float=1.0):
        super().__init__()
        self.P = P
        self.m = m_latent
        self.d_p = d_p
        self.K = K_decoder_time
        self.alpha = alpha_prior
        in_dim = m_latent + d_p
        out_dim = 1 + 2*K_decoder_time  # number of Fourier features
        self.coeff_mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def time_coeff(self, z: torch.Tensor, e_p: torch.Tensor):
        """
        Args:
            z: (B, m)
            e_p: (P, d_p)
        Returns:
            C: (B, P, 2K+1) time-basis coefficients per purpose
        """
        B = z.shape[0]
        z_tiled = z.unsqueeze(1).expand(B, self.P, self.m)     # (B,P,m)
        e_tiled = e_p.unsqueeze(0).expand(B, self.P, self.d_p) # (B,P,d_p)
        inp = torch.cat([z_tiled, e_tiled], dim=-1)            # (B,P,m+d_p)
        C = self.coeff_mlp(inp)                                # (B,P,2K+1)
        return C

    def utilities(self, z: torch.Tensor, e_p: torch.Tensor, t: torch.Tensor, log_lambda_p_t: torch.Tensor,
                  masks: Optional[Dict[str, torch.Tensor]]=None, Phi: Optional[torch.Tensor]=None):
        """
        Compute utilities u_p(t; z).
        Args:
            z: (B,m)
            e_p: (P,d_p)
            t: (Q,)
            log_lambda_p_t: (P,Q) prior log-densities
            masks: optional dict:
                - 'open_allowed': (P,) bool (allowed purposes at t=0)
                - 'close_allowed': (P,) bool (allowed purposes at t=1)
                - 'home_idx': int (purpose index for Home; forced at t=0)
                - 'home_idx_end': int (forced at t=1)
        Returns:
            u: (B,P,Q)
        """
        if Phi is None:
            Phi = fourier_time_features(t, self.K)    # (Q, 2K+1)
        else:
            Phi = Phi.to(t.device)
        C = self.time_coeff(z, e_p)             # (B,P,2K+1)
        u = torch.matmul(C, Phi.T)              # (B,P,Q)
        u = u + self.alpha * log_lambda_p_t.unsqueeze(0)  # (1,P,Q) -> (B,P,Q)

        if masks is not None:
            big_neg = -1e9
            big_pos = 1e9
            # Open at t=0
            if "open_allowed" in masks and masks["open_allowed"] is not None:
                allowed = masks["open_allowed"].view(1, -1)  # (1,P)
                block = ~allowed
                u[:, block[0], 0] = big_neg
            # Close at t=1
            if "close_allowed" in masks and masks["close_allowed"] is not None:
                allowed = masks["close_allowed"].view(1, -1)  # (1,P)
                block = ~allowed
                u[:, block[0], -1] = big_neg
            # Force Home ends
            if "home_idx" in masks and masks["home_idx"] is not None:
                idx = int(masks["home_idx"])
                u[:, :, 0] = big_neg
                u[:, idx, 0] = big_pos
            if "home_idx_end" in masks and masks["home_idx_end"] is not None:
                idx = int(masks["home_idx_end"])
                u[:, :, -1] = big_neg
                u[:, idx, -1] = big_pos
        return u

    def soft_assign(self, u: torch.Tensor):
        """Softmax over purposes: (B,P,Q) -> (B,P,Q)."""
        return F.softmax(u, dim=1)

    @staticmethod
    def tv_loss(q: torch.Tensor, w: torch.Tensor):
        """Total-variation surrogate along time: sum_p Σ |Δ q_p|."""
        diff = q[:, :, 1:] - q[:, :, :-1]  # (B,P,Q-1)
        tv = diff.abs().sum(dim=(1,2))     # (B,)
        return tv.mean()

    @staticmethod
    def ce_loss(q: torch.Tensor, y: torch.Tensor, w: torch.Tensor):
        """Time-integrated cross-entropy with quadrature weights."""
        eps = 1e-8
        logq = torch.log(q + eps)          # (B,P,Q)
        ce = -(y * logq)                   # (B,P,Q)
        ce_w = ce * w.view(1,1,-1)
        return ce_w.sum(dim=(1,2)).mean()

    @staticmethod
    def emd1d_loss(q: torch.Tensor, y: torch.Tensor, w: torch.Tensor):
        """
        1D Wasserstein-1 between time distributions per purpose.
        We compare shapes (mass-normalized per purpose) for stability.
        """
        qw = q * w.view(1,1,-1)  # (B,P,Q)
        yw = y * w.view(1,1,-1)
        qsum = qw.sum(dim=-1, keepdim=True) + 1e-8
        ysum = yw.sum(dim=-1, keepdim=True) + 1e-8
        qn = qw / qsum
        yn = yw / ysum
        qcdf = torch.cumsum(qn, dim=-1)
        ycdf = torch.cumsum(yn, dim=-1)
        emd = (qcdf - ycdf).abs().sum(dim=-1)  # (B,P)
        return emd.mean()

    @staticmethod
    def durlen_loss(q: torch.Tensor, mu_d: torch.Tensor, sigma_d: torch.Tensor, w: torch.Tensor):
        """
        Average block length surrogate using TV:
            M_p = ∫ q_p(t) dt ; TV_p ≈ Σ |Δ q_p| ; barL_p ≈ M_p / max(TV_p/2, eps)
        """
        qw = q * w.view(1,1,-1)
        M = qw.sum(dim=-1)              # (B,P)
        TV = (q[:, :, 1:] - q[:, :, :-1]).abs().sum(dim=-1)  # (B,P)
        barL = M / torch.clamp(TV/2.0, min=1e-6)             # (B,P)
        mu = mu_d.view(1,-1)
        sig2 = (sigma_d.view(1,-1)**2 + 1e-6)
        loss = ((barL - mu)**2 / sig2).mean()
        return loss

    @staticmethod
    def argmax_decode(u: torch.Tensor, t_dense: torch.Tensor):
        """
        Deterministic decoding on a dense grid.
        Returns list of length B; each element is a list of (p_idx, t0, d).
        """
        q = torch.softmax(u, dim=1)           # (B,P,L)
        p_star = q.argmax(dim=1)              # (B,L)
        B, L = p_star.shape
        out = []
        for b in range(B):
            idxs = p_star[b].cpu().numpy()
            ts = t_dense.cpu().numpy()
            segs = []
            start = 0
            for i in range(1, L):
                if idxs[i] != idxs[i-1]:
                    p = int(idxs[i-1])
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
