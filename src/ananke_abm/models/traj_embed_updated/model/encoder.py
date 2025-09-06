# ananke_abm/models/traj_embed/model/encoder.py
from __future__ import annotations
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ananke_abm.models.traj_embed_updated.model.utils_bases import fourier_time_features


def kl_gaussian_standard(mu: torch.Tensor, logvar: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    KL( N(mu, diag(exp(logvar))) || N(0, I) ) = 0.5 * sum( exp(logvar) + mu^2 - 1 - logvar )
    Args:
        mu:     [B, m]
        logvar: [B, m]
        reduction: "mean" | "sum" | "none"
    """
    kl = 0.5 * (torch.exp(logvar) + mu.pow(2) - 1.0 - logvar)
    if reduction == "mean":
        return kl.mean()
    if reduction == "sum":
        return kl.sum()
    return kl  # [B, m]


class TrajEncoderGRU(nn.Module):
    """
    β-VAE encoder for trajectories with clock-aware time tokens.

    Tokens per segment:
      [ e_p | Fourier(start_clock01; Kc) | Fourier(start_alloc01; Ka optional) | Fourier(duration; Kd) ]

    Sequence -> GRU -> concatenate (last, masked-mean) -> heads:
      mu, logvar  (pre-norm latent s);  z = s / ||s||
    """
    def __init__(
        self,
        d_p: int,
        K_time_token_clock: int = 4,
        K_time_token_alloc: int = 0,
        K_dur_token: int = 4,
        m_latent: int = 16,
        gru_hidden: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_p = d_p
        self.Kc = K_time_token_clock
        self.Ka = K_time_token_alloc
        self.Kd = K_dur_token
        self.m = m_latent
        self.H = gru_hidden

        in_dim = d_p
        in_dim += (2 * self.Kc + 1) if self.Kc > 0 else 0
        in_dim += (2 * self.Ka + 1) if self.Ka > 0 else 0
        in_dim += (2 * self.Kd + 1) if self.Kd > 0 else 0

        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=self.H,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Combine last hidden and masked mean
        feat_dim = 2 * self.H
        self.to_mu = nn.Linear(feat_dim, m_latent)
        self.to_logvar = nn.Linear(feat_dim, m_latent)

    def _make_tokens(
        self,
        p_idx_pad: torch.Tensor,   # [B, Lmax] long
        t_pad: torch.Tensor,       # [B, Lmax] in [0,1] over ALLOCATION window
        d_pad: torch.Tensor,       # [B, Lmax] in [0,1] (allocation-normalized durations)
        e_p: torch.Tensor,         # [P, d_p]
        T_alloc_minutes: Optional[int],
        T_clock_minutes: int,
    ) -> torch.Tensor:
        """
        Build per-step tokens.
        If T_alloc_minutes is provided, start_clock01 = ((t_pad*T_alloc) % T_clock)/T_clock.
        Otherwise, assume t_pad already encodes clock in [0,1].
        """
        B, Lmax = p_idx_pad.shape
        device = t_pad.device
        # Purpose embedding tokens
        ep_tok = e_p[p_idx_pad]  # [B, Lmax, d_p]

        # Clock-of-day Fourier features
        if T_alloc_minutes is not None:
            t_clock01 = ((t_pad * float(T_alloc_minutes)) % float(T_clock_minutes)) / float(T_clock_minutes)
        else:
            t_clock01 = t_pad  # assume already clock-normalized
        feats = [ep_tok]
        if self.Kc > 0:
            feats.append(fourier_time_features(t_clock01, self.Kc))  # [B,Lmax,2Kc+1]

        # Optional allocation-position Fourier features
        if self.Ka > 0:
            feats.append(fourier_time_features(t_pad, self.Ka))       # [B,Lmax,2Ka+1]

        # Duration Fourier features
        if self.Kd > 0:
            feats.append(fourier_time_features(d_pad, self.Kd))       # [B,Lmax,2Kd+1]

        x = torch.cat(feats, dim=-1)  # [B, Lmax, in_dim]
        return x

    def forward(
        self,
        p_idx_pad: torch.Tensor,
        t_pad: torch.Tensor,
        d_pad: torch.Tensor,
        lengths: List[int],
        e_p: torch.Tensor,
        T_alloc_minutes: Optional[int] = None,
        T_clock_minutes: int = 1440,
        sample: bool = True,
        return_repr: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            z:      [B, m]  (unit-normalized for decoding)
            s:      [B, m]  (pre-norm latent sample; z = s / ||s||)
            mu:     [B, m]
            logvar: [B, m]
            (optional) r: [B, 2H] concatenated representation
        """
        B, Lmax = p_idx_pad.shape
        x = self._make_tokens(p_idx_pad, t_pad, d_pad, e_p, T_alloc_minutes, T_clock_minutes)

        lengths_tensor = torch.as_tensor(lengths, device=x.device, dtype=torch.long)
        packed = pack_padded_sequence(x, lengths_tensor.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, h_last = self.gru(packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=Lmax)  # [B,Lmax,H]

        # masked mean pooling
        mask = (torch.arange(Lmax, device=x.device).unsqueeze(0) < lengths_tensor.unsqueeze(1))  # [B,Lmax]
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(out.dtype)
        mean = (out * mask.unsqueeze(-1)).sum(dim=1) / denom                                      # [B,H]

        last = h_last[-1]  # [B,H] (num_layers may be >1)
        r = torch.cat([last, mean], dim=-1)  # [B,2H]

        mu = self.to_mu(r)
        logvar = self.to_logvar(r)
        logvar = torch.clamp(logvar, -10, 8)

        if sample:
            eps = torch.randn_like(mu)
            s = mu + torch.exp(0.5 * logvar) * eps
        else:
            s = mu

        z = s / (s.norm(dim=-1, keepdim=True) + 1e-9)

        if return_repr:
            return z, s, mu, logvar, r
        return z, s, mu, logvar
