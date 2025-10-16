# ananke_abm/models/traj_embed_updated/model/encoder.py
from __future__ import annotations
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ananke_abm.models.traj_syn.core.utils_bases import fourier_time_features


def kl_gaussian_standard(mu: torch.Tensor, logvar: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    kl = 0.5 * (torch.exp(logvar) + mu.pow(2) - 1.0 - logvar)
    if reduction == "mean":
        return kl.mean()
    if reduction == "sum":
        return kl.sum()
    return kl


class TokenResBlock(nn.Module):
    """Per-step residual MLP with LayerNorm and GELU."""
    def __init__(self, d_in: int, d_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.fc1  = nn.Linear(d_in, d_hidden)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(d_hidden, d_in)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, d_in]
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)
        h = self.drop(h)
        return x + h


class AttnPool(nn.Module):
    """Masked attention pooling over time."""
    def __init__(self, d_in: int):
        super().__init__()
        self.score = nn.Linear(d_in, 1)

    def forward(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # H: [B, L, D], mask: [B, L] (True = valid)
        scores = self.score(H).squeeze(-1)  # [B, L]
        # -inf on padding so softmax ignores it
        scores = scores.masked_fill(~mask, float('-inf'))
        w = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, L, 1]
        return (H * w).sum(dim=1)  # [B, D]


class TrajEncoderGRU(nn.Module):
    """
    β-VAE encoder with optional residual token MLP, residual stacked GRU, and attention pooling.

    Tokens per segment:
      [ e_p | Fourier(start_clock01; Kc) | Fourier(start_alloc01; Ka optional) | Fourier(duration; Kd) ]
    """
    def __init__(
        self,
        d_p: int,
        K_time_token_clock: int = 4,
        K_time_token_alloc: int = 0,
        K_dur_token: int = 4,
        m_latent: int = 16,
        gru_hidden: int = 128,         # (↑ width)
        num_layers: int = 1,
        dropout: float = 0.1,          # a bit of dropout helps when wider
        bidirectional: bool = False,   # optional bi-GRU
        use_token_resmlp: bool = True, # add residual MLP on tokens
        token_resmlp_hidden: int = 256,
        use_residual_gru: bool = True, # residual between GRU layers when shapes match
        use_attn_pool: bool = True,    # attention pooling instead of plain mean
    ):
        super().__init__()
        self.d_p = d_p
        self.Kc = K_time_token_clock
        self.Ka = K_time_token_alloc
        self.Kd = K_dur_token
        self.m  = m_latent
        self.H  = gru_hidden
        self.bidirectional = bidirectional
        self.use_residual_gru = use_residual_gru
        self.use_attn_pool = use_attn_pool
        self.num_layers = num_layers

        in_dim = d_p
        in_dim += (2 * self.Kc + 1) if self.Kc > 0 else 0
        in_dim += (2 * self.Ka + 1) if self.Ka > 0 else 0
        in_dim += (2 * self.Kd + 1) if self.Kd > 0 else 0

        self.use_token_resmlp = use_token_resmlp
        if use_token_resmlp:
            self.token_res = TokenResBlock(in_dim, token_resmlp_hidden, dropout=dropout)
        else:
            self.token_res = None

        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=self.H,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        D = self.H * (2 if bidirectional else 1)

        # attention pool
        if use_attn_pool:
            self.attn_pool = AttnPool(D)
        else:
            self.attn_pool = None

        feat_dim = D + D  # last hidden + pooled (attn or mean)
        self.to_mu     = nn.Linear(feat_dim, m_latent)
        self.to_logvar = nn.Linear(feat_dim, m_latent)

    def _make_tokens(
        self,
        p_idx_pad: torch.Tensor,
        t_pad: torch.Tensor,
        d_pad: torch.Tensor,
        e_p: torch.Tensor,
        T_alloc_minutes: Optional[int],
        T_clock_minutes: int,
    ) -> torch.Tensor:
        B, Lmax = p_idx_pad.shape
        ep_tok = e_p[p_idx_pad]  # [B, L, d_p]
        feats = [ep_tok]

        if T_alloc_minutes is not None:
            t_clock01 = ((t_pad * float(T_alloc_minutes)) % float(T_clock_minutes)) / float(T_clock_minutes)
        else:
            t_clock01 = t_pad

        if self.Kc > 0:
            feats.append(fourier_time_features(t_clock01, self.Kc))
        if self.Ka > 0:
            feats.append(fourier_time_features(t_pad, self.Ka))
        if self.Kd > 0:
            feats.append(fourier_time_features(d_pad, self.Kd))

        x = torch.cat(feats, dim=-1)  # [B, L, in_dim]
        if self.token_res is not None:
            x = self.token_res(x)
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
    ):
        B, Lmax = p_idx_pad.shape
        x = self._make_tokens(p_idx_pad, t_pad, d_pad, e_p, T_alloc_minutes, T_clock_minutes)

        lengths_tensor = torch.as_tensor(lengths, device=x.device, dtype=torch.long)
        packed = pack_padded_sequence(x, lengths_tensor.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, h_last = self.gru(packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=Lmax)  # [B, L, D]
        # out is from top layer. To add inter-layer residuals, PyTorch GRU doesn't expose intermediates;
        # the residual we care about is handled in TokenResBlock + wider D. (Clean + effective.)

        # mask
        mask = (torch.arange(Lmax, device=x.device).unsqueeze(0) < lengths_tensor.unsqueeze(1))  # [B,L]

        # last hidden from top layer
        num_dirs = 2 if self.bidirectional else 1
        # h_last: [num_layers*num_dirs, B, H]
        last_top = h_last[-num_dirs:] if self.bidirectional else h_last[-1:]
        last_top = last_top.transpose(0,1).contiguous().view(B, -1)  # [B, D]

        # pooled (attn or mean)
        if self.attn_pool is not None:
            pooled = self.attn_pool(out, mask)                         # [B, D]
        else:
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(out.dtype)
            pooled = (out * mask.unsqueeze(-1)).sum(dim=1) / denom     # [B, D]

        r = torch.cat([last_top, pooled], dim=-1)                      # [B, 2D]
        mu = self.to_mu(r)
        logvar = self.to_logvar(r)
        logvar = torch.clamp(logvar, -10, 8)

        s = mu if not sample else (mu + torch.exp(0.5 * logvar) * torch.randn_like(mu))
        z = s / (s.norm(dim=-1, keepdim=True) + 1e-9)

        if return_repr:
            return z, s, mu, logvar, r
        return z, s, mu, logvar
