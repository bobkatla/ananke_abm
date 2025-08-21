
# models/dualspace.py
# Dual-space autoencoder for daily activity schedules
# - Purpose embedding with optional FiLM from purpose meta attributes
# - Time features (start/duration) as continuous encodings
# - Transformer encoder -> latent z
# - Decoder predicts:
#     (i) per-slot purpose logits (K_max slots)
#     (ii) a SINGLE shared duration simplex over K_max slots (sum to T hours)
# - Starts are cumulative sums of durations => validity by construction

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def fourier_time_feats(x: torch.Tensor, period: float, n_harmonics: int = 4) -> torch.Tensor:
    """
    x: (...,) hours in [0, T]
    returns: (..., 2*n_harmonics) [sin(2πkx/T), cos(2πkx/T)]
    """
    ks = torch.arange(1, n_harmonics + 1, device=x.device, dtype=x.dtype)
    x2 = x.unsqueeze(-1)  # (..., 1)
    angles = 2.0 * math.pi * x2 * ks / period  # (..., K)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        L = x.size(1)
        return x + self.pe[:, :L, :]


class FiLM(nn.Module):
    """Feature-wise Linear Modulation from meta-attributes -> scales & shifts for purpose embeddings."""
    def __init__(self, meta_dim: int, emb_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(meta_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * emb_dim)
        )

    def forward(self, meta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # meta: (n_purposes, meta_dim)
        out = self.net(meta)  # (n_purposes, 2*emb_dim)
        gamma, beta = out.chunk(2, dim=-1)
        return gamma, beta


class PurposeEmbeddingWithFiLM(nn.Module):
    def __init__(self, n_purposes: int, emb_dim: int, pad_idx: int, meta: Optional[torch.Tensor] = None):
        super().__init__()
        self.embed = nn.Embedding(n_purposes, emb_dim, padding_idx=pad_idx)
        self.use_film = meta is not None
        if self.use_film:
            self.film = FiLM(meta_dim=meta.size(-1), emb_dim=emb_dim)
            # store a copy for forward; keep meta as a buffer (not a parameter)
            self.register_buffer("meta", meta, persistent=False)
        else:
            self.meta = None

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (B, L)
        e = self.embed(idx)  # (B, L, D)
        if self.use_film:
            gamma, beta = self.film(self.meta)  # (n_purposes, D) each
            gamma = gamma[idx]  # (B, L, D)
            beta = beta[idx]    # (B, L, D)
            e = e * (1.0 + torch.tanh(gamma)) + beta
        return e


@dataclass
class DualSpaceConfig:
    n_purposes: int
    pad_idx: int
    k_max: int = 10
    d_purpose: int = 64
    d_time: int = 16
    d_model: int = 256
    n_layers: int = 3
    n_heads: int = 4
    dropout: float = 0.1
    d_z: int = 64
    n_time_harmonics: int = 4
    label_smoothing: float = 0.05
    duration_temp: float = 1.0  # softmax temperature for durations
    use_vae: bool = False
    kl_beta: float = 0.0  # only used if use_vae
    lambda_lap: float = 0.0
    lambda_meta_probe: float = 0.0
    duration_entropy_weight: float = 0.01  # coeff for -sum w log w
    use_time_head: bool = True            # allow decoder to propose times from z when not teacher-forced
    day_hours: float = 24.0


class DualSpaceEncoder(nn.Module):
    def __init__(self, cfg: DualSpaceConfig, purpose_embed: PurposeEmbeddingWithFiLM):
        super().__init__()
        self.cfg = cfg
        self.purpose_embed = purpose_embed
        # time features: start Fourier + (duration/self.cfg.day_hours) + log1p(duration)
        time_dim = 2 * cfg.n_time_harmonics + 2
        self.time_proj = nn.Linear(time_dim, cfg.d_time)
        self.in_proj = nn.Linear(cfg.d_purpose + cfg.d_time, cfg.d_model)
        self.posenc = PositionalEncoding(cfg.d_model, max_len=cfg.k_max+8)
        enc_layer = nn.TransformerEncoderLayer(d_model=cfg.d_model, nhead=cfg.n_heads,
                                               dim_feedforward=cfg.d_model*4, dropout=cfg.dropout,
                                               batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        # to latent
        self.pool = nn.Linear(cfg.d_model, cfg.d_z)
        if cfg.use_vae:
            self.to_mu = nn.Linear(cfg.d_model, cfg.d_z)
            self.to_logvar = nn.Linear(cfg.d_model, cfg.d_z)

    def forward(self, purpose_idx: torch.Tensor, start: torch.Tensor, duration: torch.Tensor, pad_mask: torch.Tensor):
        """
        purpose_idx: (B, L) int64
        start, duration: (B, L) float (hours)
        pad_mask: (B, L) True for PAD positions (ignored)
        returns: z, extra dict
        """
        cfg = self.cfg
        # Build time features
        start_feats = fourier_time_feats(start, cfg.n_time_harmonics)  # (B,L,2H)
        dur_norm = duration / self.cfg.day_hours
        dur_log = torch.log1p(duration)
        t_feats = torch.cat([start_feats, dur_norm.unsqueeze(-1), dur_log.unsqueeze(-1)], dim=-1)  # (B,L,Td)
        t_enc = self.time_proj(t_feats)  # (B,L,d_time)

        p_enc = self.purpose_embed(purpose_idx)  # (B,L,d_purpose)
        x = torch.cat([p_enc, t_enc], dim=-1)  # (B,L,d_purpose+d_time)
        x = self.in_proj(x)
        x = self.posenc(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)  # (B,L,D)

        # Use masked mean pooling (ignore PAD)
        mask = (~pad_mask).float().unsqueeze(-1)  # (B,L,1)
        x_masked = x * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = x_masked.sum(dim=1) / denom  # (B,D)

        # x is the encoder output (B, L, D) with PAD positions present (masked via pad_mask during encoding)
        if self.cfg.use_vae:
            mu = self.to_mu(pooled)
            logvar = self.to_logvar(pooled)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z, {"mu": mu, "logvar": logvar}, x
        else:
            z = self.pool(pooled)
            return z, {}, x


class DualSpaceDecoder(nn.Module):
    def __init__(self, cfg: DualSpaceConfig, n_purposes: int):
        super().__init__()
        self.cfg = cfg
        self.z_proj = nn.Linear(cfg.d_z, cfg.d_model)

        # Query MLP: Fourier(start) [2H] + dur/self.cfg.day_hours [1] + log1p(dur) [1] + z_proj [D]
        q_in = (2 * cfg.n_time_harmonics) + 2 + cfg.d_model
        self.q_mlp = nn.Sequential(
            nn.Linear(q_in, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        # Optional head to propose durations (weights over K) directly from z
        if cfg.use_time_head:
            self.time_head = nn.Linear(cfg.d_z, cfg.k_max)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4, dropout=self.cfg.dropout,
            batch_first=True, activation="gelu"
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.n_layers)

        self.out_purpose = nn.Linear(cfg.d_model, n_purposes)  # logits per slot
        self.dur_slot = nn.Linear(cfg.d_model, 1)              # one logit per slot

    def _fourier_hours(self, x_hours: torch.Tensor) -> torch.Tensor:
        # x_hours: [B,K] in hours; return [B,K,2H] with period=T
        ks = torch.arange(1, self.cfg.n_time_harmonics + 1, device=x_hours.device, dtype=x_hours.dtype)
        x = x_hours.unsqueeze(-1)  # [B,K,1]
        ang = 2.0 * math.pi * x * ks / self.cfg.day_hours
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # [B,K,2H]

    def _build_queries(self, z: torch.Tensor, start_h: torch.Tensor, dur_h: torch.Tensor) -> torch.Tensor:
        """
        Build decoder queries from (start,duration) and z.
        start_h, dur_h: [B,K] in hours
        """
        B, K = start_h.shape
        z_ctx = self.z_proj(z).unsqueeze(1).expand(B, K, -1)  # [B,K,D]
        f_start = self._fourier_hours(start_h)                 # [B,K,2H]
        dur_days = (dur_h / self.cfg.day_hours).unsqueeze(-1)                # [B,K,1]
        dur_log  = torch.log1p(dur_h).unsqueeze(-1)            # [B,K,1]
        q_in = torch.cat([f_start, dur_days, dur_log, z_ctx], dim=-1)  # [B,K,2H+2+D]
        return self.q_mlp(q_in)  # [B,K,D]

    def forward(
        self,
        z: torch.Tensor,
        memory: torch.Tensor,
        start_hours: torch.Tensor,
        dur_hours: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        teacher_forced: bool = True,
        time_jitter_minutes: float = 0.0,
    ):
        """
        z: [B, d_z]
        memory: [B, L, D] (encoder tokens)
        start_hours, dur_hours: [B, K] (GT times in hours) used only if teacher_forced=True
        memory_key_padding_mask: [B, L] True for PAD
        teacher_forced: if False, decoder proposes its own times from z via time_head
        time_jitter_minutes: add U(-j,+j) minutes to start & dur for building queries (TF only)
        """
        device = z.device
        B, K = start_hours.shape

        if teacher_forced:
            if time_jitter_minutes > 0.0:
                # Jitter for queries only; clamp to keep valid hours
                j = (time_jitter_minutes / 60.0)
                start_h = (start_hours + (2*j)*torch.rand_like(start_hours) - j).clamp(min=0.0, max=self.cfg.day_hours)
                dur_h   = (dur_hours   + (2*j)*torch.rand_like(dur_hours)   - j).clamp(min=0.0)
                # Re-normalize durations to sum T hour per sample to keep validity by construction
                dur_sum = dur_h.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                dur_h = self.cfg.day_hours * dur_h / dur_sum
                # Recompute starts as cumulative
                start_h = dur_h.cumsum(dim=-1) - dur_h
            else:
                start_h, dur_h = start_hours, dur_hours
        else:
            # Predict a duration weight vector from z (softmax over K), build starts from it
            assert self.cfg.use_time_head, "use_time_head=False but teacher_forced=False path requested."
            logits_w = self.time_head(z)                 # [B,K]
            if self.cfg.duration_temp != 1.0:
                logits_w = logits_w / self.cfg.duration_temp
            w = F.softmax(logits_w, dim=-1)              # [B,K]
            dur_h = self.cfg.day_hours * w
            start_h = dur_h.cumsum(dim=-1) - dur_h       # [B,K]

        # Build queries and decode
        q = self._build_queries(z, start_h, dur_h)       # [B,K,D]
        dec = self.decoder(tgt=q, memory=memory, memory_key_padding_mask=memory_key_padding_mask)  # [B,K,D]

        # Heads
        purpose_logits = self.out_purpose(dec)           # [B,K,V]
        slot_logits = self.dur_slot(dec).squeeze(-1)     # [B,K]
        if self.cfg.duration_temp != 1.0:
            slot_logits = slot_logits / self.cfg.duration_temp
        w_pred = F.softmax(slot_logits, dim=-1)          # [B,K], sum=1
        durations = self.cfg.day_hours * w_pred
        starts = durations.cumsum(dim=-1) - durations

        return purpose_logits, durations, starts, w_pred


class DualSpaceAE(nn.Module):
    def __init__(self, cfg: DualSpaceConfig, purpose_embed: PurposeEmbeddingWithFiLM, n_purposes: int, pad_idx: int):
        super().__init__()
        self.cfg = cfg
        self.pad_idx = pad_idx
        self.encoder = DualSpaceEncoder(cfg, purpose_embed)
        self.decoder = DualSpaceDecoder(cfg, n_purposes=n_purposes)

    def forward(self, batch: Dict[str, torch.Tensor], teacher_forced: bool = True, time_jitter_minutes: float = 0.0) -> Dict[str, torch.Tensor]:
        z, extra, enc_tokens = self.encoder(
            batch["purpose_idx"], batch["start"], batch["duration"], batch["pad_mask"]
        )
        purpose_logits, durations, starts, w_pred = self.decoder(
            z, memory=enc_tokens,
            start_hours=batch["start"], dur_hours=batch["duration"],
            memory_key_padding_mask=batch["pad_mask"],
            teacher_forced=teacher_forced,
            time_jitter_minutes=time_jitter_minutes,
        )
        return {
            "z": z,
            "purpose_logits": purpose_logits,
            "durations": durations,
            "starts": starts,
            "w_pred": w_pred,   # simplex weights for losses
            **extra
        }


def label_smoothing_ce(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int, smoothing: float = 0.05) -> torch.Tensor:
    """
    logits: (B,K,C), targets: (B,K) int64
    """
    B, K, C = logits.shape
    logits = logits.view(B*K, C)
    targets = targets.view(B*K)
    # Create smoothed one-hot
    with torch.no_grad():
        confidence = 1.0 - smoothing
        low = smoothing / (C - 1)
        t = torch.full_like(logits, fill_value=low)
        t.scatter_(1, targets.unsqueeze(1), confidence)
        # zero out ignored positions
        mask = (targets != ignore_index).float().unsqueeze(1)
        t = t * mask
        normalizer = mask.sum() * 1.0
    logp = F.log_softmax(logits, dim=1)
    loss = -(t * logp).sum()
    loss = loss / (normalizer.clamp(min=1.0))
    return loss


def durations_kl(w_gt: torch.Tensor, w_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    KL(w_gt || w_pred) over the K_max simplex, averaged over batch.
    w_gt, w_pred: (B,K), each row sums ~1
    """
    w_gt = w_gt.clamp(min=eps)
    w_pred = w_pred.clamp(min=eps)
    kl = (w_gt * (w_gt.log() - w_pred.log())).sum(dim=-1)  # (B,)
    return kl.mean()


def starts_l1(starts_gt: torch.Tensor, starts_pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    L1 difference between starts on the first L true segments.
    starts_*: (B,K), mask: (B,K) 1 for valid gt positions, 0 for pad
    """
    diff = (starts_gt - starts_pred).abs() * mask
    denom = mask.sum().clamp(min=1.0)
    return diff.sum() / denom


def laplacian_regularizer(emb_table: nn.Embedding, L: Optional[torch.Tensor], weight: float) -> torch.Tensor:
    """
    Tr(E^T L E) with E in R^{n_purposes x d}, L positive semidefinite Laplacian.
    """
    if L is None or weight <= 0.0:
        return torch.tensor(0.0, device=emb_table.weight.device)
    E = emb_table.weight  # (n_purposes, d)
    # ignore PAD row if padding_idx is set (typically its embedding is zero)
    return weight * torch.trace(E.t() @ L @ E)


def make_w_gt_from_durations(dur: torch.Tensor, k_max: int) -> torch.Tensor:
    """
    dur: (B,L_true) padded to K with zeros already? We'll expect (B,K) where zeros after L.
    Return normalized weights (sum=1) across K.
    """
    s = dur.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return dur / s
