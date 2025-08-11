"""
Deterministic second-order ODE model that predicts location only, with static context.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torchdiffeq import odeint
from torchsde import sdeint

from ananke_abm.models.mode_sep.config import ModeSepConfig


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class ODEFunc(nn.Module):
    def __init__(self, emb_dim: int, context_dim: int, hidden_dim: int, num_blocks: int):
        super().__init__()
        input_dim = emb_dim + emb_dim + context_dim + 2  # [p, v, h, sin, cos]
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_blocks):
            layers.append(ResidualBlock(hidden_dim))
        layers.append(nn.Linear(hidden_dim, emb_dim))  # outputs acceleration a
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # y: [batch, 2E + H] = [p, v, h]
        # Compute time features
        batch = y.shape[0]
        # Split state
        # This module will be wrapped with context that knows dims
        raise RuntimeError("ODEFunc.forward should be called through WrappedODE which injects dims.")


class WrappedSDE(nn.Module):
    def __init__(self, func: ODEFunc, emb_dim: int, context_dim: int):
        super().__init__()
        self.func = func
        self.emb_dim = emb_dim
        self.context_dim = context_dim

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch = y.shape[0]
        E = self.emb_dim
        H = self.context_dim
        p, v, h = torch.split(y, [E, E, H], dim=-1)
        # Time features
        sin_t = torch.sin(t * 2 * torch.pi / 24.0)
        cos_t = torch.cos(t * 2 * torch.pi / 24.0)
        if sin_t.dim() == 0:
            sin_t = sin_t.expand(batch)
            cos_t = cos_t.expand(batch)
        time_feat = torch.stack([sin_t, cos_t], dim=-1)  # [batch, 2]
        inp = torch.cat([p, v, h, time_feat], dim=-1)
        a = self.func.net(inp)
        dp_dt = v
        dv_dt = a
        dh_dt = torch.zeros_like(h)
        return torch.cat([dp_dt, dv_dt, dh_dt], dim=-1)

    # For torchsde: define drift f and diffusion g
    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.forward(t, y)

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Apply isotropic noise only to [p, v]; keep h deterministic
        B = y.shape[0]
        E = self.emb_dim
        H = self.context_dim
        noise = y.new_zeros(y.shape)
        # Use a scalar noise strength broadcasted across p and v
        # Actual magnitude will be handled in the caller via config
        # Here, set unit noise; scaled later by sde_noise_strength
        noise[:, : 2 * E] = 1.0
        return noise


class ModeSepModel(nn.Module):
    def __init__(self, Z: int, config: ModeSepConfig):
        super().__init__()
        self.config = config
        self.Z = Z
        E = config.emb_dim
        H = config.context_dim

        # Learnable tables
        self.class_table = nn.Parameter(torch.empty(Z, E))
        nn.init.xavier_uniform_(self.class_table)

        self.zone_embed = nn.Embedding(Z, config.zone_emb_dim)

        # Context encoder: raw -> context_dim
        ctx_in = 2 + 2 * config.zone_emb_dim  # [age_norm, income_norm, emb(home), emb(work)]
        self.context_encoder = nn.Sequential(
            nn.Linear(ctx_in, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, H),
        )

        # Drift ODE
        self.odefunc = WrappedSDE(
            ODEFunc(emb_dim=E, context_dim=H, hidden_dim=config.hidden_dim, num_blocks=config.num_res_blocks),
            emb_dim=E,
            context_dim=H,
        )

        # Decoder MLP: p -> predicted embedding
        self.decoder = nn.Sequential(
            nn.Linear(E, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, E),
        )

    def _encode_context(self, traits_raw: torch.Tensor, home_idx: torch.Tensor, work_idx: torch.Tensor) -> torch.Tensor:
        # traits_raw: [batch, 2]
        home_emb = self.zone_embed(home_idx)  # [batch, Ze]
        work_emb = self.zone_embed(work_idx)
        raw = torch.cat([traits_raw, home_emb, work_emb], dim=-1)
        return self.context_encoder(raw)  # [batch, H]

    def _normalize_rows(self, M: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return M / (M.norm(dim=-1, keepdim=True) + eps)

    def forward(
        self,
        times_union: torch.Tensor,              # [T]
        home_idx: torch.Tensor,                 # [B]
        work_idx: torch.Tensor,                 # [B]
        person_traits_raw: torch.Tensor,        # [B, 2]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = times_union.device
        B = home_idx.shape[0]
        E = self.config.emb_dim
        H = self.config.context_dim

        # Initial state
        class_table_detached = self.class_table.detach()
        p0 = class_table_detached[home_idx]         # [B, E]
        v0 = torch.zeros_like(p0)                   # [B, E]
        h = self._encode_context(person_traits_raw, home_idx, work_idx)  # [B, H]
        y0 = torch.cat([p0, v0, h], dim=-1)         # [B, 2E+H]

        # Solve ODE
        if self.config.enable_sde and self.config.sde_noise_strength > 0.0:
            # Scale diffusion by noise strength by wrapping state
            class ScaledSDE(nn.Module):
                def __init__(self, base: WrappedSDE, scale: float):
                    super().__init__()
                    self.base = base
                    self.scale = scale
                    self.noise_type = 'diagonal'
                    self.sde_type = 'ito'

                def f(self, t, y):
                    return self.base.f(t, y)

                def g(self, t, y):
                    g = self.base.g(t, y)
                    return g * self.scale

            sde = ScaledSDE(self.odefunc, self.config.sde_noise_strength)
            y_path = sdeint(
                sde,
                y0,
                times_union,
                method=self.config.sde_method,
                dt=self.config.sde_dt,
            )
        else:
            y_path = odeint(
                self.odefunc,
                y0,
                times_union,
                method=self.config.ode_method,
                rtol=self.config.rtol,
                atol=self.config.atol,
            )  # [T, B, 2E+H]
        y_path = y_path.permute(1, 0, 2)            # [B, T, 2E+H]
        p_t, v_t, _ = torch.split(y_path, [E, E, H], dim=-1)

        # Decode embeddings and compute logits
        pred_emb = self.decoder(p_t)                # [B, T, E]
        table_norm = self._normalize_rows(self.class_table)  # [Z, E]
        emb_norm = pred_emb / (pred_emb.norm(dim=-1, keepdim=True) + 1e-8)
        logits = torch.einsum("bte,ze->btz", emb_norm, table_norm) / self.config.softmax_tau

        return pred_emb, logits, v_t


