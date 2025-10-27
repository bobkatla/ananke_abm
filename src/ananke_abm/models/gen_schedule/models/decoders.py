import torch
import torch.nn as nn


class ScheduleDecoderIndependent(nn.Module):
    """
    Decoder that maps latent z -> per-time, per-purpose logits.

    We assume the same structure as the Phase 1 decoder:
    - A learned global time basis of shape (T, H)
    - A latent-conditioned factor of shape (B, P, H)
    - Combine to produce logits (B, T, P)

    Args:
        L: number of time bins
        P: number of purposes/classes
        z_dim: latent dim
        emb_dim: internal width / factor dim H
    """

    def __init__(self, L, P, z_dim, emb_dim):
        super().__init__()
        self.L = L
        self.P = P
        self.z_dim = z_dim
        self.emb_dim = emb_dim

        # time basis: (L, H)
        self.time_basis = nn.Parameter(torch.randn(L, emb_dim) * 0.01)

        # map z -> per-purpose factors (B, P, H)
        self.latent_to_factor = nn.Sequential(
            nn.Linear(z_dim, emb_dim * P),
        )

        # per-purpose bias
        self.bias = nn.Parameter(torch.zeros(P))

    def forward(self, z):
        """
        z: (B, z_dim)
        returns logits: (B, L, P)
        """
        B = z.shape[0]

        # latent factors
        latent_factors = self.latent_to_factor(z)  # (B, P*H)
        latent_factors = latent_factors.view(B, self.P, self.emb_dim)  # (B,P,H)

        # combine time basis (L,H) with per-purpose factors (B,P,H)
        # we want logits[b,t,p] = <time_basis[t,:], latent_factors[b,p,:]> + bias[p]
        # => einsum over H
        logits = torch.einsum("th,bph->btp", self.time_basis, latent_factors)  # (B,L,P)

        logits = logits + self.bias.view(1, 1, self.P)
        return logits


class ScheduleDecoderWithPDS(nn.Module):
    """
    Decoder that combines:
    (1) latent -> per-time/per-purpose logits (same idea as baseline decoder), and
    (2) fixed PDS features per (purpose, time) -> learned additive bias.

    Shapes:
      B = batch
      T = num_time_bins
      P = num_purposes
      D_lat = decoder hidden dim from latent pathway
      D_pds = num features per (p,t) from PDS (e.g. 2: m_tod, start_rate)

    Forward:
      logits = latent_logits + pds_bias
      return logits [B, T, P]
    """

    def __init__(
        self,
        num_time_bins: int,
        num_purposes: int,
        z_dim: int,
        emb_dim: int,
        pds_features: torch.Tensor,
    ):
        """
        pds_features: tensor [P, T, D_pds], fixed (registered as buffer)
        """
        super().__init__()
        self.T = num_time_bins
        self.P = num_purposes
        self.z_dim = z_dim
        self.emb_dim = emb_dim

        # ---- latent -> time-purpose logits (baseline path) ----
        #
        # We'll do a low-rank factorization:
        #   h_z = MLP(z) -> [B, P, H]
        #   time_basis = learned [T, H]
        #   latent_logits[b,t,p] = (h_z[b,p,:] · time_basis[t,:]) + bias[p]
        #
        # This matches the "per-purpose factor times global time basis" idea.

        self.latent_to_purpose = nn.Sequential(
            nn.Linear(z_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, self.P * emb_dim),
        )
        # We'll reshape to (B, P, emb_dim)

        self.time_basis = nn.Parameter(
            torch.zeros(self.T, emb_dim)
        )
        nn.init.xavier_normal_(self.time_basis)

        self.latent_bias = nn.Parameter(torch.zeros(self.P))

        # ---- PDS pathway ----
        # pds_features is fixed [P, T, D_pds]
        # We'll learn a linear projection W_pds: D_pds -> 1 (per-purpose or shared).
        #
        # Simplest, shared linear:
        #   pds_bias[t,p] = W_pds( pds_features[p,t,:] ) + b_pds[p]
        #
        # This gives [T,P].

        self.register_buffer(
            "pds_features",
            pds_features  # float32 tensor [P,T,D_pds]
        )
        D_pds = pds_features.shape[-1]

        self.pds_linear = nn.Linear(D_pds, 1, bias=False)
        self.pds_bias = nn.Parameter(torch.zeros(self.P))

        # note: we do NOT time-condition bias yet; pds_linear + pds_bias[p] is enough for now.

    def forward(self, z):
        """
        z: [B, z_dim]
        returns logits: [B, T, P]
        """

        B = z.shape[0]
        device = z.device
        T = self.T
        P = self.P
        H = self.time_basis.shape[1]

        # ---- latent pathway ----
        # latent_to_purpose(z): [B, P*H] -> reshape -> [B, P, H]
        per_purpose_factors = self.latent_to_purpose(z)
        per_purpose_factors = per_purpose_factors.view(B, P, H)

        # time_basis: [T,H]
        # We want latent_logits[b,t,p] = dot( per_purpose_factors[b,p,:], time_basis[t,:] ) + latent_bias[p]

        # (B,P,H) @ (H,T) -> (B,P,T), then swap -> (B,T,P)
        # We'll do batch matmul by reshaping:
        #   (B*P,H) x (H,T) -> (B*P,T) -> reshape (B,P,T) -> permute
        bp = per_purpose_factors.reshape(B * P, H)          # (B*P,H)
        tb = self.time_basis.t()                            # (H,T)
        latent_logits_bpT = bp @ tb                         # (B*P, T)
        latent_logits = latent_logits_bpT.view(B, P, T)     # (B,P,T)
        latent_logits = latent_logits.permute(0, 2, 1)      # (B,T,P)
        latent_logits = latent_logits + self.latent_bias.view(1, 1, P)

        # ---- PDS pathway ----
        # pds_features: [P,T,D_pds] -> apply linear -> [P,T,1] -> squeeze -> [P,T]
        # then transpose to [T,P], then add pds_bias[p].
        pds_feat = self.pds_features.to(device=device)          # [P,T,D_pds]
        pds_score = self.pds_linear(pds_feat)                   # [P,T,1]
        pds_score = pds_score.squeeze(-1)                       # [P,T]
        pds_score = pds_score + self.pds_bias.view(P, 1)        # add per-purpose bias
        pds_score = pds_score.permute(1, 0)                     # [T,P]

        # broadcast to batch
        pds_logits = pds_score.unsqueeze(0).expand(B, T, P)     # [B,T,P]

        # ---- combine ----
        logits = latent_logits + pds_logits                     # [B,T,P]

        return logits
