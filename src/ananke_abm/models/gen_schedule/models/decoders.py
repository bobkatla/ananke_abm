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
