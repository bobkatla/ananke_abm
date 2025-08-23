import torch
import torch.nn as nn
import torch.nn.functional as F

from ananke_abm.models.traj_embed.model.utils_bases import fourier_time_features

class PurposeDistributionSpace(nn.Module):
    """
    Maps per-purpose distribution features (phi_p) to a learnable embedding e_p.
    Also exposes evaluation of the time prior lambda_p(t) via Fourier coefficients.
    """
    def __init__(self, phi_p_matrix: torch.Tensor, d_p:int=16, hidden:int=64):
        """
        Args:
            phi_p_matrix: (P, D_phi) tensor of standardized per-purpose features.
            d_p: embedding dimension.
            hidden: hidden width.
        """
        super().__init__()
        self.register_buffer("phi_p", phi_p_matrix)   # (P, D_phi)
        P, D_phi = phi_p_matrix.shape
        self.mlp = nn.Sequential(
            nn.Linear(D_phi, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_p)
        )
        self.K_time_prior = None
        self.time_coeff_slice = None

    def set_time_prior_K(self, K_time_prior:int):
        self.K_time_prior = K_time_prior
        self.time_coeff_slice = slice(0, 1+2*K_time_prior)

    def forward(self) -> torch.Tensor:
        """Return e_p: (P, d_p) embedding matrix."""
        return self.mlp(self.phi_p)  # (P, d_p)

    def lambda_log(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate log(lambda_p(t)) for all purposes at continuous t in [0,1].
        Args:
            t: (Q,) or (...,) tensor of times
        Returns:
            loglam: (P, Q) matching t's last dimension, log-density up to normalization
        """
        assert self.K_time_prior is not None, "Call set_time_prior_K first."
        coeffs = self.phi_p[:, self.time_coeff_slice]  # (P, 2K+1)
        # Build Fourier features for t to match (.., 2K+1)
        feat = fourier_time_features(t, self.K_time_prior)  # (..., 2K+1)
        # Compute raw scores: (P, ...,)
        fT = feat.transpose(0, -1)  # (..., C) -> (C, ...)
        raw = torch.matmul(coeffs, fT)  # (P, ...)
        # Ensure positivity then take log (unnormalized): use softplus
        loglam = torch.log(torch.nn.functional.softplus(raw) + 1e-8)
        return loglam  # (P, ...)
