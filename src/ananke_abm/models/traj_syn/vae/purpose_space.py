import torch
import torch.nn as nn
import torch.nn.functional as F

from ananke_abm.models.traj_syn.core.utils_bases import fourier_time_features


class PurposeDistributionSpace(nn.Module):
    """
    Maps per-purpose features (phi_p) to a learnable embedding e_p, and exposes
    a 24h-periodic time-of-day prior λ_p(clock) evaluated on arbitrary inputs.

    Conventions:
      - The prior is defined on CLOCK time (24h), not on the 30h allocation axis.
      - To evaluate the prior on the allocation grid (e.g., 30h), map minutes to
        clock minutes via modulo 1440 and call `lambda_log_on_alloc_grid`.
    """
    def __init__(self, phi_p_matrix: torch.Tensor, d_p: int = 16, hidden: int = 64):
        """
        Args:
            phi_p_matrix: (P, D_phi) standardized per-purpose features.
                          The first (2*K_clock_prior+1) entries are the Fourier
                          coeffs for the CLOCK prior (DC, cos1..cosK, sin1..sinK).
            d_p: purpose embedding dimension.
            hidden: hidden width for the MLP.
        """
        super().__init__()
        self.register_buffer("phi_p", phi_p_matrix)  # (P, D_phi)
        P, D_phi = phi_p_matrix.shape
        self.mlp = nn.Sequential(
            nn.Linear(D_phi, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_p),
        )
        self.K_clock_prior: int | None = None
        self.time_coeff_slice: slice | None = None

    # ---- Prior configuration ----
    def set_clock_prior_K(self, K_clock_prior: int):
        """Declare how many harmonic pairs were used to fit λ_p(clock)."""
        self.K_clock_prior = K_clock_prior
        self.time_coeff_slice = slice(0, 1 + 2 * K_clock_prior)

    # ---- Embedding ----
    def forward(self) -> torch.Tensor:
        """Return e_p: (P, d_p) embedding matrix."""
        return self.mlp(self.phi_p)

    # ---- Prior evaluation (preferred APIs) ----
    @torch.no_grad()
    def lambda_log_clock(self, t_clock01: torch.Tensor) -> torch.Tensor:
        """
        Evaluate log λ_p(clock) on CLOCK-normalized inputs in [0,1].

        Args:
            t_clock01: (...,) tensor with values in [0,1] mapping 0..24h.
        Returns:
            log λ (unnormalized): (P, ...), matching the broadcast shape of t_clock01.
        """
        assert self.K_clock_prior is not None, "Call set_clock_prior_K(...) first."
        coeffs = self.phi_p[:, self.time_coeff_slice]  # (P, 2K+1)
        feat = fourier_time_features(t_clock01, self.K_clock_prior)  # (..., 2K+1)
        # tensordot over basis dim → (P, ...)
        raw = torch.tensordot(coeffs, feat, dims=([1], [-1]))
        # Softplus to ensure positivity, then log (unnormalized)
        return torch.log(F.softplus(raw) + 1e-8)

    @torch.no_grad()
    def lambda_log_on_alloc_grid(self, t_alloc_minutes: torch.Tensor, T_clock_minutes: int) -> torch.Tensor:
        """
        Evaluate log λ_p on an ALLOCATION grid (e.g., 30h minutes) by mapping each
        minute to its CLOCK counterpart via modulo 24h.

        Args:
            t_alloc_minutes: (L,) tensor of minute indices over allocation window.
            T_clock_minutes: int, usually 1440.
        Returns:
            log λ: (P, L)
        """
        t_clock01 = (t_alloc_minutes.to(dtype=self.phi_p.dtype) % float(T_clock_minutes)) / float(T_clock_minutes)
        return self.lambda_log_clock(t_clock01)

