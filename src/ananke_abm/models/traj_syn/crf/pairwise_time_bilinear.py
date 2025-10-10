# ananke_abm/models/traj_embed_updated/model/pairwise_time_bilinear.py
import torch
import torch.nn as nn
from ananke_abm.models.traj_embed_updated.model.utils_bases import fourier_time_features

class TimeVaryingPairwise(nn.Module):
    """
    Produces time-varying pairwise transition logits A[t, y_prev, y_curr]
    using a low-rank bilinear parameterization modulated by clock Fourier features.
    """
    def __init__(self, P: int, rank: int = 2, K_clock: int = 6, scale: float = 1.0):
        super().__init__()
        self.P = P
        self.r = rank
        self.K = K_clock
        self.scale = scale

        # U, V: [r, P]
        self.U = nn.Parameter(torch.zeros(rank, P))
        self.V = nn.Parameter(torch.zeros(rank, P))
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)

        # time weights per rank: w_k in R^{2K+1}
        self.W = nn.Parameter(torch.zeros(rank, 2 * K_clock + 1))
        nn.init.xavier_uniform_(self.W)

    def forward(self, t_alloc01: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t_alloc01: [L] normalized allocation positions in [0,1]
        Returns:
            A: [L, P, P] time-varying pairwise logits
        """
        # Fourier on allocation (we want clock timing; mapping to clock is handled by grid construction)
        Phi = fourier_time_features(t_alloc01, self.K)     # [L, 2K+1]
        # s_k(t) = Phi(t) dot W[k]
        S = Phi @ self.W.T                                 # [L, r]
        # For each k, outer product U[k]^T V[k] -> [P,P]
        # Compose A[t] = sum_k S[t,k] * (U_k^T V_k)
        M = torch.einsum("kp,kq->kpq", self.U, self.V)     # [r, P, P]
        A = torch.einsum("lr,rpq->lpq", S, M)              # [L, P, P]
        return self.scale * A
