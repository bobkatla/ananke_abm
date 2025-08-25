# ananke_abm/models/traj_embed_updated/model/crf_linear.py
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn


class LinearChainCRF(nn.Module):
    """
    Batched linear-chain CRF with Potts pairwise penalty and optional transition mask.

    Sequence score for a path y (length L, P classes):
      score(y) = sum_t theta[y_t, t] + sum_{t>0} A[y_{t-1}, y_t]
    where:
      - theta: [B, P, L] unaries (log-potentials)
      - A:     [P, P] pairwise (0 on diag, -eta off-diagonal by default)
      - endpoint_mask: [L, P] bool; at t=0 and t=L-1, disallowed states get a large negative added.

    Provides:
      - nll(theta, y, endpoint_mask)  -> scalar mean NLL
      - viterbi(theta, endpoint_mask) -> best paths [B, L] (ints)
    """
    def __init__(
        self,
        P: int,
        eta: float = 4.0,
        learn_eta: bool = False,
        transition_mask: Optional[torch.Tensor] = None,  # [P,P] bool (True=allowed)
        neg_large: float = -1e4,
    ):
        super().__init__()
        self.P = P
        self.neg_large = float(neg_large)

        if learn_eta:
            self.log_eta = nn.Parameter(torch.log(torch.tensor(float(eta), dtype=torch.float32)))
        else:
            self.register_buffer("eta_const", torch.tensor(float(eta), dtype=torch.float32))
            self.log_eta = None

        if transition_mask is not None:
            assert transition_mask.shape == (P, P)
            self.register_buffer("transition_mask", transition_mask.to(torch.bool))
        else:
            self.transition_mask = None

    def _eta(self) -> torch.Tensor:
        if self.log_eta is not None:
            return torch.exp(self.log_eta)
        return self.eta_const

    def _pairwise_matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Build A ∈ R^{P×P}: 0 on diag, -eta off-diagonal, with optional forbidden transitions.
        """
        P = self.P
        eta = self._eta().to(device=device, dtype=dtype)
        A = -eta * torch.ones((P, P), device=device, dtype=dtype)
        A.fill_(-eta)
        A.fill_diagonal_(0.0)
        if self.transition_mask is not None:
            mask = self.transition_mask.to(device=device)
            A = torch.where(mask, A, torch.full_like(A, self.neg_large))  # forbid → large negative
        return A

    @staticmethod
    def _apply_endpoint_mask_inplace(theta: torch.Tensor, endpoint_mask: Optional[torch.Tensor], neg_large: float):
        """
        theta: [B,P,L], endpoint_mask: [L,P] bool (True=allowed) or None.
        Only first/last steps are constrained; we add a large negative to forbidden unaries.
        """
        if endpoint_mask is None:
            return
        assert endpoint_mask.dim() == 2 and endpoint_mask.shape[1] == theta.shape[1], \
            "endpoint_mask must be [L, P]"
        forbid_first = ~endpoint_mask[0]   # [P]
        forbid_last  = ~endpoint_mask[-1]  # [P]
        if forbid_first.any():
            theta[:, forbid_first, 0] += neg_large
        if forbid_last.any():
            theta[:, forbid_last, -1] += neg_large

    def _forward_logZ(self, theta: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Log-partition via forward algorithm in log-space.
        Args:
            theta: [B,P,L], A: [P,P]
        Returns:
            logZ: [B]
        """
        B, P, L = theta.shape
        alpha = theta[:, :, 0]  # [B,P]
        for t in range(1, L):
            # prev → current: alpha_prev[k] + A[k,j]
            prev = alpha.unsqueeze(2) + A.unsqueeze(0)      # [B,P,P]
            alpha = theta[:, :, t] + torch.logsumexp(prev, dim=1)  # [B,P]
        logZ = torch.logsumexp(alpha, dim=1)  # [B]
        return logZ

    @staticmethod
    def _path_score(theta: torch.Tensor, A: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute sequence score for given y.
        Args:
            theta: [B,P,L], A: [P,P], y: [B,L] long
        Returns:
            score: [B]
        """
        B, P, L = theta.shape
        # Unaries
        idx = y.unsqueeze(1)  # [B,1,L]
        unary = theta.gather(dim=1, index=idx).squeeze(1).sum(dim=1)  # [B]

        # Pairwise transitions
        if L > 1:
            y_prev = y[:, :-1]  # [B,L-1]
            y_curr = y[:, 1:]   # [B,L-1]
            trans = A[y_prev, y_curr]  # [B,L-1]
            pair = trans.sum(dim=1)    # [B]
        else:
            pair = torch.zeros(B, device=theta.device, dtype=theta.dtype)

        return unary + pair

    def nll(self, theta: torch.Tensor, y: torch.Tensor, endpoint_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Negative log-likelihood (mean over batch).
        Args:
            theta: [B,P,L] log-unaries
            y:     [B,L] int labels OR [B,P,L] one-hot (will argmax)
            endpoint_mask: [L,P] bool (optional)
        """
        if y.dim() == 3:
            y = y.argmax(dim=1)
        assert y.shape == (theta.shape[0], theta.shape[2])

        # Work on a copy to avoid surprising in-place modifications outside
        theta_l = theta.clone()
        self._apply_endpoint_mask_inplace(theta_l, endpoint_mask, self.neg_large)
        A = self._pairwise_matrix(device=theta.device, dtype=theta.dtype)

        logZ = self._forward_logZ(theta_l, A)             # [B]
        score = self._path_score(theta_l, A, y)           # [B]
        nll = (logZ - score).mean()
        return nll

    def viterbi(self, theta: torch.Tensor, endpoint_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        MAP sequence via Viterbi.
        Args:
            theta: [B,P,L]
            endpoint_mask: [L,P] bool (optional)
        Returns:
            y_hat: [B,L] long
        """
        B, P, L = theta.shape
        theta_l = theta.clone()
        self._apply_endpoint_mask_inplace(theta_l, endpoint_mask, self.neg_large)
        A = self._pairwise_matrix(device=theta.device, dtype=theta.dtype)

        delta = theta_l[:, :, 0]                    # [B,P]
        backp = torch.full((B, P, L), -1, dtype=torch.long, device=theta.device)

        for t in range(1, L):
            prev = delta.unsqueeze(2) + A.unsqueeze(0)    # [B,P,P]
            best_prev_val, best_prev_idx = prev.max(dim=1)  # [B,P]
            delta = best_prev_val + theta_l[:, :, t]        # [B,P]
            backp[:, :, t] = best_prev_idx

        # backtrace
        y_hat = torch.zeros((B, L), dtype=torch.long, device=theta.device)
        last = delta.argmax(dim=1)  # [B]
        y_hat[:, -1] = last

        for t in range(L - 1, 0, -1):
            last = backp[torch.arange(B, device=theta.device), last, t]
            y_hat[:, t - 1] = last

        return y_hat
