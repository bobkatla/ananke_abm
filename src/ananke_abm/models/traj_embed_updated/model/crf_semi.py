# ananke_abm/models/traj_embed_updated/model/crf_semi.py
from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import math
import torch
import torch.nn as nn


def _make_potts_A(P: int, eta: torch.Tensor, device, dtype):
    A = -eta * torch.ones(P, P, device=device, dtype=dtype)
    A.fill_diagonal_(0.0)
    return A  # [P,P]


def build_duration_logprob_table(
    priors: Dict[str, object],
    purposes: List[str],
    step_minutes: int,
    T_alloc_minutes: int,
    Dmax_minutes: int,
    device,
    dtype=torch.float32,
) -> torch.Tensor:
    """
    Returns log p_dur[p, d] for d=1..Dmax_bins, shape [P, Dmax_bins] on the *allocation* grid.
    Uses pr.dur_mu_log / pr.dur_sigma_log when available (log-normal on normalized duration).
    Falls back to Gaussian on normalized duration (mu_d, sigma_d).
    """
    P = len(purposes)
    Dmax_bins = max(1, int(math.ceil(Dmax_minutes / step_minutes)))
    xs = torch.arange(1, Dmax_bins + 1, device=device, dtype=dtype)
    mins = xs * float(step_minutes)
    x_norm = torch.clamp(mins / float(T_alloc_minutes), min=1e-6, max=1.0)  # (0,1]

    table = torch.empty(P, Dmax_bins, dtype=dtype, device=device)
    for i, p in enumerate(purposes):
        pr = priors[p]
        if hasattr(pr, "dur_mu_log") and hasattr(pr, "dur_sigma_log"):
            mu_log = float(pr.dur_mu_log); sd_log = float(pr.dur_sigma_log) + 1e-12
            logpdf = -(torch.log(x_norm) - mu_log) ** 2 / (2 * (sd_log**2)) \
                     - torch.log(x_norm + 1e-12) - math.log(sd_log) - 0.5*math.log(2*math.pi)
        else:
            mu = float(getattr(pr, "mu_d", 0.1)); sd = float(getattr(pr, "sigma_d", 0.1)) + 1e-12
            logpdf = - (x_norm - mu) ** 2 / (2 * (sd**2)) - math.log(sd) - 0.5*math.log(2*math.pi)
        # normalize across d (keeps numbers stable)
        logpdf = logpdf - torch.logsumexp(logpdf, dim=-1, keepdim=True)
        table[i] = logpdf
    return table  # [P, Dmax_bins]


def _seg_sum_from_cumsum(Uc: torch.Tensor, s: int, t: int) -> torch.Tensor:
    """Unary sum over [s..t] using cumsum over last dim. Returns [B,P]."""
    return Uc[..., t] if s == 0 else (Uc[..., t] - Uc[..., s - 1])


class SemiMarkovCRF(nn.Module):
    """
    Segmental (semi-Markov) CRF with Potts transition and per-purpose duration prior.
    API mirrors LinearChainCRF:
      - nll(theta, gold_segments, endpoint_mask) -> scalar mean NLL
      - viterbi(theta, endpoint_mask) -> List[List[(p,s,d)]]
    Inputs:
      theta: [B,P,L] emission scores on the *allocation* grid (sum over segment bins).
      gold_segments: per-batch list of (p, s, d) in *bin indices* (NOT normalized).
      endpoint_mask: [L,P] bool; a segment (p,s,d) is valid iff mask[s,p] and mask[s+d-1,p] are True.
    """
    def __init__(self, P: int, eta: float = 4.0, learn_eta: bool = False):
        super().__init__()
        self.P = P
        if learn_eta:
            self.log_eta = nn.Parameter(torch.log(torch.tensor(float(eta), dtype=torch.float32)))
        else:
            self.register_buffer("eta_const", torch.tensor(float(eta), dtype=torch.float32))
            self.log_eta = None

    def _eta(self, device, dtype):
        return torch.exp(self.log_eta).to(device=device, dtype=dtype) if self.log_eta is not None else self.eta_const.to(device=device, dtype=dtype)

    @staticmethod
    def _check_inputs(theta: torch.Tensor, endpoint_mask: Optional[torch.Tensor]):
        assert theta.dim() == 3, "theta must be [B,P,L]"
        if endpoint_mask is not None:
            assert endpoint_mask.shape == (theta.shape[2], theta.shape[1]), "endpoint_mask must be [L,P]"

    def _logZ(self, theta: torch.Tensor, dur_logprob: torch.Tensor, endpoint_mask: torch.Tensor, Dmax_bins: int) -> torch.Tensor:
        """
        Forward DP in log-space, complexity O(B·L·(P·Dmax + P^2)).
        """
        self._check_inputs(theta, endpoint_mask)
        B, P, L = theta.shape
        device, dtype = theta.device, theta.dtype
        eta = self._eta(device, dtype)
        A = _make_potts_A(P, eta, device, dtype)                        # [P,P]
        allow = endpoint_mask.to(device=device, dtype=torch.bool)       # [L,P]
        neg_inf = torch.finfo(dtype).min / 4

        Uc = theta.cumsum(dim=-1)                                       # [B,P,L]
        alpha_at_t = []

        for t in range(L):
            alpha_t = torch.full((B, P), neg_inf, device=device, dtype=dtype)
            allow_end = allow[t]                                        # [P]
            Dmax_here = min(Dmax_bins, t + 1)
            for d in range(1, Dmax_here + 1):
                s = t - d + 1
                allow_start = allow[s]
                seg_sum = _seg_sum_from_cumsum(Uc, s, t)                # [B,P]
                dur_lp  = dur_logprob[:, d - 1].view(1, P)              # [1,P]
                if s == 0:
                    prev = torch.zeros(B, P, device=device, dtype=dtype)
                else:
                    prev_alpha = alpha_at_t[s - 1]                      # [B,P]
                    prev = torch.logsumexp(prev_alpha.unsqueeze(2) + A.unsqueeze(0), dim=1)  # [B,P]
                cand = prev + seg_sum + dur_lp                          # [B,P]
                valid = (allow_start & allow_end).view(1, P).expand_as(cand)
                cand = torch.where(valid, cand, torch.full_like(cand, neg_inf))
                alpha_t = torch.logsumexp(torch.stack([alpha_t, cand], dim=0), dim=0)
            alpha_at_t.append(alpha_t)

        return torch.logsumexp(alpha_at_t[-1], dim=1)                   # [B]

    @staticmethod
    def _score_path(theta: torch.Tensor, segments_b: List[List[Tuple[int,int,int]]], eta: torch.Tensor) -> torch.Tensor:
        """
        Score gold segmentation(s) with unary sums + Potts transitions; NO duration prior here.
        Returns [B].
        """
        B, P, L = theta.shape
        device, dtype = theta.device, theta.dtype
        A = _make_potts_A(P, eta, device, dtype)
        Uc = theta.cumsum(dim=-1)
        out = theta.new_zeros(B)
        for b, segs in enumerate(segments_b):
            s = theta.new_tensor(0.0)
            prev_p = None
            for (p, start, dur) in segs:
                end = start + dur - 1
                s += _seg_sum_from_cumsum(Uc[b:b+1], start, end)[0, p]
                if prev_p is not None:
                    s += A[prev_p, p]
                prev_p = p
            out[b] = s
        return out

    @torch.no_grad()
    def viterbi(self, theta: torch.Tensor, dur_logprob: torch.Tensor, endpoint_mask: torch.Tensor, Dmax_bins: int) -> List[List[Tuple[int,int,int]]]:
        """
        MAP segmentation. Returns per-batch list of (p, s, d) in *bin indices*.
        """
        self._check_inputs(theta, endpoint_mask)
        B, P, L = theta.shape
        device, dtype = theta.device, theta.dtype
        eta = self._eta(device, dtype)
        A = _make_potts_A(P, eta, device, dtype)
        allow = endpoint_mask.to(device=device, dtype=torch.bool)
        neg_inf = torch.finfo(dtype).min / 4
        Uc = theta.cumsum(dim=-1)

        delta = torch.full((L, B, P), neg_inf, device=device, dtype=dtype)
        bp_k  = torch.full((L, B, P), -1,    device=device, dtype=torch.long)
        bp_d  = torch.full((L, B, P),  0,    device=device, dtype=torch.long)

        for t in range(L):
            allow_end = allow[t]                                        # [P]
            Dmax_here = min(Dmax_bins, t + 1)
            best = torch.full((B, P), neg_inf, device=device, dtype=dtype)
            best_k = torch.full((B, P), -1,    device=device, dtype=torch.long)
            best_d = torch.zeros((B, P),       device=device, dtype=torch.long)
            for d in range(1, Dmax_here + 1):
                s = t - d + 1
                allow_start = allow[s]
                seg_sum = _seg_sum_from_cumsum(Uc, s, t)                # [B,P]
                dur_lp  = dur_logprob[:, d - 1].view(1, P)              # [1,P]
                if s == 0:
                    prev_val = torch.zeros(B, P, device=device, dtype=dtype)
                    prev_idx = torch.full((B, P), -1, device=device, dtype=torch.long)
                else:
                    prev = delta[s - 1]
                    tmp  = prev.unsqueeze(2) + A.unsqueeze(0)           # [B,P,P]
                    prev_val, prev_idx = tmp.max(dim=1)                 # [B,P], [B,P]
                cand = prev_val + seg_sum + dur_lp
                valid = (allow_start & allow_end).view(1, P).expand_as(cand)
                cand = torch.where(valid, cand, torch.full_like(cand, neg_inf))
                take = cand > best
                best  = torch.where(take, cand, best)
                best_k = torch.where(take, prev_idx, best_k)
                best_d = torch.where(take, torch.tensor(d, device=device, dtype=torch.long), best_d)
            delta[t] = best; bp_k[t] = best_k; bp_d[t] = best_d

        out: List[List[Tuple[int,int,int]]] = []
        last_score, last_p = delta[-1].max(dim=1)                       # [B]
        for b in range(B):
            t = L - 1
            p = int(last_p[b].item())
            segs: List[Tuple[int,int,int]] = []
            while t >= 0 and p >= 0:
                d = int(bp_d[t, b, p].item())
                if d <= 0: break
                s = t - d + 1
                segs.append((p, s, d))
                prev_p = int(bp_k[t, b, p].item())
                t = s - 1
                p = prev_p
            out.append(segs[::-1])
        return out

    def nll(
        self,
        theta: torch.Tensor,                                 # [B,P,L]
        gold_segments: List[List[Tuple[int,int,int]]],       # per-batch segments (p,s,d) in bins
        dur_logprob: torch.Tensor,                           # [P, Dmax_bins]
        endpoint_mask: torch.Tensor,                         # [L,P] bool
    ) -> torch.Tensor:
        """Mean NLL = E[logZ - score(gold)]; score(gold) adds duration prior."""
        B, P, L = theta.shape
        device, dtype = theta.device, theta.dtype
        Dmax_bins = dur_logprob.shape[1]
        eta = self._eta(device, dtype)

        logZ = self._logZ(theta, dur_logprob, endpoint_mask, Dmax_bins)             # [B]
        score_unary_trans = self._score_path(theta, gold_segments, eta=eta)         # [B]

        # duration prior for gold
        score_dur = theta.new_zeros(B)
        for b, segs in enumerate(gold_segments):
            s = 0.0
            for (p, _start, d) in segs:
                d_idx = min(d, Dmax_bins) - 1
                s += float(dur_logprob[p, d_idx].item())
            score_dur[b] = s

        return (logZ - (score_unary_trans + score_dur)).mean()
