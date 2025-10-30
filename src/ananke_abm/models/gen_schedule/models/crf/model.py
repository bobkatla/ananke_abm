import torch
import torch.nn as nn
from ananke_abm.models.gen_schedule.models.crf.linear_chain import crf_nll_batch, viterbi_decode

class TransitionCRF(nn.Module):
    """
    Linear-chain CRF with a full transition matrix A[p_prev, p_next].
    Supports constrained decoding to avoid 'all-Home' paths.
    """
    def __init__(self, num_purposes, init_scale=0.01, home_idx=None, use_bias=True):
        super().__init__()
        self.A = nn.Parameter(torch.empty(num_purposes, num_purposes))
        nn.init.uniform_(self.A, -init_scale, init_scale)
        self.home_idx = home_idx
        self.bias = nn.Parameter(torch.zeros(num_purposes)) if use_bias else None

    def nll(self, unary_logits_btp, labels_bt):
        return crf_nll_batch(unary_logits_btp, labels_bt, self.A)

    @torch.no_grad()
    def decode(self, unary_logits_btp, enforce_nonhome=False):
        """
        unary_logits_btp: (B,T,P)
        returns paths_bt: (B,T)
        """
        if self.bias is not None:
            unary_logits_btp = unary_logits_btp + self.bias.view(1,1,-1)

        if enforce_nonhome:
            assert self.home_idx is not None, "home_idx must be set to enforce non-home constraint"
            return self._decode_constrained_no_all_home(unary_logits_btp)
        else:
            return viterbi_decode(unary_logits_btp, self.A)

    @torch.no_grad()
    def _decode_constrained_no_all_home(self, U_btp):
        """
        Constrained Viterbi: ensures at least one non-Home state is visited.
        """
        B, T, P = U_btp.shape
        A = self.A
        home = self.home_idx
        NEG_INF = torch.finfo(U_btp.dtype).min / 4

        paths_bt = torch.zeros(B, T, dtype=torch.long, device=U_btp.device)

        for b in range(B):
            U = U_btp[b]  # (T,P)
            delta = torch.full((T, P, 2), NEG_INF, device=U.device, dtype=U.dtype)
            psi_p = torch.full((T, P, 2), -1, dtype=torch.long, device=U.device)
            psi_v = torch.full((T, P, 2), -1, dtype=torch.long, device=U.device)

            # init (t=0)
            delta[0, home, 0] = U[0, home]
            psi_p[0, home, 0] = -1
            psi_v[0, home, 0] = -1

            nonhome = torch.arange(P, device=U.device)
            nonhome = nonhome[nonhome != home]
            delta[0, nonhome, 1] = U[0, nonhome]

            # forward
            for t in range(1, T):
                prev0 = delta[t-1, :, 0].view(P, 1) + A
                prev1 = delta[t-1, :, 1].view(P, 1) + A

                # case v' = 0 (still no nonhome): only if staying at Home
                best_prev = prev0.argmax(dim=0)
                best_val = prev0.max(dim=0).values
                delta[t, home, 0] = best_val[home] + U[t, home]
                psi_p[t, home, 0] = best_prev[home]
                psi_v[t, home, 0] = 0

                # case v' = 1
                # p==Home: only from v_prev==1
                best_prev = prev1.argmax(dim=0)
                best_val = prev1.max(dim=0).values
                delta[t, home, 1] = best_val[home] + U[t, home]
                psi_p[t, home, 1] = best_prev[home]
                psi_v[t, home, 1] = 1

                # p!=Home: from either v_prev==0 or v_prev==1
                if P > 1:
                    nh = nonhome
                    best_q_from0 = prev0[:, nh].argmax(dim=0)
                    val_from0 = prev0[:, nh].max(dim=0).values
                    best_q_from1 = prev1[:, nh].argmax(dim=0)
                    val_from1 = prev1[:, nh].max(dim=0).values
                    use_from1 = val_from1 > val_from0
                    best_val = torch.where(use_from1, val_from1, val_from0)
                    best_q = torch.where(use_from1, best_q_from1, best_q_from0)
                    best_v = torch.where(use_from1,
                                         torch.ones_like(best_q_from1),
                                         torch.zeros_like(best_q_from0))
                    delta[t, nh, 1] = best_val + U[t, nh]
                    psi_p[t, nh, 1] = best_q
                    psi_v[t, nh, 1] = best_v

            # termination: must end with v=1
            last_p = delta[T-1, :, 1].argmax()
            y = torch.empty(T, dtype=torch.long, device=U.device)
            y[T-1] = last_p
            v = 1
            for t in range(T-1, 0, -1):
                prev_p = psi_p[t, y[t], v]
                prev_v = psi_v[t, y[t], v]
                y[t-1] = prev_p
                v = prev_v.item()
            paths_bt[b] = y

        return paths_bt
