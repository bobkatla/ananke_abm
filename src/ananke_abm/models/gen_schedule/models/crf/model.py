import torch
import torch.nn as nn
from ananke_abm.models.gen_schedule.models.crf.linear_chain import crf_nll_batch, viterbi_decode


class TransitionCRF(nn.Module):
    """
    Linear-chain CRF with a full transition matrix A[p_prev, p_next].
    No start/end bias terms for now; simplest version.
    """

    def __init__(self, num_purposes, init_scale=0.01):
        super().__init__()
        self.A = nn.Parameter(torch.empty(num_purposes, num_purposes))
        nn.init.uniform_(self.A, -init_scale, init_scale)

    def nll(self, unary_logits_btp, labels_bt):
        """
        unary_logits_btp: (B,T,P)
        labels_bt:        (B,T) long
        returns scalar mean NLL
        """
        return crf_nll_batch(unary_logits_btp, labels_bt, self.A)

    @torch.no_grad()
    def decode(self, unary_logits_btp):
        """
        unary_logits_btp: (B,T,P)
        returns paths_bt: (B,T) long
        """
        return viterbi_decode(unary_logits_btp, self.A)
