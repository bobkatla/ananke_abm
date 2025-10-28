import torch


def viterbi_decode(unary_logits_btp, A_pp):
    """
    unary_logits_btp: (B, T, P) float tensor
        Per-time per-class scores (unaries).
    A_pp: (P, P) float tensor
        Transition score matrix, where A[p_prev, p_next].

    Returns:
        paths_bt: (B, T) int64 tensor
        The highest-scoring sequence under:
        sum_t unary_logits[t, y_t] + sum_t A[y_{t-1}, y_t]
    """
    B, T, P = unary_logits_btp.shape
    device = unary_logits_btp.device

    # dp_t[p] = best score of any path ending in state p at time t
    dp = unary_logits_btp[:, 0, :]            # (B, P)
    backpointers = []

    for t in range(1, T):
        # dp_prev[:, p_prev] + A[p_prev, p_next] -> score for landing in p_next
        # dp_prev: (B,P); A: (P,P)
        # score_next: (B, P_next, P_prev) then max over prev
        dp_expanded = dp.unsqueeze(2)                     # (B, P, 1)
        A_expanded = A_pp.unsqueeze(0)                    # (1, P, P)
        scores = dp_expanded + A_expanded                 # (B, P, P)
        # We want max over prev-state dim=1 to get best path into each next-state
        best_scores, best_prev = scores.max(dim=1)        # (B, P), (B, P)
        # add current unary
        dp = best_scores + unary_logits_btp[:, t, :]      # (B, P)

        backpointers.append(best_prev)                    # list of (B, P)

    # now backtrack
    paths_bt = torch.zeros(B, T, dtype=torch.long, device=device)
    last_states = torch.argmax(dp, dim=1)                 # (B,)
    paths_bt[:, T-1] = last_states

    # go backward in time
    for rev_t, bp in enumerate(reversed(backpointers), start=1):
        t = T - 1 - rev_t
        # bp: (B, P_next) tells best prev-state for each next-state
        next_states = paths_bt[:, t+1]                    # (B,)
        prev_states = bp[torch.arange(B), next_states]    # (B,)
        paths_bt[:, t] = prev_states

    return paths_bt


def crf_log_partition(unary_logits_btp, A_pp):
    """
    Compute log Z(U, A) via forward algorithm in log-space.

    unary_logits_btp: (B,T,P)
    A_pp: (P,P)

    Returns:
        logZ_b: (B,) log-partition for each sequence
    """
    B, T, P = unary_logits_btp.shape
    device = unary_logits_btp.device

    # alpha_t[p] = log-sum-exp of scores of all paths ending in p at time t
    alpha = unary_logits_btp[:, 0, :]  # (B,P)

    for t in range(1, T):
        # alpha_prev[:, p_prev] + A[p_prev,p_next] => scores to p_next
        alpha_exp = alpha.unsqueeze(2)              # (B,P,1)
        A_exp = A_pp.unsqueeze(0)                   # (1,P,P)
        scores = alpha_exp + A_exp                  # (B,P,P)

        # log-sum-exp over prev-state dimension (dim=1)
        # result is log-sum over all prev states for each next-state
        alpha = torch.logsumexp(scores, dim=1)      # (B,P)

        # add unary for current time
        alpha = alpha + unary_logits_btp[:, t, :]   # (B,P)

    # finally log-sum-exp over last state
    logZ_b = torch.logsumexp(alpha, dim=1)          # (B,)
    return logZ_b


def crf_path_score(unary_logits_btp, labels_bt, A_pp):
    """
    Score of the gold path y under the CRF:
    s(y) = sum_t U[t, y_t] + sum_{t>0} A[y_{t-1}, y_t]

    unary_logits_btp: (B,T,P)
    labels_bt:        (B,T) long
    A_pp:             (P,P)

    Returns:
        score_b: (B,) float
    """
    B, T, P = unary_logits_btp.shape
    # gather unary terms
    unary_score = unary_logits_btp.gather(
        dim=2,
        index=labels_bt.unsqueeze(-1)
    ).squeeze(-1).sum(dim=1)  # (B,)

    # pairwise terms
    y_prev = labels_bt[:, :-1]  # (B,T-1)
    y_next = labels_bt[:, 1:]   # (B,T-1)
    pair_scores = A_pp[y_prev, y_next]  # (B,T-1)
    pair_score = pair_scores.sum(dim=1) # (B,)

    return unary_score + pair_score


def crf_nll_batch(unary_logits_btp, labels_bt, A_pp):
    """
    Negative log-likelihood for a batch.

    NLL = logZ - score_gold

    Returns scalar mean NLL over batch.
    """
    logZ_b = crf_log_partition(unary_logits_btp, A_pp)      # (B,)
    gold_score_b = crf_path_score(unary_logits_btp, labels_bt, A_pp)  # (B,)
    nll_b = logZ_b - gold_score_b                           # (B,)
    return nll_b.mean()
