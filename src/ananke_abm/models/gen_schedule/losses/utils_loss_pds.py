import torch.nn.functional as F

def loss_time_of_day_marginal(logits_btP, m_tod_emp_PT):
    """
    logits_btP: (B,T,P)
    m_tod_emp_PT: (P,T) empirical marginal Pr(purpose p at time t)

    Returns scalar L_tod.

    We'll:
      probs_btP = softmax(logits, dim=-1)     (B,T,P)
      batch_mean_tP = probs_btP.mean(dim=0)   (T,P)
      batch_mean_PT = batch_mean_tP.permute(1,0) (P,T)
      mse over (P,T)
    """
    probs_btP = F.softmax(logits_btP, dim=-1)            # (B,T,P)
    batch_mean_tP = probs_btP.mean(dim=0)                # (T,P)
    batch_mean_PT = batch_mean_tP.permute(1,0)           # (P,T)

    diff = batch_mean_PT - m_tod_emp_PT  # (P,T)
    mse = (diff * diff).mean()
    return mse


def loss_presence_rate(logits_btP, presence_emp_P):
    """
    logits_btP: (B,T,P)
    presence_emp_P: (P,)

    Returns scalar L_presence.

    We approximate "did purpose p occur at least once in this person's day"
    with:
      present_bp = 1 - prod_t (1 - probs_btP[b,t,p])
    Then average over batch and MSE to empirical.
    """
    probs_btP = F.softmax(logits_btP, dim=-1)            # (B,T,P)
    probs_not = 1.0 - probs_btP                          # (B,T,P)
    prod_not = probs_not.prod(dim=1)                     # (B,P)
    present_bp = 1.0 - prod_not                          # (B,P)
    batch_presence_P = present_bp.mean(dim=0)            # (P,)

    diff = batch_presence_P - presence_emp_P             # (P,)
    mse = (diff * diff).mean()
    return mse
