import torch
import torch.nn.functional as F

def masked_ce_mse(logits_act, logits_dur, acts, durs, mask, SOS_id=0, EOS_id=1):
    """
    Alignments:
      - logits_* are for steps t=1..L-1 (predictions for positions 1..L-1)
      - acts,durs,mask are length L; we compare targets at positions 1..L-1
      - mask excludes specials (SOS/EOS)
    """
    B, L = acts.shape
    target_act = acts[:,1:]                 # (B,L-1)
    target_dur = durs[:,1:]                 # (B,L-1)
    target_msk = mask[:,1:]                 # (B,L-1) boolean

    # activity CE
    ce = F.cross_entropy(
        logits_act[target_msk], target_act[target_msk], reduction="mean"
    ) if target_msk.any() else logits_act.sum()*0.0

    # duration MSE (sigmoid is applied inside model at inference; here we regress logits_dur directly via sigmoid+MSE)
    pred_dur = torch.sigmoid(logits_dur)
    mse = F.mse_loss(
        pred_dur[target_msk], target_dur[target_msk], reduction="mean"
    ) if target_msk.any() else pred_dur.sum()*0.0

    return ce, mse

def kl_normal(mu, logvar):
    # KL(q||N(0,I)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
