import torch
import torch.nn.functional as F

def masked_ce_mse(
    logits_act, logits_dur, acts, durs, mask,
    label_smoothing: float = 0.0,  # set to 0.05 to reduce collapse
):
    """
    logits_act: (B, L-1, V)
    logits_dur: (B, L-1)
    acts,durs,mask: (B, L), where mask=True only at non-special tokens.
    We compute loss on targets at positions 1..L-1.
    """
    B, L = acts.shape
    V = logits_act.size(-1)

    tgt_act = acts[:, 1:]          # (B, L-1) long
    tgt_dur = durs[:, 1:]          # (B, L-1) float
    tgt_msk = mask[:, 1:]          # (B, L-1) bool

    # count valid tokens
    valid = tgt_msk.sum().clamp_min(1)

    # ---- Activity CE (with optional label smoothing) ----
    if valid.item() > 0:
        # Gather only valid positions
        logits_act_flat = logits_act[tgt_msk]        # (N, V)
        tgt_act_flat    = tgt_act[tgt_msk]           # (N,)

        if label_smoothing and label_smoothing > 0.0:
            # label-smoothed CE
            n_classes = V
            eps = label_smoothing
            with torch.no_grad():
                # one-hot with smoothing
                true = torch.zeros_like(logits_act_flat).scatter_(1, tgt_act_flat.unsqueeze(1), 1.0)
                true = true * (1.0 - eps) + eps / n_classes
            logp = F.log_softmax(logits_act_flat, dim=-1)
            ce_sum = -(true * logp).sum(dim=-1).sum()
            ce = ce_sum / valid
        else:
            ce = F.cross_entropy(logits_act_flat, tgt_act_flat, reduction="mean")
    else:
        ce = logits_act.sum() * 0.0

    # ---- Duration MSE ----
    if valid.item() > 0:
        pred_dur = torch.sigmoid(logits_dur)         # (B, L-1)
        mse = F.mse_loss(pred_dur[tgt_msk], tgt_dur[tgt_msk], reduction="mean")
    else:
        pred_dur = torch.sigmoid(logits_dur)
        mse = pred_dur.sum() * 0.0

    # For logging/debug
    stats = {
        "n_tokens": int(valid.item()),
        "ce": float(ce.detach().cpu()),
        "mse": float(mse.detach().cpu()),
    }
    return ce, mse, stats

def kl_normal(mu, logvar):
    # KL(q||N(0,I))
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
