import torch
import torch.nn.functional as F


def start_end_home_loss(logits_batch, home_class_index):
    # logits_batch: (B, T, P)
    # we want high prob of home at t=0 and t=T-1
    B, T, P = logits_batch.shape
    if T < 2:
        return torch.tensor(0.0, device=logits_batch.device, dtype=logits_batch.dtype)
    logp0 = F.log_softmax(logits_batch[:, 0, :], dim=-1)      # (B,P)
    logpT = F.log_softmax(logits_batch[:, -1, :], dim=-1)     # (B,P)
    loss0 = -logp0[:, home_class_index].mean()
    lossT = -logpT[:, home_class_index].mean()
    return (loss0 + lossT) * 0.5
