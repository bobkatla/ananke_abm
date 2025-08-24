import torch
import torch.nn.functional as F

@torch.no_grad()
def rasterize_from_padded(p_idx_pad: torch.Tensor, t0_pad: torch.Tensor, d_pad: torch.Tensor,
                          lengths, P: int, t_nodes: torch.Tensor) -> torch.Tensor:
    """
    Vectorized rasterizer: (B,L), (B,L), (B,L), lengths, P, (Q,) -> (B,P,Q)
    All tensors should already be on the same device as t_nodes.
    """
    device = t_nodes.device
    B, L = p_idx_pad.shape
    # mask valid segments
    lengths_t = torch.as_tensor(lengths, device=p_idx_pad.device)
    valid = (torch.arange(L, device=p_idx_pad.device).unsqueeze(0) < lengths_t.unsqueeze(1))  # (B,L)
    # segment time spans
    t1_pad = t0_pad + d_pad
    tq = t_nodes.view(1, 1, -1)  # (1,1,Q)
    seg_mask = (tq >= t0_pad.unsqueeze(-1)) & (tq < t1_pad.unsqueeze(-1))  # (B,L,Q)
    seg_mask = seg_mask & valid.unsqueeze(-1)                               # (B,L,Q)
    seg_mask = seg_mask.to(torch.float32)
    # route by purpose with one‑hot and sum over segments
    onehot = F.one_hot(p_idx_pad.clamp_min(0), num_classes=P).to(seg_mask.dtype)  # (B,L,P)
    y = torch.einsum('blp,blq->bpq', onehot, seg_mask)
    return y.clamp_(0, 1)

def rasterize_batch(segments_batch, purpose_to_idx, t_nodes):
    """
    Convert batch of variable-length segments into one-hot over time nodes.
    Args:
        segments_batch: list of lists; for each b: [(purpose_str, t0_norm, d_norm), ...]
        purpose_to_idx: dict mapping purpose string -> index [0..P-1]
        t_nodes: (Q,) tensor in [0,1], quadrature nodes
    Returns:
        y: (B, P, Q) tensor with one-hot along P at each time node
    """
    B = len(segments_batch)
    Q = t_nodes.shape[0]
    P = len(purpose_to_idx)
    y = torch.zeros((B, P, Q), dtype=torch.float32)
    for b, segs in enumerate(segments_batch):
        for (p, t0, d) in segs:
            p_idx = purpose_to_idx[p]
            t1 = t0 + d
            mask = (t_nodes >= t0) & (t_nodes < t1)
            y[b, p_idx, mask] = 1.0
    return y
