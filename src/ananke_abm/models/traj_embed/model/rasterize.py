import torch

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
