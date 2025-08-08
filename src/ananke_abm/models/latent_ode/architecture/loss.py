"""
Composite loss function for the SDE model with segment-based mode loss.
"""
import torch
import torch.nn.functional as F

def calculate_path_loss(batch, model_outputs, model, distance_matrix, config):
    """
    Calculates snap-level losses for location and purpose, and stay velocity loss.
    """
    (
        pred_loc_logits, pred_loc_embed, 
        pred_purp_logits, pred_purpose_features,
        pred_p, pred_v,
        mu, log_var
    ) = model_outputs
    
    # --- Map union grid predictions to dense grid for loss calculation ---
    union_indices = batch['union_to_dense']
    
    pred_loc_logits_union = pred_loc_logits[:, union_indices, :]
    pred_loc_embed_union = pred_loc_embed[:, union_indices, :]
    pred_purp_logits_union = pred_purp_logits[:, union_indices, :]
    pred_purpose_features_union = pred_purpose_features[:, union_indices, :]

    # --- 1. Location & Purpose Losses (at GT snaps on the union grid) ---
    is_gt_mask = batch['is_gt_union']
    
    # Location CE Loss
    loss_loc_ce = F.cross_entropy(
        pred_loc_logits_union.transpose(1, 2), 
        batch['loc_ids_union'],
        ignore_index=-100, # Important: ignore interpolated points
        reduction='none'
    )
    loss_loc_ce = (loss_loc_ce * is_gt_mask).sum() / is_gt_mask.sum().clamp(min=1)

    # Purpose CE Loss
    loss_purp_ce = F.cross_entropy(
        pred_purp_logits_union.transpose(1, 2),
        batch['purp_ids_union'],
        ignore_index=-100,
        reduction='none'
    )
    loss_purp_ce = (loss_purp_ce * is_gt_mask).sum() / is_gt_mask.sum().clamp(min=1)
    
    # --- Feature Reconstruction MSE Loss (on Normalized Embeddings) ---
    pred_loc_embed_union_norm = F.normalize(pred_loc_embed_union, p=2, dim=-1)
    gt_loc_emb_union_norm = F.normalize(batch['loc_emb_union'], p=2, dim=-1) # Normalize GT for fair comparison
    loss_loc_mse = F.mse_loss(pred_loc_embed_union_norm, gt_loc_emb_union_norm, reduction='none').mean(dim=-1)
    loss_loc_mse = (loss_loc_mse * is_gt_mask).sum() / is_gt_mask.sum().clamp(min=1)
    
    pred_purpose_features_union_norm = F.normalize(pred_purpose_features_union, p=2, dim=-1)
    gt_purp_emb_union_norm = F.normalize(batch['purp_emb_union'], p=2, dim=-1)
    loss_purp_mse = F.mse_loss(pred_purpose_features_union_norm, gt_purp_emb_union_norm, reduction='none').mean(dim=-1)
    loss_purp_mse = (loss_purp_mse * is_gt_mask).sum() / is_gt_mask.sum().clamp(min=1)
    
    # --- 2. Stay Velocity Loss ---
    stay_mask = batch['stay_mask']
    velocity_loss_raw = pred_v.pow(2)
    loss_stay_velocity = (velocity_loss_raw.sum(dim=-1) * stay_mask).sum()
    num_stay_points = stay_mask.sum().clamp(min=1)
    loss_stay_velocity = loss_stay_velocity / num_stay_points

    # --- 3. KL Divergence ---
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / mu.shape[0]

    return loss_loc_ce, loss_loc_mse, loss_purp_ce, loss_purp_mse, kl_loss, loss_stay_velocity

def calculate_segment_mode_loss(seg_logits, seg_h, segments_batch, config):
    """
    Calculates segment-level mode classification and feature reconstruction loss.
    """
    if not segments_batch:
        device = seg_logits.device if isinstance(seg_logits, torch.Tensor) and seg_logits.numel() > 0 else torch.device("cpu")
        return torch.zeros((), device=device), torch.zeros((), device=device)

    # Gather ground truth from the ragged segment batch
    target_mode_ids = torch.tensor([s['mode_id'] for s in segments_batch], dtype=torch.long, device=seg_logits.device)
    target_mode_protos = torch.stack([s['mode_proto'] for s in segments_batch])

    # Mode CE Loss
    loss_mode_ce = F.cross_entropy(seg_logits, target_mode_ids)

    # Mode Feature MSE Loss (on Normalized Embeddings)
    seg_h_norm = F.normalize(seg_h, p=2, dim=-1)
    target_mode_protos_norm = F.normalize(target_mode_protos, p=2, dim=-1)
    loss_mode_feat = F.mse_loss(seg_h_norm, target_mode_protos_norm)
    
    return loss_mode_ce, loss_mode_feat
