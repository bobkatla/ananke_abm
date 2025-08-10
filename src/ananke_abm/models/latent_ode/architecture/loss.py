"""
Composite loss function for the Generative Latent ODE model.
"""
import torch
import torch.nn.functional as F

def calculate_composite_loss(batch, model_outputs, model, distance_matrix, config):
    """
    Calculates a weighted, composite loss for a BATCH of data.
    - Uses a loss_mask to only compute loss on real (non-interpolated) data points.
    - Performs time-weighted interpolation for the location embedding loss.
    - Includes classification and MSE loss for both purpose and mode.
    """
    (
        pred_y_loc_logits, pred_y_loc_embed, 
        pred_y_purp_logits, pred_y_mode_logits, 
        pred_purpose_features, pred_mode_features,
        mu, log_var
    ) = model_outputs
    
    t_unified = batch['t_unified']
    target_y_loc_dense = batch['y_loc_dense']
    target_y_purp_dense = batch['y_purp_dense']
    target_y_mode_dense = batch['y_mode_dense']
    target_y_purp_feat_dense = batch['y_purp_feat_dense']
    target_y_mode_feat_dense = batch['y_mode_feat_dense']
    loss_mask = batch['loss_mask']
    
    batch_size = pred_y_loc_logits.shape[0]
    
    # --- 1. Location Classification Loss (Cross-Entropy) ---
    pred_y_logits_flat = pred_y_loc_logits.view(-1, pred_y_loc_logits.shape[-1])
    target_y_loc_flat = target_y_loc_dense.view(-1)
    loss_classification_unmasked = F.cross_entropy(pred_y_logits_flat, target_y_loc_flat, ignore_index=-1, reduction='none')
    loss_classification = (loss_classification_unmasked * loss_mask.view(-1)).sum() / loss_mask.sum()

    # --- 2. Location Embedding Loss (Time-Weighted MSE) ---
    all_candidate_embeds = model.zone_feature_encoder(batch['all_zone_features'])
    prev_real_zone_ids = torch.gather(target_y_loc_dense, 1, batch['prev_real_indices'])
    next_real_zone_ids = torch.gather(target_y_loc_dense, 1, batch['next_real_indices'])
    prev_embeds = all_candidate_embeds[prev_real_zone_ids.clamp(min=0)]
    next_embeds = all_candidate_embeds[next_real_zone_ids.clamp(min=0)]
    
    t_prev = t_unified[batch['prev_real_indices']]
    t_next = t_unified[batch['next_real_indices']]
    
    w_next = (t_unified.unsqueeze(0) - t_prev) / (t_next - t_prev + 1e-8)
    w_next = torch.clamp(w_next, 0, 1).unsqueeze(-1)
    
    target_y_embeds = (1 - w_next) * prev_embeds + w_next * next_embeds
    
    loss_embedding_unmasked = F.mse_loss(pred_y_loc_embed, target_y_embeds, reduction='none').mean(dim=-1)
    loss_embedding = (loss_embedding_unmasked * loss_mask).sum() / loss_mask.sum()
    
    # --- 3. Physical Distance Loss ---
    pred_y_ids = torch.argmax(pred_y_loc_logits, dim=2)
    physical_distances = distance_matrix[pred_y_ids, target_y_loc_dense.clamp(min=0)]
    loss_distance = (physical_distances * loss_mask).sum() / loss_mask.sum()
    
    # --- 4a. Purpose Classification Loss ---
    pred_purpose_logits_flat = pred_y_purp_logits.view(-1, pred_y_purp_logits.shape[-1])
    target_y_purp_flat = target_y_purp_dense.view(-1)
    loss_purpose_class_unmasked = F.cross_entropy(pred_purpose_logits_flat, target_y_purp_flat, ignore_index=-1, reduction='none')
    loss_purpose_class = (loss_purpose_class_unmasked * loss_mask.view(-1)).sum() / loss_mask.sum()

    # --- 4b. Purpose Feature MSE Loss ---
    loss_purpose_mse_unmasked = F.mse_loss(pred_purpose_features, target_y_purp_feat_dense, reduction='none').mean(dim=-1)
    loss_purpose_mse = (loss_purpose_mse_unmasked * loss_mask).sum() / loss_mask.sum()

    # --- 5a. Mode Classification Loss ---
    pred_mode_logits_flat = pred_y_mode_logits.view(-1, pred_y_mode_logits.shape[-1])
    target_y_mode_flat = target_y_mode_dense.view(-1)
    loss_mode_class_unmasked = F.cross_entropy(pred_mode_logits_flat, target_y_mode_flat, ignore_index=-1, reduction='none')
    loss_mode_class = (loss_mode_class_unmasked * loss_mask.view(-1)).sum() / loss_mask.sum()

    # --- 5b. Mode Feature MSE Loss ---
    loss_mode_mse_unmasked = F.mse_loss(pred_mode_features, target_y_mode_feat_dense, reduction='none').mean(dim=-1)
    loss_mode_mse = (loss_mode_mse_unmasked * loss_mask).sum() / loss_mask.sum()

    # --- 6. KL Divergence (Averaged over batch) ---
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size
    
    # --- 7. Combine all losses with their weights ---
    total_loss = (
        config.loss_weight_classification * loss_classification +
        config.loss_weight_embedding * loss_embedding +
        config.loss_weight_distance * loss_distance +
        config.loss_weight_purpose_class * loss_purpose_class +
        config.loss_weight_mode_class * loss_mode_class +
        config.loss_weight_purpose_mse * loss_purpose_mse +
        config.loss_weight_mode_mse * loss_mode_mse +
        config.kl_weight * kl_loss
    )
    
    return (
        total_loss, loss_classification, loss_embedding, loss_distance,
        loss_purpose_class, loss_purpose_mse, 
        loss_mode_class, loss_mode_mse,
        kl_loss
    )
