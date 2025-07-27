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
    """
    pred_y_logits, pred_y_embeds, pred_purpose_logits, mu, log_var = model_outputs
    
    t_unified = batch['t_unified']
    target_y_loc_dense = batch['y_loc_dense']
    target_y_purp_dense = batch['y_purp_dense']
    loss_mask = batch['loss_mask']
    
    batch_size = pred_y_logits.shape[0]
    
    # --- 1. Location Classification Loss (Cross-Entropy) ---
    pred_y_logits_flat = pred_y_logits.view(-1, pred_y_logits.shape[-1])
    target_y_loc_flat = target_y_loc_dense.view(-1)
    loss_classification_unmasked = F.cross_entropy(pred_y_logits_flat, target_y_loc_flat, ignore_index=-1, reduction='none')
    loss_classification = (loss_classification_unmasked * loss_mask.view(-1)).sum() / loss_mask.sum()

    # --- 2. Location Embedding Loss (Time-Weighted MSE) ---
    # Get embeddings for previous and next real anchor points
    prev_embeds = torch.gather(model.zone_embedder(target_y_loc_dense.clamp(min=0)), 1, batch['prev_real_indices'].unsqueeze(-1).expand(-1, -1, config.zone_embed_dim))
    next_embeds = torch.gather(model.zone_embedder(target_y_loc_dense.clamp(min=0)), 1, batch['next_real_indices'].unsqueeze(-1).expand(-1, -1, config.zone_embed_dim))
    
    # Get timestamps for the anchor points
    t_prev = t_unified[batch['prev_real_indices']]
    t_next = t_unified[batch['next_real_indices']]
    
    # Calculate interpolation weights
    w_next = (t_unified.unsqueeze(0) - t_prev) / (t_next - t_prev + 1e-8)
    w_next = torch.clamp(w_next, 0, 1).unsqueeze(-1)
    
    # Create the dense target trajectory via interpolation
    target_y_embeds = (1 - w_next) * prev_embeds + w_next * next_embeds
    
    # Calculate masked MSE loss against the interpolated target
    loss_embedding_unmasked = F.mse_loss(pred_y_embeds, target_y_embeds, reduction='none').mean(dim=-1)
    loss_embedding = (loss_embedding_unmasked * loss_mask).sum() / loss_mask.sum()
    
    # --- 3. Physical Distance Loss ---
    pred_y_ids = torch.argmax(pred_y_logits, dim=2)
    physical_distances = distance_matrix[pred_y_ids, target_y_loc_dense.clamp(min=0)]
    loss_distance = (physical_distances * loss_mask).sum() / loss_mask.sum()
    
    # --- 4. Purpose Classification Loss ---
    pred_purpose_logits_flat = pred_purpose_logits.view(-1, pred_purpose_logits.shape[-1])
    target_y_purp_flat = target_y_purp_dense.view(-1)
    loss_purpose_unmasked = F.cross_entropy(pred_purpose_logits_flat, target_y_purp_flat, ignore_index=-1, reduction='none')
    loss_purpose = (loss_purpose_unmasked * loss_mask.view(-1)).sum() / loss_mask.sum()

    # --- 5. KL Divergence (Averaged over batch) ---
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size
    
    # --- 6. Combine all losses with their weights ---
    total_loss = (
        config.loss_weight_classification * loss_classification +
        config.loss_weight_embedding * loss_embedding +
        config.loss_weight_distance * loss_distance +
        config.loss_weight_purpose * loss_purpose +
        config.kl_weight * kl_loss
    )
    
    return total_loss, loss_classification, loss_embedding, loss_distance, loss_purpose, kl_loss 