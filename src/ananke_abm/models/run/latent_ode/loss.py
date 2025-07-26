"""
Composite loss function for the Generative Latent ODE model.
"""
import torch
import torch.nn.functional as F

def calculate_composite_loss(
    pred_y_logits,
    pred_y_embeds,
    pred_purpose_logits,
    target_y_ids,
    target_purpose_ids,
    model,
    mu,
    log_var,
    distance_matrix,
    config
):
    """
    Calculates a weighted, composite loss with four main components:
    1.  Location Classification Loss (Cross-Entropy on zone logits).
    2.  Location Embedding Loss (MSE on predicted vs. target zone embeddings).
    3.  Physical Distance Loss (Physical distance between predicted and target zones).
    4.  Purpose Classification Loss (Cross-Entropy on purpose logits).
    """
    # Squeeze to remove the batch dimension of 1 for single-person training
    pred_y_logits = pred_y_logits.squeeze(0)
    pred_y_embeds = pred_y_embeds.squeeze(0)
    pred_purpose_logits = pred_purpose_logits.squeeze(0)

    # --- Create Anchor Mask for the first time step ---
    # This gives extra weight to getting the first point right.
    anchor_mask = torch.ones(target_y_ids.shape[0], device=target_y_ids.device)
    anchor_mask[0] = config.initial_step_loss_weight

    # --- 1. Location Classification Loss (Cross-Entropy) ---
    loss_classification = F.cross_entropy(pred_y_logits, target_y_ids, reduction='none')
    loss_classification = (loss_classification * anchor_mask).mean()

    # --- 2. Location Embedding Loss (MSE) ---
    target_y_embeds = model.zone_embedder(target_y_ids)
    # Calculate MSE per element, then apply mask
    loss_embedding_per_element = F.mse_loss(pred_y_embeds, target_y_embeds, reduction='none').mean(dim=1)
    loss_embedding = (loss_embedding_per_element * anchor_mask).mean()
    
    # --- 3. Physical Distance Loss ---
    pred_y_ids = torch.argmax(pred_y_logits, dim=1)
    physical_distances = distance_matrix[pred_y_ids, target_y_ids]
    loss_distance = (physical_distances * anchor_mask).mean()
    
    # --- 4. Purpose Classification Loss ---
    loss_purpose = F.cross_entropy(pred_purpose_logits, target_purpose_ids, reduction='none')
    loss_purpose = (loss_purpose * anchor_mask).mean()

    # --- 5. KL Divergence (not affected by the anchor) ---
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # --- 6. Combine all losses with their weights ---
    total_loss = (
        config.loss_weight_classification * loss_classification +
        config.loss_weight_embedding * loss_embedding +
        config.loss_weight_distance * loss_distance +
        config.loss_weight_purpose * loss_purpose +
        config.kl_weight * kl_loss
    )
    
    return total_loss, loss_classification, loss_embedding, loss_distance, loss_purpose, kl_loss 