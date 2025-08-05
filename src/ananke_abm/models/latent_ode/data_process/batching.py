"""
Implements the 'Unified Timeline' batching strategy for the Latent ODE model.

This module provides a function to be used as a `collate_fn` in a PyTorch
DataLoader. It takes a list of individual, variable-length time series samples
and combines them into a single, dense batch suitable for parallel processing.
"""
import torch
from ananke_abm.data_generator.feature_engineering import get_feature_dimensions, MODE_ID_MAP, PURPOSE_ID_MAP


def unify_and_interpolate_batch(batch):
    """
    Takes a list of individual data samples and collates them into a single batch
    using the "Unified Timeline" strategy, with intelligent interpolation and
    pre-computed anchor indices for time-weighted embedding loss.
    """
    # --- 1. Collect all features and create the unified timeline ---
    all_times = [s['times'] for s in batch]
    all_y_loc = [s['trajectory_y'] for s in batch]
    all_y_purp = [s['target_purpose_ids'] for s in batch]
    all_y_mode = [s['target_mode_ids'] for s in batch]
    all_y_purp_feat = [s['target_purpose_features'] for s in batch]
    all_y_mode_feat = [s['target_mode_features'] for s in batch]
    all_imp_weights = [s['importance_weights'] for s in batch]
    
    t_unified = torch.cat(all_times).unique(sorted=True)
    unified_len = len(t_unified)
    batch_size = len(batch)
    device = batch[0]['person_features'].device
    config = batch[0]['config']
    
    # Get feature dimensions
    mode_feat_dim, purp_feat_dim = get_feature_dimensions()

    # --- 2. Create dense tensors and masks ---
    y_loc_dense = torch.full((batch_size, unified_len), -1, dtype=torch.long, device=device)
    y_purp_dense = torch.full((batch_size, unified_len), -1, dtype=torch.long, device=device)
    y_mode_dense = torch.full((batch_size, unified_len), -1, dtype=torch.long, device=device)
    y_purp_feat_dense = torch.zeros((batch_size, unified_len, purp_feat_dim), dtype=torch.float32, device=device)
    y_mode_feat_dense = torch.zeros((batch_size, unified_len, mode_feat_dim), dtype=torch.float32, device=device)
    
    loss_mask = torch.zeros((batch_size, unified_len), device=device)
    importance_mask = torch.ones((batch_size, unified_len), device=device)
    
    if config.train_on_interpolated_points:
        loss_mask.fill_(1.0)
    
    # Get the ID for "Travel" for intelligent filling
    travel_purpose_id = PURPOSE_ID_MAP["travel"]
    
    # Get the ID for "Stay" mode for intelligent filling (for mode transition)
    stay_mode_id = MODE_ID_MAP["stay"]
    
    # --- 3. Pre-compute anchor indices and fill dense tensors ---
    prev_real_indices = torch.zeros((batch_size, unified_len), dtype=torch.long, device=device)
    next_real_indices = torch.zeros((batch_size, unified_len), dtype=torch.long, device=device)
    
    t_to_idx = {t.item(): i for i, t in enumerate(t_unified)}

    for i in range(batch_size):
        original_times = all_times[i]
        indices_in_unified = torch.tensor([t_to_idx[t.item()] for t in original_times], device=device)
        
        # Fill dense tensors with data from real observation points
        y_loc_dense[i, indices_in_unified] = all_y_loc[i]
        y_purp_dense[i, indices_in_unified] = all_y_purp[i]
        y_mode_dense[i, indices_in_unified] = all_y_mode[i]
        y_purp_feat_dense[i, indices_in_unified] = all_y_purp_feat[i]
        y_mode_feat_dense[i, indices_in_unified] = all_y_mode_feat[i]
        importance_mask[i, indices_in_unified] = all_imp_weights[i]
        
        if not config.train_on_interpolated_points:
            loss_mask[i, indices_in_unified] = 1.0

        real_indices = (y_loc_dense[i] != -1).nonzero().squeeze(-1)
        if len(real_indices) == 0:
            continue

        # Vectorized calculation of previous and next real indices
        arange_vec = torch.arange(unified_len, device=device)
        next_indices_in_real = torch.searchsorted(real_indices, arange_vec, side='right')
        prev_indices_in_real = torch.searchsorted(real_indices, arange_vec, side='left') - 1
        
        next_indices_in_real = torch.clamp(next_indices_in_real, 0, len(real_indices) - 1)
        prev_indices_in_real = torch.clamp(prev_indices_in_real, 0, len(real_indices) - 1)
        
        prev_real_indices[i] = real_indices[prev_indices_in_real]
        next_real_indices[i] = real_indices[next_indices_in_real]

        # Intelligently fill purpose and mode IDs for interpolated points
        if len(real_indices) > 1:
            for j in range(len(real_indices) - 1):
                start_idx, end_idx = real_indices[j], real_indices[j+1]
                if start_idx + 1 < end_idx:
                    start_purp, end_purp = y_purp_dense[i, start_idx], y_purp_dense[i, end_idx]
                    fill_value = travel_purpose_id if start_purp != end_purp else start_purp.item()
                    y_purp_dense[i, start_idx + 1 : end_idx] = fill_value
                    
                    start_mode, end_mode = y_mode_dense[i, start_idx], y_mode_dense[i, end_idx]
                    if start_purp != end_purp:
                        transition_mode = start_mode.item() if start_mode != stay_mode_id else end_mode.item()
                        y_mode_dense[i, start_idx + 1 : end_idx] = transition_mode
                    else:
                        y_mode_dense[i, start_idx + 1 : end_idx] = start_mode.item()

    # Combine the masks here to create the final weight mask
    final_loss_mask = loss_mask * importance_mask

    # --- 4. Stack all other features ---
    return {
        't_unified': t_unified,
        'y_loc_dense': y_loc_dense,
        'y_purp_dense': y_purp_dense,
        'y_mode_dense': y_mode_dense,
        'y_purp_feat_dense': y_purp_feat_dense,
        'y_mode_feat_dense': y_mode_feat_dense,
        'loss_mask': final_loss_mask,
        'prev_real_indices': prev_real_indices,
        'next_real_indices': next_real_indices,
        'person_features': torch.stack([s['person_features'] for s in batch]),
        'home_zone_features': torch.stack([s['home_zone_features'] for s in batch]),
        'work_zone_features': torch.stack([s['work_zone_features'] for s in batch]),
        'all_zone_features': batch[0]['all_zone_features'], # Same for all samples
        'num_zones': batch[0]['num_zones'],
        'purpose_groups': config.purpose_groups,
        'person_names': [s['person_name'] for s in batch]
    }
