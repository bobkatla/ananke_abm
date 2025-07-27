"""
Implements the 'Unified Timeline' batching strategy for the Latent ODE model.

This module provides a function to be used as a `collate_fn` in a PyTorch
DataLoader. It takes a list of individual, variable-length time series samples
and combines them into a single, dense batch suitable for parallel processing.
"""
import torch

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
    
    t_unified = torch.cat(all_times).unique(sorted=True)
    unified_len = len(t_unified)
    batch_size = len(batch)
    device = batch[0]['person_features'].device
    
    # --- 2. Create dense tensors and masks ---
    y_loc_dense = torch.full((batch_size, unified_len), -1, dtype=torch.long, device=device)
    y_purp_dense = torch.full((batch_size, unified_len), -1, dtype=torch.long, device=device)
    loss_mask = torch.zeros((batch_size, unified_len), device=device)
    
    config = batch[0]['config']
    if config.train_on_interpolated_points:
        loss_mask.fill_(1.0)

    # --- 3. Pre-compute anchor indices and fill dense tensors ---
    prev_real_indices = torch.zeros((batch_size, unified_len), dtype=torch.long, device=device)
    next_real_indices = torch.zeros((batch_size, unified_len), dtype=torch.long, device=device)
    
    travel_purpose_id = len(batch[0]['purpose_summary_features']) - 1
    t_to_idx = {t.item(): i for i, t in enumerate(t_unified)}

    for i in range(batch_size):
        original_times = all_times[i]
        indices_in_unified = torch.tensor([t_to_idx[t.item()] for t in original_times], device=device)
        
        y_loc_dense[i, indices_in_unified] = all_y_loc[i]
        y_purp_dense[i, indices_in_unified] = all_y_purp[i]
        
        # Only overwrite with 1s if we are not training on all points
        if not config.train_on_interpolated_points:
            loss_mask[i, indices_in_unified] = 1.0
        else:
            # When training on all points, we still need to know the real locations
            # for the anchor-based interpolation. We can reuse the y_loc_dense
            # where -1 indicates an interpolated point.
            pass


        real_indices = (y_loc_dense[i] != -1).nonzero().squeeze(-1)
        if len(real_indices) == 0:
            continue

        # Vectorized calculation of previous and next real indices
        arange_vec = torch.arange(unified_len, device=device)
        # For each point in the timeline, find the insertion point into the list of real indices
        # 'right=True' finds the index of the *next* real point
        next_indices_in_real = torch.searchsorted(real_indices, arange_vec, side='right')
        # 'side=left' gives the index of the *previous* real point, but we need to subtract 1
        prev_indices_in_real = torch.searchsorted(real_indices, arange_vec, side='left') - 1

        # Clamp to ensure we don't go out of bounds
        next_indices_in_real = torch.clamp(next_indices_in_real, 0, len(real_indices) - 1)
        prev_indices_in_real = torch.clamp(prev_indices_in_real, 0, len(real_indices) - 1)
        
        # Use the computed indices to get the actual indices in the unified timeline
        prev_real_indices[i] = real_indices[prev_indices_in_real]
        next_real_indices[i] = real_indices[next_indices_in_real]

        # Intelligently fill purpose IDs
        if len(real_indices) > 1:
            for j in range(len(real_indices) - 1):
                start_idx, end_idx = real_indices[j], real_indices[j+1]
                if start_idx + 1 < end_idx:
                    start_purp, end_purp = y_purp_dense[i, start_idx], y_purp_dense[i, end_idx]
                    fill_value = travel_purpose_id if start_purp != end_purp else start_purp.item()
                    y_purp_dense[i, start_idx + 1 : end_idx] = fill_value

    # --- 4. Stack all other features ---
    return {
        't_unified': t_unified,
        'y_loc_dense': y_loc_dense,
        'y_purp_dense': y_purp_dense,
        'loss_mask': loss_mask,
        'prev_real_indices': prev_real_indices,
        'next_real_indices': next_real_indices,
        'person_features': torch.stack([s['person_features'] for s in batch]),
        'home_zone_id': torch.tensor([s['home_zone_id'] for s in batch], dtype=torch.long, device=device),
        'work_zone_id': torch.tensor([s['work_zone_id'] for s in batch], dtype=torch.long, device=device),
        'purpose_summary_features': torch.stack([s['purpose_summary_features'] for s in batch]),
        'num_zones': batch[0]['num_zones'],
        'purpose_groups': config.purpose_groups,
        'person_names': [s['person_name'] for s in batch]
    } 