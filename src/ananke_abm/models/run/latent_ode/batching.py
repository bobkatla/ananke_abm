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
    for interpolation in the embedding space.

    This function prepares the raw materials for the "Unified Timeline" strategy:
    1.  It creates a unified timeline from all timestamps in the batch.
    2.  It creates a "sparse" ground-truth tensor with real zone IDs at their
        original timestamps and a fill_value elsewhere.
    3.  It creates a loss mask (1 for real, 0 for interpolated).
    4.  Most importantly, it computes the indices of the previous and next
        real observations for every point on the timeline. This is used later
        to perform a fast, vectorized interpolation.

    Args:
        batch (list): A list of dictionaries from the `LatentODEDataset`.

    Returns:
        dict: A dictionary of batched tensors ready for the model.
    """
    # --- 1. Collect features and create unified timeline ---
    all_times = [s['times'] for s in batch]
    all_y = [s['trajectory_y'] for s in batch]
    
    t_unified = torch.cat(all_times).unique(sorted=True)
    unified_len = len(t_unified)
    batch_size = len(batch)
    device = batch[0]['person_features'].device

    # --- 2. Create sparse tensors and masks ---
    y_sparse_ids = torch.full((batch_size, unified_len), -1, dtype=torch.long)
    loss_mask = torch.zeros((batch_size, unified_len))
    
    # Pre-compute a map from time value to index in the unified timeline
    t_to_idx = {t.item(): i for i, t in enumerate(t_unified)}

    # --- 3. Compute previous and next indices for interpolation ---
    prev_indices = torch.zeros((batch_size, unified_len), dtype=torch.long)
    next_indices = torch.zeros((batch_size, unified_len), dtype=torch.long)

    for i in range(batch_size):
        # Populate sparse y with real values
        original_times = all_times[i]
        original_y = all_y[i]
        indices_in_unified = torch.tensor([t_to_idx[t.item()] for t in original_times])
        
        y_sparse_ids[i, indices_in_unified] = original_y
        loss_mask[i, indices_in_unified] = 1.0

        # Vectorized calculation of previous and next indices
        real_indices = (y_sparse_ids[i] != -1).nonzero().squeeze(-1)
        if len(real_indices) == 0:
            # Handle case with no real observations if necessary
            continue

        arange_vec = torch.arange(unified_len)
        
        # Find index in `real_indices` for each point in the timeline
        # `right=True` makes it find the insertion point, giving us the next real index
        next_indices_in_real = torch.searchsorted(real_indices, arange_vec, right=False)
        # Clamp to ensure we don't go out of bounds
        next_indices_in_real = torch.clamp(next_indices_in_real, 0, len(real_indices) - 1)
        
        prev_indices_in_real = next_indices_in_real - 1
        # Clamp to ensure we don't go below the first valid index
        prev_indices_in_real = torch.clamp(prev_indices_in_real, 0, len(real_indices) - 1)

        prev_indices[i] = real_indices[prev_indices_in_real]
        next_indices[i] = real_indices[next_indices_in_real]

    # --- 4. Stack all other features ---
    return {
        't_unified': t_unified.to(device),
        'y_sparse_ids': y_sparse_ids.to(device),
        'loss_mask': loss_mask.to(device),
        'prev_indices': prev_indices.to(device),
        'next_indices': next_indices.to(device),
        'person_features': torch.stack([s['person_features'] for s in batch]).to(device),
        'home_zone_id': torch.tensor([s['home_zone_id'] for s in batch], dtype=torch.long).to(device),
        'work_zone_id': torch.tensor([s['work_zone_id'] for s in batch], dtype=torch.long).to(device),
        'purpose_features': torch.stack([s['purpose_features'] for s in batch]).to(device),
        'num_zones': batch[0]['num_zones'],
        'person_names': [s['person_name'] for s in batch]
    } 