"""
Implements a simplified 'Unified Timeline' batching strategy for the SDE model.

This module provides a collate_fn for a PyTorch DataLoader that combines 
variable-length time series samples into a dense batch suitable for the SDE model,
using linear interpolation for state embeddings between ground-truth points.
"""
import torch


def sde_collate_fn(batch):
    """
    Collates individual data samples into a single batch using a simplified 
    "Unified Timeline" strategy with linear interpolation.
    """
    # --- 1. Collect all features and create the unified timeline in minutes ---
    all_gt_times = [s['gt_times'] for s in batch]
    grid_times = torch.cat(all_gt_times).unique(sorted=True)
    
    batch_size = len(batch)
    grid_len = len(grid_times)
    device = grid_times.device
    
    # Assuming embeddings dimensions are consistent across the batch
    d_loc = batch[0]['gt_loc_emb'].shape[1]
    d_purp = batch[0]['gt_purp_emb'].shape[1]

    # --- 2. Create dense tensors and masks ---
    loc_emb_batch = torch.zeros((batch_size, grid_len, d_loc), dtype=torch.float32, device=device)
    purp_emb_batch = torch.zeros((batch_size, grid_len, d_purp), dtype=torch.float32, device=device)
    is_gt_batch = torch.zeros((batch_size, grid_len), dtype=torch.float32, device=device)
    anchor_mask_batch = torch.zeros((batch_size, grid_len), dtype=torch.float32, device=device)

    # --- 3. Interpolate and fill dense tensors for each person ---
    for i in range(batch_size):
        gt_times = batch[i]['gt_times']
        gt_loc_emb = batch[i]['gt_loc_emb']
        gt_purp_emb = batch[i]['gt_purp_emb']
        gt_anchor = batch[i]['gt_anchor']

        # Find indices of person's GT times in the unified grid
        gt_indices_in_grid = torch.searchsorted(grid_times, gt_times)
        
        # Populate batch tensors at ground-truth points
        loc_emb_batch[i, gt_indices_in_grid] = gt_loc_emb
        purp_emb_batch[i, gt_indices_in_grid] = gt_purp_emb
        is_gt_batch[i, gt_indices_in_grid] = 1.0
        anchor_mask_batch[i, gt_indices_in_grid] = gt_anchor
        
        # Interpolate between ground-truth points
        for j in range(len(gt_times) - 1):
            t_prev, t_next = gt_times[j], gt_times[j+1]
            idx_prev, idx_next = gt_indices_in_grid[j], gt_indices_in_grid[j+1]

            if idx_next > idx_prev + 1: # If there are points to interpolate
                loc_emb_prev, loc_emb_next = gt_loc_emb[j], gt_loc_emb[j+1]
                purp_emb_prev, purp_emb_next = gt_purp_emb[j], gt_purp_emb[j+1]
                
                # Time-weighted interpolation
                time_gap = (t_next - t_prev).clamp(min=1e-6)
                interp_times = grid_times[idx_prev + 1 : idx_next]
                w_next = (interp_times - t_prev) / time_gap
                w_prev = 1.0 - w_next

                loc_emb_batch[i, idx_prev + 1 : idx_next] = w_prev.unsqueeze(1) * loc_emb_prev + w_next.unsqueeze(1) * loc_emb_next
                purp_emb_batch[i, idx_prev + 1 : idx_next] = w_prev.unsqueeze(1) * purp_emb_prev + w_next.unsqueeze(1) * purp_emb_next

        # Clamp before the first and after the last snap
        if gt_indices_in_grid[0] > 0:
            loc_emb_batch[i, :gt_indices_in_grid[0]] = gt_loc_emb[0]
            purp_emb_batch[i, :gt_indices_in_grid[0]] = gt_purp_emb[0]
        if gt_indices_in_grid[-1] < grid_len - 1:
            loc_emb_batch[i, gt_indices_in_grid[-1]+1:] = gt_loc_emb[-1]
            purp_emb_batch[i, gt_indices_in_grid[-1]+1:] = gt_purp_emb[-1]

    return {
        'grid_times': grid_times,
        'loc_emb_batch': loc_emb_batch,
        'purp_emb_batch': purp_emb_batch,
        'is_gt_batch': is_gt_batch,
        'anchor_mask_batch': anchor_mask_batch,
        'person_ids': [s['person_id'] for s in batch]
    }
