"""
Implements the dense evaluation grid batching strategy for the SDE model.
"""
import torch

K_INTERNAL = 8  # Number of evaluation points per segment (including endpoints)

def sde_collate_fn(batch):
    """
    Collates samples into a batch with a dense evaluation grid and ragged segments.
    """
    # --- 1. Unified GT Snap Grid ---
    all_gt_times = [s['gt_times'] for s in batch]
    gt_union = torch.cat(all_gt_times).unique(sorted=True)
    
    # --- 2. Dense Evaluation Grid ---
    dense_times = []
    for i in range(len(gt_union) - 1):
        t_start, t_end = gt_union[i], gt_union[i+1]
        dense_times.append(torch.linspace(t_start, t_end, K_INTERNAL)[:-1])
    dense_times.append(gt_union[-1].unsqueeze(0))
    grid_times = torch.cat(dense_times)
    
    is_gt_grid = torch.isin(grid_times, gt_union)
    
    # --- 3. Interpolate State to GT Union Grid ---
    batch_size = len(batch)
    d_loc = batch[0]['gt_loc_emb'].shape[1]
    d_purp = batch[0]['gt_purp_emb'].shape[1]
    s_gt = len(gt_union)

    loc_emb_union = torch.zeros((batch_size, s_gt, d_loc), dtype=torch.float32)
    purp_emb_union = torch.zeros((batch_size, s_gt, d_purp), dtype=torch.float32)
    loc_ids_union = torch.full((batch_size, s_gt), -100, dtype=torch.long) # Use ignore_index
    purp_ids_union = torch.full((batch_size, s_gt), -100, dtype=torch.long) # Use ignore_index
    is_gt_union = torch.zeros((batch_size, s_gt), dtype=torch.float32)
    anchor_union = torch.zeros((batch_size, s_gt), dtype=torch.float32)

    for i in range(batch_size):
        person_gt_times = batch[i]['gt_times']
        person_gt_indices = torch.searchsorted(gt_union, person_gt_times)
        
        loc_ids_union[i, person_gt_indices] = batch[i]['gt_loc_ids']
        purp_ids_union[i, person_gt_indices] = batch[i]['gt_purp_ids']
        is_gt_union[i, person_gt_indices] = 1.0
        anchor_union[i, person_gt_indices] = batch[i]['gt_anchor']

        for j, t_union in enumerate(gt_union):
            # Find bracketing GT snaps for this person
            k_next = torch.searchsorted(person_gt_times, t_union, side='left')
            k_prev = k_next - 1
            
            k_prev = torch.clamp(k_prev, 0, len(person_gt_times) - 1)
            k_next = torch.clamp(k_next, 0, len(person_gt_times) - 1)
            
            t_prev, t_next = person_gt_times[k_prev], person_gt_times[k_next]
            
            if t_prev == t_next:
                w_prev, w_next = 1.0, 0.0
            else:
                w_next = (t_union - t_prev) / (t_next - t_prev).clamp(min=1e-6)
                w_prev = 1.0 - w_next

            loc_emb_union[i, j] = w_prev * batch[i]['gt_loc_emb'][k_prev] + w_next * batch[i]['gt_loc_emb'][k_next]
            purp_emb_union[i, j] = w_prev * batch[i]['gt_purp_emb'][k_prev] + w_next * batch[i]['gt_purp_emb'][k_next]

    # --- 4. Map Union -> Dense Indices ---
    union_to_dense = torch.searchsorted(grid_times, gt_union)

    # --- 5. Segments as Ragged List ---
    segments_batch = []
    for i in range(batch_size):
        person_gt_times = batch[i]['gt_times']
        union_indices = torch.searchsorted(gt_union, person_gt_times)
        
        for seg in batch[i]['segments']:
            segments_batch.append({
                "b": i,
                "i0": union_to_dense[union_indices[seg["snap_i0"]]].item(),
                "i1": union_to_dense[union_indices[seg["snap_i1"]]].item(),
                "mode_id": seg["mode_id"],
                "mode_proto": seg["mode_proto"],
            })

    # --- 6. Create Stay Mask for Velocity Loss ---
    device = batch[0]['gt_times'].device
    B, S_dense = len(batch), len(grid_times)
    stay_mask = torch.zeros(B, S_dense, device=device)
    for i, item in enumerate(batch):
        for start_time, end_time in item['stay_intervals']:
            interval_mask = (grid_times >= start_time) & (grid_times <= end_time)
            stay_mask[i, interval_mask] = 1.0

    return {
        'grid_times': grid_times,
        'is_gt_grid': is_gt_grid,
        'gt_union_times': gt_union,
        'loc_emb_union': loc_emb_union,
        'loc_ids_union': loc_ids_union,
        'purp_emb_union': purp_emb_union,
        'purp_ids_union': purp_ids_union,
        'is_gt_union': is_gt_union,
        'anchor_union': anchor_union,
        'union_to_dense': union_to_dense,
        'segments_batch': segments_batch,
        'stay_mask': stay_mask,
        'person_ids': [s['person_id'] for s in batch]
    }
