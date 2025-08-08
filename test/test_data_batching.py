"""
Tests for the dense grid, snap-fitting, and ragged segment batching pipeline.
"""
import pytest
import torch
import sys
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ananke_abm.models.latent_ode.data_process.data import LatentSDEDataset, DataProcessor
from src.ananke_abm.models.latent_ode.data_process.batching import sde_collate_fn, K_INTERNAL


@pytest.fixture(scope="module")
def batch_and_processor():
    """Fixture to load test data, initialize processor, and create a single batch."""
    device = torch.device("cpu")
    periods_path = "test/test_periods_small.csv"
    snaps_path = "test/test_snaps_small.csv"

    processor = DataProcessor(device, periods_path=periods_path, snaps_path=snaps_path)
    dataset = LatentSDEDataset(person_ids=[1, 2], processor=processor)
    loader = DataLoader(dataset, batch_size=2, collate_fn=sde_collate_fn)
    batch = next(iter(loader))
    
    return batch, processor

def test_grid_creation(batch_and_processor):
    """Validates the dense evaluation grid."""
    batch, _ = batch_and_processor
    gt_union = batch['gt_union_times']
    
    assert torch.all(batch['grid_times'][1:] > batch['grid_times'][:-1]), "Grid times not strictly increasing"
    assert batch['is_gt_grid'].sum() == len(gt_union), "Mismatch between is_gt_grid and gt_union"
    assert torch.all(batch['grid_times'][batch['is_gt_grid']] == gt_union), "GT times in grid do not match gt_union"
    
    expected_dense_len = (len(gt_union) - 1) * (K_INTERNAL - 1) + 1
    assert len(batch['grid_times']) == expected_dense_len, "Dense grid has incorrect number of points"

def test_state_interpolation_shapes(batch_and_processor):
    """Validates the shapes of the interpolated state tensors."""
    batch, _ = batch_and_processor
    b, s_gt = batch['is_gt_union'].shape
    
    assert b == 2
    assert batch['loc_emb_union'].shape[:2] == (b, s_gt)
    assert batch['purp_emb_union'].shape[:2] == (b, s_gt)
    assert batch['anchor_union'].shape == (b, s_gt)
    assert batch['loc_emb_union'].dtype == torch.float32

def test_flat_stays(batch_and_processor):
    """Asserts that embeddings are flat during stay periods on the union grid."""
    batch, processor = batch_and_processor
    gt_union = batch['gt_union_times']
    
    p1_data = processor.get_data(1)
    start_time, end_time = p1_data['segments'][0]['t0'], p1_data['segments'][0]['t1'] # First travel
    
    stay_start_time = p1_data['gt_times'][0]
    stay_end_time = start_time # End of first stay is start of first travel

    start_idx = torch.searchsorted(gt_union, stay_start_time).item()
    end_idx = torch.searchsorted(gt_union, stay_end_time).item()
    
    for j in range(start_idx, end_idx + 1):
        assert torch.allclose(batch['loc_emb_union'][0, j], p1_data['gt_loc_emb'][0], atol=1e-6)

def test_ragged_segments(batch_and_processor):
    """Validates the structure and indices of the ragged segments batch."""
    batch, processor = batch_and_processor
    
    assert isinstance(batch['segments_batch'], list)
    total_segs = len(processor.get_data(1)['segments']) + len(processor.get_data(2)['segments'])
    assert len(batch['segments_batch']) == total_segs
    
    for seg in batch['segments_batch']:
        assert set(seg.keys()) == {"b", "i0", "i1", "mode_id", "mode_proto"}
        assert batch['is_gt_grid'][seg['i0']] and batch['is_gt_grid'][seg['i1']]
        assert seg['i0'] < seg['i1']

def test_final_summary(batch_and_processor):
    """Prints a summary of the batch structure if all tests pass."""
    batch, _ = batch_and_processor
    b, s_gt = batch['is_gt_union'].shape
    s_dense = len(batch['grid_times'])
    
    print(f"\nOK: B={b} S_gt={s_gt} S_dense={s_dense} | "
          f"Total Segments: {len(batch['segments_batch'])}")
