"""
Tests for the data processing and batching pipeline.
"""
import pytest
import torch
import sys
from pathlib import Path
from torch.utils.data import DataLoader

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ananke_abm.models.latent_ode.data_process.data import LatentSDEDataset, DataProcessor
from src.ananke_abm.models.latent_ode.data_process.batching import sde_collate_fn


@pytest.fixture(scope="module")
def test_data():
    """Fixture to load test data and prepare a batch."""
    device = torch.device("cpu")
    periods_path = "test/test_periods_small.csv"
    snaps_path = "test/test_snaps_small.csv"

    processor = DataProcessor(device, periods_path=periods_path, snaps_path=snaps_path)
    dataset = LatentSDEDataset(person_ids=[1, 2], processor=processor)
    
    # Use a DataLoader to create a batch
    loader = DataLoader(dataset, batch_size=2, collate_fn=sde_collate_fn)
    batch = next(iter(loader))
    
    return batch, processor

def test_batch_creation_and_shapes(test_data):
    """Test the basic properties and shapes of the created batch."""
    batch, _ = test_data
    
    # Time assertions
    assert torch.all(batch['grid_times'] >= 0) and torch.all(batch['grid_times'] < 24 * 60)
    assert torch.all(batch['grid_times'][1:] > batch['grid_times'][:-1])
    
    # Shape assertions
    b, s = batch['is_gt_batch'].shape
    assert b == 2
    assert batch['loc_emb_batch'].shape[0] == b and batch['loc_emb_batch'].shape[1] == s
    assert batch['purp_emb_batch'].shape[0] == b and batch['purp_emb_batch'].shape[1] == s
    assert batch['anchor_mask_batch'].shape == (b, s)

    # No ID leak assertion
    assert 'loc_id' not in batch and 'zone_id' not in batch
    assert batch['loc_emb_batch'].dtype == torch.float32
    assert batch['purp_emb_batch'].dtype == torch.float32

def test_gt_and_anchor_masks(test_data):
    """Test the ground-truth and anchor masks."""
    batch, processor = test_data
    
    gt_counts = [torch.sum(batch['is_gt_batch'][i]).item() for i in range(2)]
    anchor_counts = [torch.sum(batch['anchor_mask_batch'][i]).item() for i in range(2)]

    expected_gt_counts = [len(processor.get_data(i+1)['gt_times']) for i in range(2)]
    assert gt_counts == expected_gt_counts
    
    # Anchor assertions
    assert all(count == 2 for count in anchor_counts)
    assert torch.all((batch['anchor_mask_batch'] == 0) | (batch['is_gt_batch'] == 1))

def test_stay_interpolation(test_data):
    """Test that embeddings are constant during stay periods."""
    batch, processor = test_data
    grid_times = batch['grid_times']

    for i in range(2): # For each person
        person_id = i + 1
        data = processor.get_data(person_id)
        
        # Check first stay period (home)
        start_time, end_time = data['gt_times'][0], data['gt_times'][1]
        start_idx = torch.searchsorted(grid_times, start_time).item()
        end_idx = torch.searchsorted(grid_times, end_time).item()

        for t_idx in range(start_idx, end_idx + 1):
            assert torch.allclose(batch['loc_emb_batch'][i, t_idx], data['gt_loc_emb'][0], atol=1e-6)
            assert torch.allclose(batch['purp_emb_batch'][i, t_idx], data['gt_purp_emb'][0], atol=1e-6)

def test_travel_interpolation(test_data):
    """Test the linear interpolation during travel periods."""
    batch, processor = test_data
    grid_times = batch['grid_times']

    # Person 1: home -> work
    p1_data = processor.get_data(1)
    t_prev, t_next = p1_data['gt_times'][1], p1_data['gt_times'][2] # 9:00 and 9:30
    tm = (t_prev + t_next) / 2
    
    tm_idx = torch.searchsorted(grid_times, tm)
    
    time_gap = t_next - t_prev
    w_prev = (t_next - grid_times[tm_idx]) / time_gap
    w_next = 1 - w_prev

    expected_loc_emb = w_prev * p1_data['gt_loc_emb'][1] + w_next * p1_data['gt_loc_emb'][2]
    expected_purp_emb = w_prev * p1_data['gt_purp_emb'][1] + w_next * p1_data['gt_purp_emb'][2]

    assert torch.allclose(batch['loc_emb_batch'][0, tm_idx], expected_loc_emb, atol=1e-6)
    assert torch.allclose(batch['purp_emb_batch'][0, tm_idx], expected_purp_emb, atol=1e-6)


def test_output_summary(test_data):
    """Prints a summary if all other tests pass."""
    batch, _ = test_data
    b, s, d_loc = batch['loc_emb_batch'].shape
    d_p = batch['purp_emb_batch'].shape[2]
    gt_per_person = [int(torch.sum(batch['is_gt_batch'][i]).item()) for i in range(b)]
    anchors_per_person = [int(torch.sum(batch['anchor_mask_batch'][i]).item()) for i in range(b)]
    
    print(f"\nOK: B={b} S={s} D_loc={d_loc} D_p={d_p} | "
          f"GT per person: {gt_per_person} | "
          f"anchors per person: {anchors_per_person}")

