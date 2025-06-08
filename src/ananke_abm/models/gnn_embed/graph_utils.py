#!/usr/bin/env python3
"""
Utilities for preparing batched graph data for GNN models.
"""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Batch

from ananke_abm.data_generator.load_data import load_mobility_data

def resample_trajectories_to_common_time(
    trajectory_dict: Dict, common_times: np.ndarray
) -> Dict:
    """Interpolates individual trajectories onto a common time vector."""
    resampled_trajectories = {}
    for person_id, traj in trajectory_dict.items():
        # Use numpy for efficient interpolation
        original_times = traj["times"].numpy()
        original_zones = traj["zones"].numpy()
        
        # Interpolate using 'previous' value to fill, which is standard for discrete states
        interp_zones = np.interp(common_times, original_times, original_zones, left=original_zones[0], right=original_zones[-1])
        
        resampled_trajectories[person_id] = {
            "times": torch.from_numpy(common_times).float(),
            "zones": torch.from_numpy(interp_zones).long()
        }
    return resampled_trajectories

def prepare_household_batch(trajectories_df: pd.DataFrame, people_df: pd.DataFrame) -> Tuple[Batch, Dict, Dict, torch.Tensor]:
    """
    Prepares a PyTorch Geometric Batch object where each disjoint graph is a household.

    Args:
        trajectories_df: DataFrame of agent movement trajectories.
        people_df: DataFrame of agent attributes, including 'household_id'.

    Returns:
        A tuple containing:
        - batch: A PyG Batch object ready for the model.
        - resampled_trajectories: Trajectory dict with all people on a common time vector.
        - zone_id_to_idx: Mapping from original zone ID to a 0-indexed integer.
        - common_times: The unified time tensor used for resampling.
    """
    print("Preparing household-based graph data...")

    # Create mappings
    all_person_ids = sorted(people_df['person_id'].unique())
    person_id_to_idx = {pid: i for i, pid in enumerate(all_person_ids)}

    all_zone_ids = sorted(trajectories_df['zone_id'].unique())
    zone_id_to_idx = {zid: i for i, zid in enumerate(all_zone_ids)}

    # Group people by household
    households = people_df.groupby('household_id')

    data_list = []
    trajectory_dict = {}

    for household_id, members in households:
        household_person_indices = [person_id_to_idx[pid] for pid in members['person_id']]

        # Create edges between all members of the household (fully connected)
        edge_list = []
        for i in range(len(household_person_indices)):
            for j in range(i + 1, len(household_person_indices)):
                u, v = household_person_indices[i], household_person_indices[j]
                edge_list.append([u, v])
                edge_list.append([v, u]) # Undirected graph

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)

        # For this example, household features are just the household ID
        household_attrs = torch.tensor([household_id], dtype=torch.float32)

        # The node features `x` will be assembled in the runner
        # This Data object just defines the graph structure and household-level features
        graph_data = Data(
            edge_index=edge_index,
            household_attrs=household_attrs,
            num_nodes=len(all_person_ids) # All graphs in batch must know total num nodes
        )
        graph_data.household_id = household_id
        data_list.append(graph_data)

        # Store trajectory data for each person
        for person_id in members['person_id']:
            person_traj = trajectories_df[trajectories_df['person_id'] == person_id]
            trajectory_dict[person_id] = {
                "times": torch.tensor(person_traj['time'].values, dtype=torch.float32),
                "zones": torch.tensor([zone_id_to_idx[z] for z in person_traj['zone_id']], dtype=torch.long)
            }

    # --- Create a unified time vector and resample trajectories ---
    all_times = np.unique(np.concatenate([t['times'].numpy() for t in trajectory_dict.values()]))
    common_times = torch.from_numpy(np.linspace(all_times.min(), all_times.max(), num=max(len(t['times']) for t in trajectory_dict.values()))).float()
    resampled_trajectories = resample_trajectories_to_common_time(trajectory_dict, common_times.numpy())
    # ---

    # Node features (x) for all people
    # For now, it's just their initial zone index
    person_features = []
    for person_id in all_person_ids:
        initial_zone = trajectories_df[trajectories_df['person_id'] == person_id]['zone_id'].iloc[0]
        initial_zone_idx = zone_id_to_idx[initial_zone]
        person_features.append([float(initial_zone_idx)])
    
    x = torch.tensor(person_features, dtype=torch.float32)

    # Add features to the first graph object which will be broadcast by the batching mechanism
    if data_list:
        data_list[0].x = x

    # Create a single batch object from the list of graphs
    batch = Batch.from_data_list(data_list)

    print(f"✅ Created batch with {batch.num_graphs} household(s) and {batch.num_nodes} people.")
    print(f"   Resampled all trajectories to {len(common_times)} common time points.")
    return batch, resampled_trajectories, zone_id_to_idx, common_times

if __name__ == '__main__':
    # Example usage:
    # from ananke_abm.data_generator.load_data import load_mobility_data # Already imported
    
    trajectories_dict, people_df, zones_df = load_mobility_data()

    # Convert the trajectories dict to a single DataFrame
    traj_list = []
    for person_name, data in trajectories_dict.items():
        for t, z in zip(data['times'], data['zones']):
            traj_list.append({
                'person_id': data['person_id'],
                'time': t,
                'zone_id': z
            })
    trajectories_df = pd.DataFrame(traj_list)

    # Make Sarah and Marcus a household
    people_df.loc[people_df['person_id'] == 1, 'household_id'] = 101
    people_df.loc[people_df['person_id'] == 2, 'household_id'] = 101

    batch, trajectories, zone_map, times = prepare_household_batch(trajectories_df, people_df)

    print("\n--- Batch Object ---")
    print(batch)
    print("\n--- Trajectories ---")
    print(trajectories)
    print("\n--- Zone Map ---")
    print(zone_map)
    print(f"\nCommon Times (first 10): {times[:10].numpy()}")
    print(f"\nPerson node features (initial zone index):\n {batch.x}")
    print(f"\nIntra-household edges:\n {batch.edge_index}")
    print(f"\nPerson to household mapping:\n {batch.batch}") 