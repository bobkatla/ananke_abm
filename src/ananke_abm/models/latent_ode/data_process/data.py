"""
Data processing for the Generative SDE model, providing snap-based state
and segment-based travel information.
"""
import torch
import pandas as pd
import numpy as np
from ananke_abm.data_generator.feature_engineering import (
    get_purpose_features,
    get_mode_features,
    PURPOSE_ID_MAP,
    MODE_ID_MAP,
)
from ananke_abm.data_generator.mock_locations import create_mock_zone_graph
from torch.utils.data import Dataset


class LatentSDEDataset(Dataset):
    """PyTorch Dataset to provide individual samples for the DataLoader."""

    def __init__(self, person_ids, processor):
        self.person_ids = person_ids
        self.processor = processor

    def __len__(self):
        return len(self.person_ids)

    def __getitem__(self, idx):
        person_id = self.person_ids[idx]
        return self.processor.get_data(person_id)


class DataProcessor:
    """Processes CSV data to produce snap and segment tensors for the SDE model."""

    def __init__(self, device, location_to_embedding, location_name_to_id, periods_path="data/periods.csv", snaps_path="data/snaps.csv"):
        self.device = device
        self.person_data = {}
        self.location_embeddings = torch.stack(list(location_to_embedding.values())).to(self.device)
        self.location_name_to_id = location_name_to_id
        self._load_and_process_data(periods_path, snaps_path, location_to_embedding)

    def _get_purpose_embeddings(self):
        return {name: get_purpose_features(name) for name in PURPOSE_ID_MAP.keys()}
    
    def _get_mode_embeddings(self):
        return {name: get_mode_features(mid) for name, mid in MODE_ID_MAP.items()}

    def _load_and_process_data(self, periods_path, snaps_path, location_to_embedding):
        """Loads, validates, and processes CSV data for all persons."""
        periods_df = pd.read_csv(periods_path)
        snaps_df = pd.read_csv(snaps_path)

        purpose_to_embedding = self._get_purpose_embeddings()
        mode_to_embedding = self._get_mode_embeddings()

        for person_id in periods_df["person_id"].unique():
            person_periods = periods_df[periods_df["person_id"] == person_id].sort_values(by="start_time")
            
            # --- Snaps Processing ---
            person_snaps = snaps_df[snaps_df["person_id"] == person_id].sort_values(by="timestamp")
            gt_times_minutes = torch.tensor(np.round(person_snaps["timestamp"].values * 60, 2), dtype=torch.float32)
            gt_loc_emb = torch.stack([location_to_embedding[loc] for loc in person_snaps["location"]])
            gt_loc_ids = torch.tensor([self.location_name_to_id[loc] for loc in person_snaps["location"]], dtype=torch.long)
            gt_purp_emb = torch.stack([purpose_to_embedding[purp] for purp in person_snaps["purpose"]])
            gt_purp_ids = torch.tensor([PURPOSE_ID_MAP[purp] for purp in person_snaps["purpose"]], dtype=torch.long)
            gt_anchor = torch.tensor(person_snaps["anchor"].values, dtype=torch.float32)

            # --- Segments Processing ---
            travel_periods = person_periods[person_periods["type"] == "travel"]
            segments = []
            time_to_snap_idx = {round(t.item(), 2): i for i, t in enumerate(gt_times_minutes)}

            for _, period in travel_periods.iterrows():
                t0 = round(period["start_time"] * 60, 2)
                t1 = round(period["end_time"] * 60, 2)
                
                assert t0 in time_to_snap_idx, f"Segment start time {t0} not in snaps for person {person_id}"
                assert t1 in time_to_snap_idx, f"Segment end time {t1} not in snaps for person {person_id}"

                segments.append({
                    "t0": t0,
                    "t1": t1,
                    "mode_id": MODE_ID_MAP[period["mode"]],
                    "mode_proto": mode_to_embedding[period["mode"]],
                    "snap_i0": time_to_snap_idx[t0],
                    "snap_i1": time_to_snap_idx[t1],
                })

            self.person_data[person_id] = {
                "gt_times": gt_times_minutes.to(self.device),
                "gt_loc_emb": gt_loc_emb.to(self.device),
                "gt_loc_ids": gt_loc_ids.to(self.device),
                "gt_purp_emb": gt_purp_emb.to(self.device),
                "gt_purp_ids": gt_purp_ids.to(self.device),
                "gt_anchor": gt_anchor.to(self.device),
                "segments": segments,
                "person_id": person_id,
            }

    def get_data(self, person_id):
        return self.person_data[person_id]
