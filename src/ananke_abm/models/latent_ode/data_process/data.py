"""
Data processing for the Generative Latent ODE model.
"""
import torch
import pandas as pd
from ananke_abm.data_generator.feature_engineering import (
    get_purpose_features,
    PURPOSE_ID_MAP,
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
    """Processes mock data, preparing it for the Generative SDE model."""

    def __init__(self, device, periods_path="data/periods.csv", snaps_path="data/snaps.csv"):
        self.device = device
        self.person_data = {}
        self._load_and_process_data(periods_path, snaps_path)

    def _get_location_embeddings(self):
        _, zones_raw, _ = create_mock_zone_graph()
        location_to_embedding = {}
        for zone_id, zone_data in zones_raw.items():
            # Normalize features as done in the original data generation
            features = [
                zone_data["population"] / 10000.0,
                zone_data["job_opportunities"] / 5000.0,
                zone_data["retail_accessibility"],
                zone_data["transit_accessibility"],
                zone_data["attractiveness"],
                zone_data["coordinates"][0] / 5.0,
                zone_data["coordinates"][1] / 5.0,
            ]
            location_to_embedding[zone_data["name"]] = torch.tensor(
                features, dtype=torch.float32
            )
        return location_to_embedding

    def _get_purpose_embeddings(self):
        purpose_to_embedding = {}
        for name, pid in PURPOSE_ID_MAP.items():
            purpose_to_embedding[name] = get_purpose_features(pid)
        return purpose_to_embedding

    def _load_and_process_data(self, periods_path, snaps_path):
        """Loads, validates, and processes CSV data for all persons."""
        periods_df = pd.read_csv(periods_path)
        snaps_df = pd.read_csv(snaps_path)

        location_to_embedding = self._get_location_embeddings()
        purpose_to_embedding = self._get_purpose_embeddings()

        for person_id in periods_df["person_id"].unique():
            person_periods = periods_df[periods_df["person_id"] == person_id].sort_values(
                by="start_time"
            )
            person_snaps = snaps_df[snaps_df["person_id"] == person_id].sort_values(
                by="timestamp"
            )

            # --- Validation ---
            assert person_periods.iloc[0]["type"] == "stay", f"Person {person_id} does not start with a stay."
            assert person_periods.iloc[-1]["type"] == "stay", f"Person {person_id} does not end with a stay."
            assert all(
                person_periods["type"].iloc[i] != person_periods["type"].iloc[i + 1]
                for i in range(len(person_periods) - 1)
            ), f"Person {person_id} has consecutive periods of the same type."

            for _, period in person_periods[person_periods["type"] == "stay"].iterrows():
                stay_snaps = person_snaps[
                    (person_snaps["timestamp"] == period["start_time"])
                    | (person_snaps["timestamp"] == period["end_time"])
                ]
                assert len(stay_snaps) == 2, f"Stay period at {period['start_time']} for person {person_id} does not have 2 snaps."
                assert stay_snaps.iloc[0]["location"] == period["location"], "Snap location mismatch."

            # --- Feature Engineering ---
            # Convert times to minutes since midnight
            gt_times = torch.tensor(person_snaps["timestamp"].values * 60, dtype=torch.float32)

            gt_loc_emb = torch.stack(
                [
                    location_to_embedding[loc]
                    for loc in person_snaps["location"]
                ]
            )
            gt_purp_emb = torch.stack(
                [
                    purpose_to_embedding[purp]
                    for purp in person_snaps["purpose"]
                ]
            )
            gt_anchor = torch.tensor(person_snaps["anchor"].values, dtype=torch.float32)

            self.person_data[person_id] = {
                "gt_times": gt_times.to(self.device),
                "gt_loc_emb": gt_loc_emb.to(self.device),
                "gt_purp_emb": gt_purp_emb.to(self.device),
                "gt_anchor": gt_anchor.to(self.device),
                "person_id": person_id,
            }

    def get_data(self, person_id):
        """Returns the processed data for a single person."""
        return self.person_data[person_id]
