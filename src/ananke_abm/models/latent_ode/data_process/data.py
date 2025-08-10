"""
Data processing for the Generative Latent ODE model.
"""
import torch
import torch.nn.functional as F
import networkx as nx

from ananke_abm.data_generator.mock_2p import (
    create_training_data_single_person,
    create_sarah_daily_pattern,
    create_marcus_daily_pattern,
    create_sarah, create_marcus
)
from ananke_abm.data_generator.mock_locations import create_mock_zone_graph
from ananke_abm.data_generator.feature_engineering import (
    get_purpose_features,
    get_mode_features,
    PURPOSE_ID_MAP,
    MODE_ID_MAP,
)
from ananke_abm.models.latent_ode.config import GenerativeODEConfig
from torch.utils.data import Dataset

class LatentODEDataset(Dataset):
    """PyTorch Dataset to provide individual samples for the DataLoader."""
    def __init__(self, person_ids, processor):
        self.person_ids = person_ids
        self.processor = processor

    def __len__(self):
        return len(self.person_ids)

    def __getitem__(self, idx):
        person_id = self.person_ids[idx]
        data = self.processor.get_data(person_id)
        data['config'] = self.processor.config
        return data

class DataProcessor:
    """Processes mock data, preparing it for the Generative ODE model."""

    def __init__(self, device, config: GenerativeODEConfig):
        self.device = device
        self.config = config
        
        # Load the static graph and distance matrix once
        self.zone_graph, self.zones_raw, self.distance_matrix = create_mock_zone_graph()
        self.distance_matrix = self.distance_matrix.to(device)

        # --- Activity/Purpose Processing ---
        # Define the mapping from detailed activities to broader categories
        self.activity_to_group = {
            # Home activities
            "sleep": "home", "morning_routine": "home", "evening": "home", 
            "dinner": "home", "arrive_home": "home",
            # Work/Education
            "work": "work", "arrive_work": "work", "end_work": "work",
            # Subsistence
            "lunch": "shopping", "lunch_start": "shopping", "lunch_end": "shopping", # Simplified to shopping
            # Leisure & Recreation
            "gym": "social", "gym_end": "social", # Simplified to social
            "exercise": "social", "leaving_park": "social",
            # Social
            "social": "social", "leaving_social": "social", "dinner_social": "social",
            # Travel/Transit
            "prepare_commute": "travel", "start_commute": "travel",
            "transit": "travel", "leaving_home": "travel",
            "break": "travel",
        }
        self.purpose_map = PURPOSE_ID_MAP
        
        # --- Mode Processing ---
        # Define the mapping from travel modes to mode IDs
        self.mode_map = MODE_ID_MAP

    def get_data(self, person_id):
        """
        Processes mock data for a single person, resampling it to a uniform time grid.
        Now includes sequences of purpose IDs and mode IDs for loss calculation,
        as well as rich feature vectors for model input.
        """
        if person_id == 1:
            schedule = create_sarah_daily_pattern()
            person_obj = create_sarah()
        else:
            schedule = create_marcus_daily_pattern()
            person_obj = create_marcus()
            
        data = create_training_data_single_person(
            person=person_obj,
            schedule=schedule,
            zone_graph=self.zone_graph,
            repeat_pattern=False
        )

        activities = data["activities"]
        travel_modes = data["travel_modes"]
        
        # --- Create the target sequence of purpose IDs ---
        target_purpose_ids = torch.tensor(
            [self.purpose_map[self.activity_to_group.get(act, "travel")] for act in activities],
            dtype=torch.long
        ).to(self.device)
        
        # --- Create the target sequence of mode IDs ---
        target_mode_ids = torch.tensor(
            [self.mode_map.get(mode.lower(), self.mode_map["stay"]) for mode in travel_modes],
            dtype=torch.long
        ).to(self.device)

        # --- Create rich feature vectors from IDs ---
        target_purpose_features = torch.stack([get_purpose_features(pid.item()) for pid in target_purpose_ids]).to(self.device)
        target_mode_features = torch.stack([get_mode_features(mid.item()) for mid in target_mode_ids]).to(self.device)

        # Convert importance strings to numerical weights
        importance_weights = [
            self.config.anchor_loss_weight if imp == 'anchor' else 1.0
            for imp in data['importances']
        ]
        
        zone_features = data['zone_features'].to(self.device)
        home_zone_features = zone_features[data['home_zone_id']]
        work_zone_features = zone_features[data['work_zone_id']]
        
        adjacency_matrix = torch.tensor(nx.to_numpy_array(self.zone_graph), dtype=torch.float32, device=self.device)
        adjacency_matrix.fill_diagonal_(1)

        return {
            "person_features": data["person_attrs"].to(self.device),
            "times": data["times"].to(self.device),
            "trajectory_y": data["zone_observations"].to(self.device),
            "target_purpose_ids": target_purpose_ids,
            "target_mode_ids": target_mode_ids,
            "target_purpose_features": target_purpose_features,
            "target_mode_features": target_mode_features,
            "importance_weights": torch.tensor(importance_weights, dtype=torch.float32, device=self.device),
            "num_zones": len(self.zones_raw),
            "person_name": data['person_name'],
            "home_zone_features": home_zone_features,
            "work_zone_features": work_zone_features,
            "all_zone_features": zone_features,
        }
