"""
Data processing for the Generative Latent ODE model.
"""
import torch
import torch.nn.functional as F

from ananke_abm.data_generator.mock_2p import (
    create_training_data_single_person,
    create_sarah_daily_pattern,
    create_marcus_daily_pattern,
    create_sarah, create_marcus
)
from ananke_abm.data_generator.mock_locations import create_mock_zone_graph
from ananke_abm.models.run.latent_ode.config import GenerativeODEConfig
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
            "sleep": "Home", "morning_routine": "Home", "evening": "Home", 
            "dinner": "Home", "arrive_home": "Home",
            # Work/Education
            "work": "Work/Education", "arrive_work": "Work/Education", "end_work": "Work/Education",
            # Subsistence
            "lunch": "Subsistence", "lunch_start": "Subsistence", "lunch_end": "Subsistence",
            # Leisure & Recreation
            "gym": "Leisure & Recreation", "gym_end": "Leisure & Recreation", 
            "exercise": "Leisure & Recreation", "leaving_park": "Leisure & Recreation",
            # Social
            "social": "Social", "leaving_social": "Social", "dinner_social": "Social",
            # Travel/Transit
            "prepare_commute": "Travel/Transit", "start_commute": "Travel/Transit",
            "transit": "Travel/Transit", "leaving_home": "Travel/Transit",
            "break": "Travel/Transit",
        }
        self.purpose_map = {name: i for i, name in enumerate(self.config.purpose_groups)}

    def get_data(self, person_id):
        """
        Processes mock data for a single person, resampling it to a uniform time grid.
        Now includes a sequence of purpose IDs for loss calculation.
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
        
        # --- Create the target sequence of purpose IDs ---
        # The `activities` list directly corresponds to the `zone_observations`
        target_purpose_ids = torch.tensor(
            [self.purpose_map[self.activity_to_group.get(act, "Travel/Transit")] for act in activities],
            dtype=torch.long
        ).to(self.device)

        # Convert importance strings to numerical weights
        importance_weights = [
            self.config.anchor_loss_weight if imp == 'anchor' else 1.0
            for imp in data['importances']
        ]

        return {
            "person_features": data["person_attrs"].to(self.device),
            "times": data["times"].to(self.device),
            "trajectory_y": data["zone_observations"].to(self.device),
            "target_purpose_ids": target_purpose_ids,
            "importance_weights": torch.tensor(importance_weights, dtype=torch.float32, device=self.device),
            "start_purpose_id": target_purpose_ids[0].item(), # Known starting purpose
            "num_zones": len(self.zones_raw),
            "person_name": data['person_name'],
            "home_zone_id": data["home_zone_id"],
            "work_zone_id": data["work_zone_id"],
        } 