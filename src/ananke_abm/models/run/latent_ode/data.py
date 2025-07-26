"""
Data processing for the Generative Latent ODE model.
"""
import torch
import torch.nn.functional as F

from ananke_abm.data_generator.mock_2p import create_two_person_training_data
from .config import GenerativeODEConfig

class DataProcessor:
    """Processes mock data, preparing it for the Generative ODE model."""

    def __init__(self, device, config: GenerativeODEConfig):
        self.device = device
        self.config = config
        self.sarah_data, self.marcus_data = create_two_person_training_data(repeat_pattern=False)

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

    def get_data(self, person_id: int):
        data = self.sarah_data if person_id == 1 else self.marcus_data

        # --- Process Purpose Features ---
        activities = data["activities"]
        purpose_counts = torch.zeros(len(self.config.purpose_groups))
        for activity in activities:
            group = self.activity_to_group.get(activity, "Travel/Transit")
            group_idx = self.purpose_map[group]
            purpose_counts[group_idx] += 1
        purpose_features = F.normalize(purpose_counts, p=1, dim=0)

        return {
            "person_features": data["person_attrs"].to(self.device),
            "trajectory_y": data["zone_observations"].to(self.device),
            "times": data["times"].to(self.device),
            "num_zones": data["num_zones"],
            "home_zone_id": data["home_zone_id"],
            "work_zone_id": data["work_zone_id"],
            "person_name": data.get("person_name", f"Person {person_id}"),
            "purpose_features": purpose_features.to(self.device),
        } 