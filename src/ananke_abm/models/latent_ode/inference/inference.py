"""
Inference module for the SDE model with segment-based mode prediction.
"""
import torch
import numpy as np
from typing import List, Dict, Optional

from ananke_abm.models.latent_ode.config import GenerativeODEConfig
from ananke_abm.models.latent_ode.data_process.data import DataProcessor
from ananke_abm.models.latent_ode.architecture.model import GenerativeODE

STAY_VELOCITY_THRESHOLD = 0.1

class InferenceEngine:
    def __init__(self, model_path: str, config: Optional[GenerativeODEConfig] = None, device: str = "auto"):
        self.config = config or GenerativeODEConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else "cpu")
        self.processor = DataProcessor(self.device)
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        sample_data = self.processor.get_data(person_id=1)
        person_feat_dim = 8 # Placeholder
        
        self.model = GenerativeODE(
            person_feat_dim=person_feat_dim,
            num_zone_features=sample_data['gt_loc_emb'].shape[-1],
            config=self.config,
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict_trajectories(self, person_ids: List[int], time_resolution: int = 500) -> List[Dict]:
        """Generates full itineraries (stays and trips) for a list of people."""
        itineraries = []
        with torch.no_grad():
            for person_id in person_ids:
                data = self.processor.get_data(person_id)
                times = torch.linspace(0, 24 * 60, time_resolution).to(self.device)
                
                # --- SDE Forward Pass ---
                model_outputs = self.model(
                    person_features=torch.randn(1, 8, device=self.device), # Placeholder
                    home_zone_features=data['gt_loc_emb'][0].unsqueeze(0),
                    work_zone_features=data['gt_loc_emb'][0].unsqueeze(0), # Simplification
                    initial_purpose_features=data['gt_purp_emb'][0].unsqueeze(0),
                    times=times,
                    all_zone_features=data['gt_loc_emb'] # Simplification
                )
                pred_p, pred_v = model_outputs[4][0], model_outputs[5][0]

                # --- Itinerary Assembly ---
                itinerary = self._assemble_itinerary(pred_p, pred_v, times)
                itineraries.append({"person_id": person_id, "itinerary": itinerary})
        return itineraries

    def _assemble_itinerary(self, p_path, v_path, times):
        """Identifies stays and trips from a latent path."""
        velocities = torch.norm(v_path, p=2, dim=-1)
        is_staying = (velocities < STAY_VELOCITY_THRESHOLD).cpu().numpy()
        
        itinerary = []
        current_segment = None

        for t_idx in range(len(times)):
            if is_staying[t_idx]:
                if current_segment is None or current_segment['type'] == 'travel':
                    # Start of a new stay
                    if current_segment is not None: itinerary.append(current_segment)
                    current_segment = {'type': 'stay', 'start_time': times[t_idx].item(), 'path_indices': [t_idx]}
                else:
                    # Continue existing stay
                    current_segment['path_indices'].append(t_idx)
            else: # Traveling
                if current_segment is None or current_segment['type'] == 'stay':
                    # Start of a new travel
                    if current_segment is not None: itinerary.append(current_segment)
                    current_segment = {'type': 'travel', 'start_time': times[t_idx].item(), 'path_indices': [t_idx]}
                else:
                    # Continue existing travel
                    current_segment['path_indices'].append(t_idx)
        
        if current_segment is not None:
            itinerary.append(current_segment)
        
        # Finalize segments
        for seg in itinerary:
            seg['end_time'] = times[seg['path_indices'][-1]].item()
            path_slice = p_path[seg['path_indices']]
            
            if seg['type'] == 'stay':
                # For stays, determine location and purpose from average embedding
                avg_embedding = path_slice.mean(dim=0)
                loc_emb, purp_emb = torch.split(avg_embedding, [self.config.zone_embed_dim, self.config.purpose_feature_dim])
                # Note: Reverse mapping from embedding to label is non-trivial, returning embeddings for now
                seg['location_embedding'] = loc_emb.cpu().numpy()
                seg['purpose_embedding'] = purp_emb.cpu().numpy()
            else: # Travel
                # For travel, predict mode
                v_slice = v_path[seg['path_indices']]
                t_slice = times[seg['path_indices']]
                logits, h = self.model.mode_predictor(path_slice.unsqueeze(0), v_slice.unsqueeze(0), t_slice)
                seg['mode_id'] = torch.argmax(logits, dim=-1).item()
                seg['mode_embedding'] = h.squeeze(0).cpu().numpy()

        return itinerary
