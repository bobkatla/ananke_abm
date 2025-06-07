#!/usr/bin/env python3
"""
Hybrid Physics Model - Soft Training + Hard Inference
Combines soft training with hard inference for best of both worlds.
"""

import torch
import torch.nn as nn

class HybridPhysicsModel(nn.Module):
    """Combines soft training with hard inference - best of both worlds"""
    
    def __init__(self, person_attrs_dim=8, num_zones=8):
        super().__init__()
        
        self.num_zones = num_zones
        
        # Use the lightweight diffusion architecture (easy training)
        self.path_generator = nn.Sequential(
            nn.Linear(person_attrs_dim + 1, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_zones)
        )
        
        self.adjacency_matrix = None
        
    def set_graph_data(self, zone_features, edge_index, person_attrs):
        """Set adjacency constraints"""
        self.adjacency_matrix = torch.zeros(self.num_zones, self.num_zones)
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i], edge_index[1, i]
            self.adjacency_matrix[u, v] = 1.0
            self.adjacency_matrix[v, u] = 1.0
        for i in range(self.num_zones):
            self.adjacency_matrix[i, i] = 1.0
    
    def forward(self, person_attrs, times, zone_features, edge_index, training=True):
        """Training uses soft constraints, inference uses hard constraints"""
        
        self.set_graph_data(zone_features, edge_index, person_attrs)
        
        predictions = []
        current_zone = 0
        
        for i, time in enumerate(times):
            # Input: person + time (lightweight)
            input_vec = torch.cat([person_attrs, torch.tensor([time.item()])])
            
            # Generate raw logits
            raw_logits = self.path_generator(input_vec.unsqueeze(0)).squeeze(0)
            
            if training:
                # SOFT CONSTRAINTS during training (allows gradient flow)
                adjacency_row = self.adjacency_matrix[current_zone]
                penalty = 10.0
                constrained_logits = raw_logits - penalty * (1 - adjacency_row)
                predictions.append(constrained_logits)
                
                # Update with soft prediction
                current_zone = torch.argmax(constrained_logits).item()
            else:
                # HARD CONSTRAINTS during inference (guarantees validity)
                adjacency_row = self.adjacency_matrix[current_zone]
                valid_indices = torch.where(adjacency_row > 0)[0]
                
                if len(valid_indices) > 0:
                    # Hard masking: only valid zones get real scores
                    hard_logits = torch.full((self.num_zones,), -float('inf'))
                    hard_logits[valid_indices] = raw_logits[valid_indices]
                    predictions.append(hard_logits)
                    
                    # Guaranteed valid next zone
                    valid_scores = raw_logits[valid_indices]
                    best_idx = torch.argmax(valid_scores)
                    current_zone = valid_indices[best_idx].item()
                else:
                    predictions.append(raw_logits)
                    current_zone = 0
        
        zone_logits = torch.stack(predictions)
        return zone_logits, None 