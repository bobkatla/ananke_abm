#!/usr/bin/env python3
"""
Simplified Diffusion Model with Soft Constraints
Lightweight architecture with penalty-based physics enforcement.
"""

import torch
import torch.nn as nn

class SimplifiedDiffusionModel(nn.Module):
    """Simplified diffusion model that maintains gradients properly"""
    
    def __init__(self, person_attrs_dim=8, num_zones=8):
        super().__init__()
        
        self.num_zones = num_zones
        
        # Path generator network
        self.path_generator = nn.Sequential(
            nn.Linear(person_attrs_dim + 1, 128),  # person + time
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
    
    def forward(self, person_attrs, times, zone_features, edge_index):
        """Generate physics-constrained path with proper gradients"""
        
        self.set_graph_data(zone_features, edge_index, person_attrs)
        
        predictions = []
        current_zone = 0  # Start at home
        
        for i, time in enumerate(times):
            # Input: person attributes + time
            input_vec = torch.cat([person_attrs, torch.tensor([time.item()])])
            
            # Generate raw logits
            raw_logits = self.path_generator(input_vec.unsqueeze(0)).squeeze(0)
            
            # Apply physics constraints using soft masking (preserves gradients)
            adjacency_row = self.adjacency_matrix[current_zone]
            
            # Soft constraint: large penalty instead of -inf
            penalty = 10.0
            constrained_logits = raw_logits - penalty * (1 - adjacency_row)
            
            predictions.append(constrained_logits)
            
            # Update current zone for next step
            valid_indices = torch.where(adjacency_row > 0)[0]
            if len(valid_indices) > 0:
                valid_scores = raw_logits[valid_indices]
                best_idx = torch.argmax(valid_scores)
                current_zone = valid_indices[best_idx].item()
        
        zone_logits = torch.stack(predictions)
        return zone_logits, None 