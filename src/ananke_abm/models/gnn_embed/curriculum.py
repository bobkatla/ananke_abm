#!/usr/bin/env python3
"""
Curriculum Physics Model - Progressive Hardening
Gradually transitions from soft to hard constraints during training.
"""

import torch
import torch.nn as nn

class CurriculumPhysicsModel(nn.Module):
    """Gradually transitions from soft to hard constraints during training"""
    
    def __init__(self, person_attrs_dim=8, num_zones=8):
        super().__init__()
        
        self.num_zones = num_zones
        self.training_step = 0
        self.max_training_steps = 8000 * 22  # epochs * samples_per_epoch
        
        # Lightweight architecture
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
    
    def get_constraint_strength(self):
        """Progressive hardening: start soft, end hard"""
        progress = min(self.training_step / self.max_training_steps, 1.0)
        
        # Exponential schedule: starts at 1.0, ends at 100.0
        min_penalty, max_penalty = 1.0, 100.0
        penalty = min_penalty * (max_penalty / min_penalty) ** progress
        
        return penalty
    
    def forward(self, person_attrs, times, zone_features, edge_index):
        """Progressive constraint hardening during training"""
        
        self.set_graph_data(zone_features, edge_index, person_attrs)
        
        predictions = []
        current_zone = 0
        
        # Get current constraint strength
        penalty = self.get_constraint_strength()
        
        for i, time in enumerate(times):
            input_vec = torch.cat([person_attrs, torch.tensor([time.item()])])
            raw_logits = self.path_generator(input_vec.unsqueeze(0)).squeeze(0)
            
            # Progressive constraint application
            adjacency_row = self.adjacency_matrix[current_zone]
            constrained_logits = raw_logits - penalty * (1 - adjacency_row)
            predictions.append(constrained_logits)
            
            # Update current zone
            current_zone = torch.argmax(constrained_logits).item()
            
        self.training_step += 1
        
        zone_logits = torch.stack(predictions)
        return zone_logits, None 