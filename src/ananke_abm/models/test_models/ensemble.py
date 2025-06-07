#!/usr/bin/env python3
"""
Ensemble Physics Model - Combined Predictions
Combines predictions from both soft and hard models with learned weights.
"""

import torch
import torch.nn as nn
from .diffusion import SimplifiedDiffusionModel
from .strict_physics import ImprovedStrictPhysicsModel

class EnsemblePhysicsModel(nn.Module):
    """Combines predictions from both soft and hard models"""
    
    def __init__(self, person_attrs_dim=8, num_zones=8):
        super().__init__()
        
        # Soft model (diffusion-style)
        self.soft_model = SimplifiedDiffusionModel(person_attrs_dim, num_zones)
        
        # Hard model (strict-style)  
        self.hard_model = ImprovedStrictPhysicsModel(person_attrs_dim, num_zones)
        
        # Learned combination weights
        self.combination_weights = nn.Parameter(torch.tensor([0.7, 0.3]))  # Start favoring soft
        
    def forward(self, person_attrs, times, zone_features, edge_index):
        """Ensemble prediction with learned weights"""
        
        # Get predictions from both models
        soft_logits, _ = self.soft_model(person_attrs, times, zone_features, edge_index)
        hard_logits, _ = self.hard_model(person_attrs, times, zone_features, edge_index)
        
        # Handle -inf values in hard logits
        hard_logits_clean = hard_logits.clone()
        hard_logits_clean[torch.isinf(hard_logits_clean)] = -100.0
        
        # Learned weighted combination
        weights = torch.softmax(self.combination_weights, dim=0)
        combined_logits = weights[0] * soft_logits + weights[1] * hard_logits_clean
        
        return combined_logits, None 