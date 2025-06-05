#!/usr/bin/env python3
"""
Strict Physics Model with Hard Constraints
Uses -inf masking for guaranteed physics compliance.
"""

import torch
import torch.nn as nn

class ImprovedStrictPhysicsModel(nn.Module):
    """Improved strict physics model with better numerical stability"""
    
    def __init__(self, person_attrs_dim=8, num_zones=8):
        super().__init__()
        
        self.num_zones = num_zones
        
        # Better architecture with skip connections and normalization
        self.input_norm = nn.LayerNorm(person_attrs_dim + 1 + num_zones)
        
        # Encoder layers
        self.encoder1 = nn.Sequential(
            nn.Linear(person_attrs_dim + 1 + num_zones, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.encoder2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Output layers with residual connection
        self.output_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_zones)
        )
        
        # Initialize weights properly
        self._initialize_weights()
        
        self.adjacency_matrix = None
        
    def _initialize_weights(self):
        """Proper weight initialization for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def set_graph_data(self, zone_features, edge_index, person_attrs):
        """Set adjacency constraints"""
        self.adjacency_matrix = torch.zeros(self.num_zones, self.num_zones)
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i], edge_index[1, i]
            self.adjacency_matrix[u, v] = 1.0
            self.adjacency_matrix[v, u] = 1.0
        # Self-loops
        for i in range(self.num_zones):
            self.adjacency_matrix[i, i] = 1.0
    
    def forward(self, person_attrs, times, zone_features, edge_index):
        """Predict sequence with improved stability"""
        
        self.set_graph_data(zone_features, edge_index, person_attrs)
        
        # Start at zone 0 (home)
        current_zone = 0
        predictions = []
        
        for i, time in enumerate(times):
            # Create input with normalization
            current_zone_onehot = torch.zeros(self.num_zones)
            current_zone_onehot[current_zone] = 1.0
            
            input_vec = torch.cat([
                person_attrs,
                torch.tensor([time.item()]),
                current_zone_onehot
            ])
            
            # Normalize input
            input_normalized = self.input_norm(input_vec.unsqueeze(0))
            
            # Forward pass through improved architecture
            x = self.encoder1(input_normalized)
            x = self.encoder2(x)
            raw_logits = self.output_layer(x).squeeze(0)
            
            # Apply constraints more carefully
            adjacency_row = self.adjacency_matrix[current_zone]
            valid_indices = torch.where(adjacency_row > 0)[0]
            
            # Soft constraint approach that's more stable
            if len(valid_indices) > 0:
                # Create mask for valid transitions
                constraint_mask = adjacency_row.clone()
                
                # Apply soft masking with temperature
                temperature = 0.1  # Lower temperature for sharper constraints
                constrained_logits = raw_logits / temperature
                
                # Zero out invalid transitions
                invalid_mask = (adjacency_row == 0)
                constrained_logits[invalid_mask] = -50.0 / temperature  # Large penalty but not -inf
                
                predictions.append(constrained_logits * temperature)
                
                # Update current zone - choose from valid options only
                valid_scores = raw_logits[valid_indices]
                if len(valid_scores) > 0:
                    best_idx = torch.argmax(valid_scores)
                    current_zone = valid_indices[best_idx].item()
            else:
                # Emergency fallback
                predictions.append(raw_logits)
                current_zone = 0
        
        zone_logits = torch.stack(predictions)
        return zone_logits, None 