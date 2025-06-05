#!/usr/bin/env python3
"""
Physics-Informed ODE Models for Zone Dynamics
Continuous trajectory modeling with physics constraints.
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint

class PhysicsInformedODE(nn.Module):
    """ODE that models smooth flow along graph edges (physics-informed)"""
    
    def __init__(self, zone_features_dim=7, person_attrs_dim=8, embedding_dim=64, num_zones=8):
        super().__init__()
        
        self.num_zones = num_zones
        self.embedding_dim = embedding_dim
        
        # Fixed zone embeddings - these are the "attractors" in the space
        self.zone_embeddings = nn.Parameter(torch.randn(num_zones, embedding_dim))
        
        # Time encoding
        self.time_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        
        # Person encoding
        self.person_encoder = nn.Sequential(
            nn.Linear(person_attrs_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Flow prediction network - predicts desired movement direction
        self.flow_net = nn.Sequential(
            nn.Linear(embedding_dim + embedding_dim + 16, 64),  # current_pos + person + time
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim)  # Desired velocity vector
        )
        
        # Graph constraints
        self.adjacency_matrix = None
        self.person_attrs = None
        
    def set_graph_data(self, zone_features, edge_index, person_attrs):
        """Set graph constraints"""
        self.person_attrs = person_attrs
        
        # Create adjacency matrix
        self.adjacency_matrix = torch.zeros(self.num_zones, self.num_zones)
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i], edge_index[1, i]
            self.adjacency_matrix[u, v] = 1.0
            self.adjacency_matrix[v, u] = 1.0
        # Self-loops
        for i in range(self.num_zones):
            self.adjacency_matrix[i, i] = 1.0
    
    def get_current_zone(self, state):
        """Find which zone we're currently closest to"""
        distances = torch.norm(self.zone_embeddings - state.unsqueeze(0), dim=1)
        return torch.argmin(distances).item()
    
    def get_allowed_directions(self, current_zone):
        """Get unit vectors pointing toward allowed adjacent zones"""
        current_pos = self.zone_embeddings[current_zone]
        
        # Get allowed adjacent zones
        adjacent_zones = torch.where(self.adjacency_matrix[current_zone] > 0)[0]
        
        # Compute direction vectors to adjacent zones
        directions = []
        for zone_idx in adjacent_zones:
            target_pos = self.zone_embeddings[zone_idx]
            direction = target_pos - current_pos
            # Normalize direction (unit vector)
            if torch.norm(direction) > 1e-6:
                direction = direction / torch.norm(direction)
            directions.append(direction)
        
        return torch.stack(directions) if directions else torch.zeros(1, self.embedding_dim)
    
    def forward(self, t, state):
        """Compute physics-informed velocity: smooth flow along graph edges"""
        
        # Handle time
        if isinstance(t, torch.Tensor):
            t_val = t.item() if t.dim() == 0 else t[0].item()
        else:
            t_val = float(t)
        
        t_tensor = torch.tensor([[t_val]], dtype=torch.float32)
        t_encoded = self.time_encoder(t_tensor).squeeze(0)
        
        # Encode person attributes
        person_encoded = self.person_encoder(self.person_attrs)
        
        # Find current zone
        current_zone = self.get_current_zone(state)
        
        # Get allowed movement directions (physics constraint)
        allowed_directions = self.get_allowed_directions(current_zone)
        
        # Predict desired movement using neural network
        combined_input = torch.cat([state, person_encoded, t_encoded], dim=0)
        desired_velocity = self.flow_net(combined_input.unsqueeze(0)).squeeze(0)
        
        # PROJECT desired velocity onto allowed directions (HARD PHYSICS CONSTRAINT)
        if len(allowed_directions) > 0:
            # Compute projection of desired velocity onto each allowed direction
            projections = torch.matmul(allowed_directions, desired_velocity.unsqueeze(1)).squeeze(1)
            
            # Use weighted combination of allowed directions
            weights = torch.softmax(projections, dim=0)
            constrained_velocity = torch.sum(weights.unsqueeze(1) * allowed_directions, dim=0)
        else:
            # If no allowed directions, stay put
            constrained_velocity = torch.zeros_like(desired_velocity)
        
        # Scale velocity for stability
        constrained_velocity = constrained_velocity * 0.5
        
        return constrained_velocity

class SmoothTrajectoryPredictor(nn.Module):
    """Predicts smooth trajectories that respect graph physics"""
    
    def __init__(self, ode_func, num_zones=8, embedding_dim=64):
        super().__init__()
        
        self.ode_func = ode_func
        self.num_zones = num_zones
        
        # Initial position predictor
        self.initial_position_net = nn.Sequential(
            nn.Linear(8, 32),  # person_attrs
            nn.ReLU(),
            nn.Linear(32, num_zones)
        )
        
    def forward(self, person_attrs, times, zone_features, edge_index):
        """Predict smooth trajectory with physics constraints"""
        
        # Set graph data
        self.ode_func.set_graph_data(zone_features, edge_index, person_attrs)
        
        # Predict initial zone and set initial position
        initial_zone_logits = self.initial_position_net(person_attrs)
        initial_zone_idx = torch.argmax(initial_zone_logits).item()
        initial_state = self.ode_func.zone_embeddings[initial_zone_idx]
        
        # Solve ODE for smooth trajectory
        try:
            trajectory = odeint(
                self.ode_func,
                initial_state,
                times,
                method='euler',
                options={'step_size': 0.02}  # Small steps for smooth flow
            )
        except Exception as e:
            print(f"ODE failed: {e}")
            # Fallback: linear interpolation between zones
            trajectory = self._linear_fallback(initial_state, times)
        
        # Convert continuous trajectory to zone predictions
        zone_logits = []
        for state in trajectory:
            # Distance-based soft assignment to zones
            distances = torch.norm(self.ode_func.zone_embeddings - state.unsqueeze(0), dim=1)
            # Convert to probabilities (closer = higher probability)
            logits = -distances * 5.0  # Temperature parameter
            zone_logits.append(logits)
        
        zone_logits = torch.stack(zone_logits)
        return zone_logits, trajectory
    
    def _linear_fallback(self, initial_state, times):
        """Simple fallback if ODE fails"""
        trajectory = [initial_state]
        for i in range(1, len(times)):
            # Just stay at initial position
            trajectory.append(initial_state)
        return torch.stack(trajectory) 