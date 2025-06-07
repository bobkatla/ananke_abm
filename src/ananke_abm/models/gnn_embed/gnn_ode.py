#!/usr/bin/env python3
"""
GNN-Based Physics-Informed ODE Models
Uses Graph Neural Networks to embed locations and people, then applies ODE dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool
from torchdiffeq import odeint
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from .HomoGraph import HomoGraph

class GNNEmbedder(nn.Module):
    """GNN-based embedder for homogeneous graphs"""
    
    def __init__(self, input_dim: int, embedding_dim: int = 64, num_layers: int = 2):
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, embedding_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gnn_layers.append(GCNConv(embedding_dim, embedding_dim))
        
        # Attention layer for context-aware embeddings
        self.attention = GATConv(
            embedding_dim, 
            embedding_dim, 
            heads=4, 
            concat=False,
            dropout=0.1
        )
        
        # Output projection
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, input_dim] node features
            edge_index: [2, num_edges] edge connectivity
        
        Returns:
            node_embeddings: [num_nodes, embedding_dim]
        """
        
        # Project input features
        h = self.input_proj(x)
        h = F.relu(h)
        
        # Apply GNN layers
        for layer in self.gnn_layers:
            h = layer(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, training=self.training)
        
        # Apply attention
        h = self.attention(h, edge_index)
        
        # Final projection
        h = self.output_proj(h)
        
        return h

class GNNPhysicsODE(nn.Module):
    """GNN-based Physics-Informed ODE for agent movement"""
    
    def __init__(self, location_graph: HomoGraph, person_graph: HomoGraph, 
                 embedding_dim: int = 64, num_gnn_layers: int = 2):
        super().__init__()
        
        self.location_graph = location_graph
        self.person_graph = person_graph
        self.embedding_dim = embedding_dim
        
        # Extract schemas
        location_schema = location_graph.extract_schema()
        person_schema = person_graph.extract_schema()
        
        # GNN embedders for locations and people
        self.location_embedder = GNNEmbedder(
            location_schema['node_feature_dim'], embedding_dim, num_gnn_layers
        )
        self.person_embedder = GNNEmbedder(
            person_schema['node_feature_dim'], embedding_dim, num_gnn_layers
        )
        
        # Time encoding
        self.time_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim)
        )
        
        # Flow prediction network
        self.flow_net = nn.Sequential(
            nn.Linear(embedding_dim * 3, 128),  # current_location + person + time
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, embedding_dim)  # Velocity in embedding space
        )
        
        # Zone prediction layer (for discrete predictions)
        self.zone_predictor = nn.Linear(embedding_dim, location_schema['num_nodes'])
        
        # Cache embeddings
        self.location_embeddings = None
        self.person_embeddings = None
        self._compute_embeddings()
        
    def _compute_embeddings(self):
        """Compute and cache graph embeddings"""
        with torch.no_grad():
            # Get location embeddings
            location_data = self.location_graph.get_data()
            self.location_embeddings = self.location_embedder(
                location_data.x, location_data.edge_index
            )
            
            # Get person embeddings
            person_data = self.person_graph.get_data()
            self.person_embeddings = self.person_embedder(
                person_data.x, person_data.edge_index
            )
    
    def get_current_zone_idx(self, state: torch.Tensor) -> int:
        """Find which zone we're currently closest to"""
        distances = torch.norm(self.location_embeddings - state.unsqueeze(0), dim=1)
        return torch.argmin(distances).item()
    
    def get_physics_constraints(self, current_zone_idx: int) -> torch.Tensor:
        """Get allowed movement directions based on graph connectivity - HARD constraints"""
        edge_index = self.location_graph.edge_index
        
        # Find neighboring zones
        neighbors = edge_index[1][edge_index[0] == current_zone_idx]
        
        if len(neighbors) == 0:
            # If isolated, can only stay in current zone
            return torch.tensor([current_zone_idx])
        
        # Include current zone (can stay) and neighbors
        allowed_zones = torch.cat([torch.tensor([current_zone_idx]), neighbors])
        return allowed_zones.unique()
    
    def get_allowed_directions(self, current_zone_idx: int) -> torch.Tensor:
        """Get unit vectors pointing toward allowed adjacent zones"""
        allowed_zones = self.get_physics_constraints(current_zone_idx)
        current_pos = self.location_embeddings[current_zone_idx]
        
        # Compute direction vectors to allowed zones
        directions = []
        for zone_idx in allowed_zones:
            target_pos = self.location_embeddings[zone_idx]
            direction = target_pos - current_pos
            # Normalize direction (unit vector)
            if torch.norm(direction) > 1e-6:
                direction = direction / torch.norm(direction)
            else:
                direction = torch.zeros_like(direction)  # Stay put
            directions.append(direction)
        
        return torch.stack(directions) if directions else torch.zeros(1, self.embedding_dim)
    
    def ode_function(self, t: torch.Tensor, state: torch.Tensor, person_idx: int) -> torch.Tensor:
        """
        ODE function: dx/dt = f(x, t, person)
        
        Args:
            t: current time
            state: current state in embedding space [embedding_dim]
            person_idx: which person this is for
        
        Returns:
            velocity: derivative dx/dt [embedding_dim]
        """
        
        # Handle time encoding
        if isinstance(t, torch.Tensor):
            t_val = t.item() if t.dim() == 0 else t[0].item()
        else:
            t_val = float(t)
        
        t_tensor = torch.tensor([[t_val]], dtype=torch.float32)
        t_encoded = self.time_encoder(t_tensor).squeeze(0)
        
        # Get person embedding
        person_embedding = self.person_embeddings[person_idx]
        
        # Predict desired movement
        combined_input = torch.cat([
            state,                # Current position embedding
            person_embedding,     # Person characteristics
            t_encoded            # Time information
        ])
        
        velocity = self.flow_net(combined_input.unsqueeze(0)).squeeze(0)
        
        # Apply HARD physics constraints (like the working implementation)
        current_zone_idx = self.get_current_zone_idx(state)
        allowed_directions = self.get_allowed_directions(current_zone_idx)
        
        if len(allowed_directions) > 1:
            # PROJECT desired velocity onto allowed directions (HARD CONSTRAINT)
            projections = torch.matmul(allowed_directions, velocity.unsqueeze(1)).squeeze(1)
            
            # Use weighted combination of allowed directions
            weights = F.softmax(projections, dim=0)
            constrained_velocity = torch.sum(weights.unsqueeze(1) * allowed_directions, dim=0)
        else:
            # If no allowed directions, stay put
            constrained_velocity = torch.zeros_like(velocity)
        
        # Scale for stability (same as working implementation)
        return constrained_velocity * 0.5
    
    def predict_trajectory(self, person_idx: int, times: torch.Tensor, 
                         initial_zone_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict trajectory for a person using ODE solver
        
        Args:
            person_idx: index of person in person graph
            times: time points to predict [num_times]
            initial_zone_idx: starting zone index
        
        Returns:
            trajectory: continuous trajectory in embedding space [num_times, embedding_dim]
            zone_predictions: discrete zone predictions [num_times]
        """
        
        # Set initial state
        initial_state = self.location_embeddings[initial_zone_idx]
        
        # Define ODE function for this specific person
        def ode_func(t, state):
            return self.ode_function(t, state, person_idx)
        
        # Solve ODE using torchdiffeq
        try:
            trajectory = odeint(
                ode_func,
                initial_state,
                times,
                method='euler',      # Simple Euler like working implementation
                options={'step_size': 0.02}  # Small steps for smooth flow
            )
            
        except Exception as e:
            print(f"ODE solver failed: {e}, using fallback")
            # Fallback to simple integration if ODE fails
            trajectory = []
            current_state = initial_state
            
            for i, t in enumerate(times):
                if i > 0:
                    dt = (times[i] - times[i-1]).item()
                    velocity = self.ode_function(t, current_state, person_idx)
                    current_state = current_state + velocity * dt
                
                trajectory.append(current_state)
            
            trajectory = torch.stack(trajectory)
        
        # Convert continuous trajectory to discrete zone predictions
        zone_predictions = []
        for state in trajectory:
            zone_idx = self.get_current_zone_idx(state)
            zone_predictions.append(zone_idx)
        
        zone_predictions = torch.tensor(zone_predictions, dtype=torch.long)
        
        return trajectory, zone_predictions
    
    def compute_loss(self, trajectories: Dict, loss_type: str = "mse") -> torch.Tensor:
        """
        Compute loss against observed trajectories
        
        Args:
            trajectories: dict with person trajectory data
            loss_type: "mse" for continuous or "ce" for discrete
        
        Returns:
            loss: scalar loss tensor
        """
        
        total_loss = 0.0
        num_trajectories = 0
        
        for person_name, traj_data in trajectories.items():
            person_id = traj_data["person_id"]
            times = traj_data["times"]
            observed_zones = traj_data["zones"]
            
            # Find person index in graph
            person_node_ids = self.person_graph.map_idx_to_node_id
            person_idx = person_node_ids.index(person_id)
            
            # Convert zone IDs to indices (zones are already 0-indexed from mock data)
            zone_indices = observed_zones  # No conversion needed!
            initial_zone_idx = zone_indices[0].item()
            
            # Predict trajectory
            try:
                pred_trajectory, pred_zones = self.predict_trajectory(
                    person_idx, times, initial_zone_idx
                )
            except Exception as e:
                print(f"        ERROR predicting trajectory: {e}")
                # Use a simple fallback loss
                return torch.tensor(1.0, requires_grad=True)
            
            if loss_type == "mse":
                # Continuous loss in embedding space
                target_embeddings = self.location_embeddings[zone_indices]
                loss = F.mse_loss(pred_trajectory, target_embeddings)
            else:  # "ce"
                # Discrete classification loss (stronger gradients like working implementation)
                zone_logits = self.zone_predictor(pred_trajectory)
                loss = F.cross_entropy(zone_logits, zone_indices)
            
            total_loss += loss
            num_trajectories += 1
        
        final_loss = total_loss / num_trajectories if num_trajectories > 0 else torch.tensor(0.0)
        return final_loss

class GNNODETrainer:
    """Trainer for GNN-ODE model"""
    
    def __init__(self, model: GNNPhysicsODE, lr: float = 0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
    def train_step(self, trajectories: Dict) -> float:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        loss = self.model.compute_loss(trajectories, loss_type="ce")
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, trajectories: Dict, num_epochs: int = 100, verbose: bool = True):
        """Train the model"""
        
        for epoch in range(num_epochs):
            loss = self.train_step(trajectories)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.6f}")
            
            self.scheduler.step(loss)
            
            # Early stopping
            if loss < 1e-6:
                print(f"Converged at epoch {epoch+1}")
                break
    
    def evaluate(self, trajectories: Dict) -> Dict:
        """Evaluate model performance"""
        self.model.eval()
        
        results = {}
        
        with torch.no_grad():
            for person_name, traj_data in trajectories.items():
                person_id = traj_data["person_id"]
                times = traj_data["times"]
                observed_zones = traj_data["zones"]
                
                # Find person index
                person_node_ids = self.model.person_graph.map_idx_to_node_id
                person_idx = person_node_ids.index(person_id)
                
                # Predict
                initial_zone_idx = observed_zones[0].item()
                pred_trajectory, pred_zones = self.model.predict_trajectory(
                    person_idx, times, initial_zone_idx
                )
                
                # Convert back to same indexing for comparison
                pred_zones_corrected = pred_zones
                
                # Compute accuracy
                accuracy = (pred_zones_corrected == observed_zones).float().mean().item()
                
                results[person_name] = {
                    "predicted_zones": pred_zones_corrected,
                    "observed_zones": observed_zones,
                    "accuracy": accuracy,
                    "times": times
                }
        
        return results
